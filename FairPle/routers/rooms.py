"""
페어플 (FairPle) — 방 생성, 목록 조회, 상세 조회 및 팀원 합류 라우터 (server/routers/rooms.py)
참조: prd_02_core_features.md [F-01], prd_03_nfr_and_architecture.md 5.6절
- POST /api/rooms
- GET  /api/rooms
- GET  /api/rooms/{code}
- POST /api/rooms/{code}/join
"""

import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from deps import (
    UserProfile,
    get_current_user,
    get_optional_user,
    local_store,
    supabase_rest_request,
    verify_gas_proxy_secret,
)
from services.scoring import STANDARD_TASK_CATALOG, calculate_base_points

router = APIRouter(
    prefix="/api/rooms",
    tags=["rooms"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class TaskInputItem(BaseModel):
    title: str
    category: str = "research"
    unit_label: Optional[str] = None
    quantity: int = Field(..., gt=0)
    custom_base_points: Optional[int] = None
    requirements: Optional[str] = ""
    due_at: Optional[str] = None


class CreateRoomRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=120)
    member_limit: int = Field(default=4, ge=2, le=10)
    final_due_at: Optional[str] = None
    auction_duration_hours: int = Field(default=24, ge=1, le=72)
    tasks: List[TaskInputItem] = Field(..., min_length=1)


class JoinRoomRequest(BaseModel):
    display_name: Optional[str] = None


def _generate_invite_code() -> str:
    prefix = "".join(random.choices(string.ascii_uppercase, k=4))
    digits = "".join(random.choices(string.digits, k=3))
    code = f"{prefix}-{digits}"
    while code in local_store.rooms_by_code:
        prefix = "".join(random.choices(string.ascii_uppercase, k=4))
        digits = "".join(random.choices(string.digits, k=3))
        code = f"{prefix}-{digits}"
    return code


def serialize_room_detail(room: Dict[str, Any], viewer: Optional[UserProfile] = None) -> Dict[str, Any]:
    room_id = room["id"]
    members = local_store.room_members.get(room_id, [])
    tasks = [t for t in local_store.tasks_by_id.values() if t["room_id"] == room_id]
    takeovers = local_store.takeover_logs_by_room.get(room_id, [])
    room_bids = local_store.bids_by_room.get(room_id, [])

    # 각 과업에 최신 제출물 및 승인 현황 첨부
    enriched_tasks = []
    n_members = max(2, len(members))
    majority_threshold = max(1, (n_members - 1 + 1) // 2)  # ceil((N-1)/2)

    now_iso = datetime.now(timezone.utc).isoformat()
    for t in tasks:
        t_copy = dict(t)
        t_subs = [
            s for s in local_store.submissions_by_id.values() if s["task_id"] == t["id"]
        ]
        latest_sub = t_subs[-1] if t_subs else None
        approvals = (
            local_store.peer_approvals_by_sub.get(latest_sub["id"], [])
            if latest_sub
            else []
        )
        approve_cnt = sum(1 for a in approvals if a["decision"] == "APPROVED")
        revision_cnt = sum(1 for a in approvals if a["decision"] == "REVISION_REQUESTED")

        # 마감 경과 여부 판정
        due_str = t.get("due_at") or ""
        is_overdue = bool(
            due_str
            and due_str < now_iso
            and t.get("status") not in ("IN_REVIEW", "APPROVED")
        )

        t_copy["latest_submission"] = latest_sub
        t_copy["approvals"] = approvals
        t_copy["approved_count"] = approve_cnt
        t_copy["revision_count"] = revision_cnt
        t_copy["required_approvals"] = majority_threshold
        t_copy["is_overdue"] = is_overdue
        t_copy["can_takeover"] = is_overdue or bool(t.get("voluntary_takeover_open"))
        enriched_tasks.append(t_copy)

    # [보안 핵심] AUCTION_OPEN 상태에서는 본인 입찰가만 반환하고 타인 입찰가(bid_points)는 절대 미노출!
    my_bids: Dict[str, int] = {}
    if viewer:
        for b in room_bids:
            if b["user_id"] == viewer.id:
                my_bids[b["task_id"]] = b["bid_points"]

    return {
        "room": room,
        "members": members,
        "tasks": enriched_tasks,
        "my_bids": my_bids,
        "takeover_logs": takeovers,
        "majority_threshold": majority_threshold,
    }


@router.post("")
def create_room(
    req: CreateRoomRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    room_id = str(uuid.uuid4())
    invite_code = _generate_invite_code()

    validated_tasks: List[Dict[str, Any]] = []
    for item in req.tasks:
        cat_key = (item.category or "custom").strip().lower()
        unit_label = item.unit_label
        if not unit_label and cat_key in STANDARD_TASK_CATALOG:
            unit_label = str(STANDARD_TASK_CATALOG[cat_key]["unit_label"])

        try:
            base_pts = calculate_base_points(
                category=cat_key,
                quantity=item.quantity,
                custom_base_points=item.custom_base_points,
                unit_label=unit_label,
                title=item.title,
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            )

        task_id = str(uuid.uuid4())
        due_at = item.due_at or (now + timedelta(days=3)).isoformat()
        validated_tasks.append(
            {
                "id": task_id,
                "room_id": room_id,
                "title": item.title.strip(),
                "category": cat_key,
                "unit_label": unit_label or "건",
                "quantity": item.quantity,
                "base_points": base_pts,
                "awarded_points": base_pts,
                "assignee_id": None,
                "assignee_name": "미배정 (경매 중)",
                "status": "UNASSIGNED",
                "requirements": (item.requirements or "").strip(),
                "due_at": due_at,
                "created_at": now.isoformat(),
            }
        )

    auction_ends = (now + timedelta(hours=req.auction_duration_hours)).isoformat()
    final_due = req.final_due_at or (now + timedelta(days=7)).isoformat()

    room_doc = {
        "id": room_id,
        "invite_code": invite_code,
        "title": req.title.strip(),
        "member_limit": req.member_limit,
        "status": "AUCTION_OPEN",
        "auction_ends_at": auction_ends,
        "final_due_at": final_due,
        "is_archived": False,
        "created_by": current_user.id,
        "created_at": now.isoformat(),
    }

    creator_member = {
        "id": str(uuid.uuid4()),
        "room_id": room_id,
        "user_id": current_user.id,
        "display_name": current_user.name,
        "has_submitted_bid": False,
        "joined_at": now.isoformat(),
    }

    local_store.rooms_by_code[invite_code] = room_doc
    local_store.rooms_by_id[room_id] = room_doc
    local_store.room_members[room_id] = [creator_member]
    local_store.bids_by_room[room_id] = []
    local_store.takeover_logs_by_room[room_id] = []
    for t in validated_tasks:
        local_store.tasks_by_id[t["id"]] = t

    if settings.is_supabase_configured:
        try:
            supabase_rest_request("POST", "rooms", payload=room_doc)
            supabase_rest_request("POST", "room_members", payload=creator_member)
            for t in validated_tasks:
                t_db = {k: v for k, v in t.items() if k != "assignee_name"}
                supabase_rest_request("POST", "tasks", payload=t_db)
        except Exception:
            pass

    return {
        "status": "success",
        **serialize_room_detail(room_doc, viewer=current_user),
    }


@router.get("")
def list_rooms(
    current_user: Optional[UserProfile] = Depends(get_optional_user),
) -> Dict[str, Any]:
    rooms = [
        r for r in local_store.rooms_by_code.values() if not r.get("is_archived", False)
    ]
    return {
        "status": "success",
        "rooms": rooms,
    }


@router.get("/{code}")
def get_room_by_code(
    code: str,
    current_user: Optional[UserProfile] = Depends(get_optional_user),
) -> Dict[str, Any]:
    clean_code = code.strip().upper()
    room = local_store.rooms_by_code.get(clean_code)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"초대코드 '{clean_code}'에 해당하는 팀플 방을 찾을 수 없습니다.",
        )
    return {
        "status": "success",
        **serialize_room_detail(room, viewer=current_user),
    }


@router.post("/{code}/join")
def join_room(
    code: str,
    req: JoinRoomRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    clean_code = code.strip().upper()
    room = local_store.rooms_by_code.get(clean_code)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"초대코드 '{clean_code}'에 해당하는 방이 없습니다.",
        )

    room_id = room["id"]
    members = local_store.room_members.setdefault(room_id, [])
    existing = next((m for m in members if m["user_id"] == current_user.id), None)
    if existing:
        return {
            "status": "success",
            "message": "이미 참여 중인 방입니다.",
            **serialize_room_detail(room, viewer=current_user),
        }

    if len(members) >= int(room["member_limit"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="방 정원이 가득 찼습니다.",
        )

    new_member = {
        "id": str(uuid.uuid4()),
        "room_id": room_id,
        "user_id": current_user.id,
        "display_name": (req.display_name or current_user.name).strip(),
        "has_submitted_bid": False,
        "joined_at": datetime.now(timezone.utc).isoformat(),
    }
    members.append(new_member)

    if settings.is_supabase_configured:
        try:
            supabase_rest_request("POST", "room_members", payload=new_member)
        except Exception:
            pass

    return {
        "status": "success",
        **serialize_room_detail(room, viewer=current_user),
    }
