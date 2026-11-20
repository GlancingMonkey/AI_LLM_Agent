"""
페어플 (FairPle) — 24시간 밀봉 역경매 입찰, 즉시 정산 및 GAS 자동 마감 라우터 (server/routers/auction.py)
참조: prd_02_core_features.md [F-02], prd_03_nfr_and_architecture.md 5.6절
- POST /api/rooms/{code}/bids
- POST /api/rooms/{code}/settle
- POST /api/internal/close-auctions
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from deps import (
    UserProfile,
    get_current_user,
    local_store,
    supabase_rest_request,
    verify_gas_proxy_secret,
)
from routers.rooms import serialize_room_detail
from services.auction_engine import settle_reverse_auction, validate_bid_points

router = APIRouter(
    tags=["auction"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class BidEntryItem(BaseModel):
    task_id: str
    bid_points: int = Field(..., gt=0)


class SubmitBidsRequest(BaseModel):
    bids: List[BidEntryItem] = Field(..., min_length=1)


def _execute_room_settlement(room: Dict[str, Any]) -> Dict[str, Any]:
    room_id = room["id"]
    tasks = [t for t in local_store.tasks_by_id.values() if t["room_id"] == room_id]
    members = local_store.room_members.get(room_id, [])
    bids = local_store.bids_by_room.get(room_id, [])

    settlement = settle_reverse_auction(
        tasks=tasks,
        members=members,
        bids=bids,
        seed_key=room["invite_code"],
    )

    for assign in settlement["assignments"]:
        tid = assign["task_id"]
        if tid in local_store.tasks_by_id:
            t_obj = local_store.tasks_by_id[tid]
            t_obj["assignee_id"] = assign["assignee_id"]
            t_obj["assignee_name"] = assign["assignee_name"]
            t_obj["awarded_points"] = assign["awarded_points"]
            if t_obj["status"] == "UNASSIGNED":
                t_obj["status"] = "ASSIGNED"

            if settings.is_supabase_configured:
                try:
                    supabase_rest_request(
                        "PATCH",
                        "tasks",
                        query=f"id=eq.{tid}",
                        payload={
                            "assignee_id": assign["assignee_id"],
                            "awarded_points": assign["awarded_points"],
                            "status": t_obj["status"],
                        },
                    )
                except Exception:
                    pass

    room["status"] = "IN_PROGRESS"
    room["last_settlement"] = settlement

    if settings.is_supabase_configured:
        try:
            supabase_rest_request(
                "PATCH",
                "rooms",
                query=f"id=eq.{room_id}",
                payload={"status": "IN_PROGRESS"},
            )
        except Exception:
            pass

    return settlement


@router.post("/api/rooms/{code}/bids")
def submit_bids(
    code: str,
    req: SubmitBidsRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    clean_code = code.strip().upper()
    room = local_store.rooms_by_code.get(clean_code)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"초대코드 '{clean_code}'에 해당하는 방을 찾을 수 없습니다.",
        )

    room_id = room["id"]
    members = local_store.room_members.setdefault(room_id, [])
    member_doc = next((m for m in members if m["user_id"] == current_user.id), None)
    if not member_doc:
        member_doc = {
            "id": str(uuid.uuid4()),
            "room_id": room_id,
            "user_id": current_user.id,
            "display_name": current_user.name,
            "has_submitted_bid": False,
            "joined_at": datetime.now(timezone.utc).isoformat(),
        }
        members.append(member_doc)

    room_tasks = {
        t["id"]: t
        for t in local_store.tasks_by_id.values()
        if t["room_id"] == room_id
    }

    for entry in req.bids:
        task_obj = room_tasks.get(entry.task_id)
        if not task_obj:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"유효하지 않은 과업 ID({entry.task_id})입니다.",
            )
        try:
            validate_bid_points(
                base_points=int(task_obj["base_points"]),
                bid_points=entry.bid_points,
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            )

    # 기존 본인 입찰 갱신
    room_bids = local_store.bids_by_room.setdefault(room_id, [])
    local_store.bids_by_room[room_id] = [
        b for b in room_bids if b["user_id"] != current_user.id
    ]
    now_iso = datetime.now(timezone.utc).isoformat()
    for entry in req.bids:
        bid_doc = {
            "id": str(uuid.uuid4()),
            "room_id": room_id,
            "task_id": entry.task_id,
            "user_id": current_user.id,
            "bid_points": entry.bid_points,
            "created_at": now_iso,
        }
        local_store.bids_by_room[room_id].append(bid_doc)
        if settings.is_supabase_configured:
            try:
                supabase_rest_request(
                    "POST",
                    "bids",
                    payload=bid_doc,
                    extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
                )
            except Exception:
                pass

    member_doc["has_submitted_bid"] = True
    if settings.is_supabase_configured:
        try:
            supabase_rest_request(
                "PATCH",
                "room_members",
                query=f"room_id=eq.{room_id}&user_id=eq.{current_user.id}",
                payload={"has_submitted_bid": True},
            )
        except Exception:
            pass

    # 전원 입찰 완료 여부 확인 -> 전원 제출 시 24시간 전이라도 즉시 정산·개봉
    all_submitted = (
        len(members) >= int(room["member_limit"])
        and all(m.get("has_submitted_bid") for m in members)
    )
    settlement = None
    if all_submitted:
        settlement = _execute_room_settlement(room)

    return {
        "status": "success",
        "all_submitted": all_submitted,
        "settlement": settlement,
        **serialize_room_detail(room, viewer=current_user),
    }


@router.post("/api/rooms/{code}/settle")
def settle_room_auction(
    code: str,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    clean_code = code.strip().upper()
    room = local_store.rooms_by_code.get(clean_code)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"초대코드 '{clean_code}'에 해당하는 방을 찾을 수 없습니다.",
        )

    settlement = _execute_room_settlement(room)
    return {
        "status": "success",
        "settlement": settlement,
        **serialize_room_detail(room, viewer=current_user),
    }


@router.post("/api/internal/close-auctions")
def close_expired_auctions() -> Dict[str, Any]:
    """GAS 10분 주기 시간 트리거(autoCloseExpiredAuctions)가 호출하는 만료 경매 일괄 자동 마감"""
    now_iso = datetime.now(timezone.utc).isoformat()
    closed_codes: List[str] = []

    for code, room in list(local_store.rooms_by_code.items()):
        if room.get("status") == "AUCTION_OPEN" and not room.get("is_archived", False):
            ends_at = room.get("auction_ends_at") or ""
            if ends_at and ends_at <= now_iso:
                _execute_room_settlement(room)
                closed_codes.append(code)

    return {
        "status": "success",
        "closed_count": len(closed_codes),
        "closed_rooms": closed_codes,
        "checked_at": now_iso,
    }
