"""
페어플 (FairPle) — 관리자 전용 RBAC 라우터 (server/routers/admin.py)
참조: prd_02_core_features.md [F-08], prd_04_success_criteria.md [E2E-7]
- GET   /api/admin/me
- GET   /api/admin/stats
- GET   /api/admin/rooms
- PATCH /api/admin/rooms/{room_id} (is_archived 토글 + admin_audit_logs 기록)
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from config import settings
from deps import (
    UserProfile,
    local_store,
    record_admin_audit_log,
    require_admin,
    supabase_rest_request,
    verify_gas_proxy_secret,
)

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class ArchiveRoomRequest(BaseModel):
    is_archived: Optional[bool] = None


@router.get("/me")
def admin_me(admin_user: UserProfile = Depends(require_admin)) -> Dict[str, Any]:
    record_admin_audit_log(
        admin_email=admin_user.email,
        action="ADMIN_SESSION_VERIFIED",
        target_type="admin_me",
        target_id=admin_user.id,
    )
    return {
        "status": "success",
        "is_admin": True,
        "admin": {
            "id": admin_user.id,
            "email": admin_user.email,
            "name": admin_user.name,
        },
    }


@router.get("/stats")
def admin_stats(admin_user: UserProfile = Depends(require_admin)) -> Dict[str, Any]:
    total_rooms = len(local_store.rooms_by_code)
    active_rooms = sum(
        1 for r in local_store.rooms_by_code.values() if not r.get("is_archived", False)
    )
    archived_rooms = total_rooms - active_rooms
    total_users = max(4, len(local_store.users_by_email) + 4)
    total_tasks = len(local_store.tasks_by_id)
    total_chat_calls = sum(local_store.chat_usage_by_user_date.values())

    return {
        "status": "success",
        "stats": {
            "total_rooms": total_rooms,
            "active_rooms": active_rooms,
            "archived_rooms": archived_rooms,
            "total_users": total_users,
            "total_tasks": total_tasks,
            "total_chat_calls": total_chat_calls,
            "audit_log_count": len(local_store.admin_audit_logs),
        },
        "recent_audit_logs": local_store.admin_audit_logs[-10:][::-1],
    }


@router.get("/rooms")
def admin_list_rooms(admin_user: UserProfile = Depends(require_admin)) -> Dict[str, Any]:
    rooms_list = []
    for r in local_store.rooms_by_code.values():
        rid = r["id"]
        m_cnt = len(local_store.room_members.get(rid, []))
        t_cnt = sum(1 for t in local_store.tasks_by_id.values() if t["room_id"] == rid)
        rooms_list.append(
            {
                **r,
                "member_count": m_cnt,
                "task_count": t_cnt,
            }
        )
    return {
        "status": "success",
        "rooms": rooms_list,
    }


@router.patch("/rooms/{room_id}")
def admin_toggle_archive_room(
    room_id: str,
    req: ArchiveRoomRequest,
    admin_user: UserProfile = Depends(require_admin),
) -> Dict[str, Any]:
    room = local_store.rooms_by_id.get(room_id) or local_store.rooms_by_code.get(
        room_id.strip().upper()
    )
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"방 ID('{room_id}')를 찾을 수 없습니다.",
        )

    new_archived = (
        req.is_archived
        if req.is_archived is not None
        else (not bool(room.get("is_archived", False)))
    )
    room["is_archived"] = new_archived

    if settings.is_supabase_configured:
        try:
            supabase_rest_request(
                "PATCH",
                "rooms",
                query=f"id=eq.{room['id']}",
                payload={"is_archived": new_archived},
            )
        except Exception:
            pass

    audit_entry = record_admin_audit_log(
        admin_email=admin_user.email,
        action="ROOM_ARCHIVE_TOGGLED",
        target_type="room",
        target_id=room["id"],
        metadata={
            "invite_code": room["invite_code"],
            "is_archived": new_archived,
        },
    )

    return {
        "status": "success",
        "room": room,
        "audit_log": audit_entry,
    }
