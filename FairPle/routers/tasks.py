"""
페어플 (FairPle) — 산출물 제출, 과반수 1클릭 확인 및 실시간 업무 인수 라우터 (server/routers/tasks.py)
참조: prd_02_core_features.md [F-03], [F-04], prd_03_nfr_and_architecture.md 5.6절
- POST /api/tasks/{id}/submit
- POST /api/submissions/{id}/approve
- POST /api/tasks/{id}/takeover
"""

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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

router = APIRouter(
    tags=["tasks"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class SubmitDeliverableRequest(BaseModel):
    external_url: str = Field(..., min_length=5, max_length=500)
    summary: str = Field(..., min_length=2, max_length=1000)
    proof_image_path: Optional[str] = None
    proof_image_size_bytes: Optional[int] = Field(default=None, le=3 * 1024 * 1024)


class PeerApprovalRequest(BaseModel):
    decision: str = Field(default="APPROVED", pattern="^(APPROVED|REVISION_REQUESTED)$")
    reason_code: Optional[str] = None


class TaskTakeoverRequest(BaseModel):
    trigger_type: str = Field(
        default="OVERDUE",
        pattern="^(OVERDUE|MAJORITY_CONSENT|VOLUNTARY)$",
    )
    has_majority_consent: bool = False


@router.post("/api/tasks/{task_id}/submit")
def submit_task_deliverable(
    task_id: str,
    req: SubmitDeliverableRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    task_obj = local_store.tasks_by_id.get(task_id)
    if not task_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"과업 ID('{task_id}')를 찾을 수 없습니다.",
        )

    url_clean = req.external_url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="산출물 링크(external_url)는 http:// 또는 https://로 시작해야 합니다.",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    due_at = task_obj.get("due_at") or ""
    is_late = bool(due_at and now_iso > due_at)

    sub_id = str(uuid.uuid4())
    sub_doc = {
        "id": sub_id,
        "task_id": task_id,
        "submitted_by": current_user.id,
        "submitted_by_name": current_user.name,
        "external_url": url_clean,
        "summary": req.summary.strip(),
        "proof_image_path": req.proof_image_path,
        "is_late": is_late,
        "submitted_at": now_iso,
    }
    local_store.submissions_by_id[sub_id] = sub_doc
    local_store.peer_approvals_by_sub[sub_id] = []

    task_obj["status"] = "IN_REVIEW"
    if not task_obj.get("assignee_id"):
        task_obj["assignee_id"] = current_user.id
        task_obj["assignee_name"] = current_user.name

    if settings.is_supabase_configured:
        try:
            sub_db = {k: v for k, v in sub_doc.items() if k != "submitted_by_name"}
            supabase_rest_request("POST", "submissions", payload=sub_db)
            supabase_rest_request(
                "PATCH",
                "tasks",
                query=f"id=eq.{task_id}",
                payload={"status": "IN_REVIEW"},
            )
        except Exception:
            pass

    return {
        "status": "success",
        "submission": sub_doc,
        "task": task_obj,
    }


@router.post("/api/submissions/{submission_id}/approve")
def review_submission(
    submission_id: str,
    req: PeerApprovalRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    sub_doc = local_store.submissions_by_id.get(submission_id)
    if not sub_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"제출물 ID('{submission_id}')를 찾을 수 없습니다.",
        )

    task_obj = local_store.tasks_by_id.get(sub_doc["task_id"])
    if not task_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="연결된 과업을 찾을 수 없습니다.",
        )

    if current_user.id == sub_doc["submitted_by"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="제출자 본인은 자신의 산출물을 직접 승인할 수 없습니다.",
        )

    room_id = task_obj["room_id"]
    members = local_store.room_members.get(room_id, [])
    n_members = max(2, len(members))
    majority_threshold = max(1, math.ceil((n_members - 1) / 2))

    approvals = local_store.peer_approvals_by_sub.setdefault(submission_id, [])
    approvals = [a for a in approvals if a["reviewer_id"] != current_user.id]

    approval_doc = {
        "id": str(uuid.uuid4()),
        "submission_id": submission_id,
        "reviewer_id": current_user.id,
        "reviewer_name": current_user.name,
        "decision": req.decision,
        "reason_code": req.reason_code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    approvals.append(approval_doc)
    local_store.peer_approvals_by_sub[submission_id] = approvals

    approved_count = sum(1 for a in approvals if a["decision"] == "APPROVED")
    revision_count = sum(1 for a in approvals if a["decision"] == "REVISION_REQUESTED")

    if approved_count >= majority_threshold:
        task_obj["status"] = "APPROVED"
    elif revision_count >= majority_threshold:
        task_obj["status"] = "REVISION_REQUESTED"

    if settings.is_supabase_configured:
        try:
            app_db = {k: v for k, v in approval_doc.items() if k != "reviewer_name"}
            supabase_rest_request(
                "POST",
                "peer_approvals",
                payload=app_db,
                extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
            supabase_rest_request(
                "PATCH",
                "tasks",
                query=f"id=eq.{task_obj['id']}",
                payload={"status": task_obj["status"]},
            )
        except Exception:
            pass

    return {
        "status": "success",
        "task_status": task_obj["status"],
        "approved_count": approved_count,
        "revision_count": revision_count,
        "required_approvals": majority_threshold,
        "task": task_obj,
    }


@router.post("/api/tasks/{task_id}/takeover")
def takeover_task(
    task_id: str,
    req: TaskTakeoverRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    [F-04] 실시간 업무 인수 및 기여도 100% 통째 이관
    악용 차단 조건:
    1. 개별 마감일(due_at)이 경과했음에도 IN_REVIEW/APPROVED가 아닌 경우 (OVERDUE)
    2. 기존 담당자가 포기 요청(VOLUNTARY)했거나 나머지 팀원 과반수가 동의(MAJORITY_CONSENT)한 경우
    """
    task_obj = local_store.tasks_by_id.get(task_id)
    if not task_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"과업 ID('{task_id}')를 찾을 수 없습니다.",
        )

    if task_obj.get("status") == "APPROVED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 승인 완료된 과업은 인수할 수 없습니다.",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    due_at = task_obj.get("due_at") or ""
    is_overdue = bool(
        due_at
        and due_at < now_iso
        and task_obj.get("status") not in ("IN_REVIEW", "APPROVED")
    )
    voluntary_open = bool(task_obj.get("voluntary_takeover_open", False))

    allowed = (
        is_overdue
        or voluntary_open
        or (req.trigger_type in ("VOLUNTARY", "MAJORITY_CONSENT") and req.has_majority_consent)
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="마감 기한이 남아 있고 비상 인수 동의 요건이 충족되지 않아 업무를 인수할 수 없습니다 (점수 가로채기 방지).",
        )

    prev_assignee_id = task_obj.get("assignee_id")
    prev_assignee_name = task_obj.get("assignee_name", "기존 담당자")
    transferred_pts = int(task_obj.get("awarded_points") or task_obj.get("base_points") or 0)

    task_obj["assignee_id"] = current_user.id
    task_obj["assignee_name"] = current_user.name
    task_obj["awarded_points"] = transferred_pts
    task_obj["status"] = "ASSIGNED"
    task_obj["due_at"] = (datetime.now(timezone.utc)).isoformat()  # 인수 시점 갱신

    log_doc = {
        "id": str(uuid.uuid4()),
        "room_id": task_obj["room_id"],
        "task_id": task_id,
        "task_title": task_obj["title"],
        "from_user_id": prev_assignee_id,
        "from_user_name": prev_assignee_name,
        "to_user_id": current_user.id,
        "to_user_name": current_user.name,
        "transferred_points": transferred_pts,
        "trigger_type": "OVERDUE" if is_overdue else req.trigger_type,
        "created_at": now_iso,
    }
    local_store.takeover_logs_by_room.setdefault(task_obj["room_id"], []).append(log_doc)

    if settings.is_supabase_configured:
        try:
            supabase_rest_request(
                "PATCH",
                "tasks",
                query=f"id=eq.{task_id}",
                payload={
                    "assignee_id": current_user.id,
                    "awarded_points": transferred_pts,
                    "status": "ASSIGNED",
                },
            )
            log_db = {
                k: v
                for k, v in log_doc.items()
                if k not in ("task_title", "from_user_name", "to_user_name")
            }
            supabase_rest_request("POST", "takeover_logs", payload=log_db)
        except Exception:
            pass

    return {
        "status": "success",
        "takeover_log": log_doc,
        "task": task_obj,
    }
