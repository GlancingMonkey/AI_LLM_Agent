"""
페어플 (FairPle) — Gemini AI 서비스·포트폴리오 도우미 챗봇 라우터 (server/routers/chat.py)
참조: prd_02_core_features.md [F-07], prd_04_success_criteria.md [E2E-6]
- POST /api/chat
- 일일 사용자당 최대 20회 제한 (21번째 요청 429 차단)
- 단일 질문 최대 300자 제한
- 범위 외 질문 입력 시 서버 고정 상수 문구(OUT_OF_SCOPE_REFUSAL_MESSAGE)만 반환
"""

from datetime import date
from typing import Any, Dict, List, Optional

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
from services.gemini_guardrail import (
    OUT_OF_SCOPE_REFUSAL_MESSAGE,
    generate_guarded_reply,
)

DAILY_CHAT_LIMIT = 20
MAX_MESSAGE_LENGTH = 300

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    room_code: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


@router.post("")
def chat_with_copilot(
    req: ChatRequest,
    current_user: UserProfile = Depends(get_current_user),
) -> Dict[str, Any]:
    msg = (req.message or "").strip()
    if len(msg) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"질문은 최대 {MAX_MESSAGE_LENGTH}자 이내로 입력해 주세요 (현재 {len(msg)}자).",
        )

    today_str = date.today().isoformat()
    usage_key = f"{current_user.id}:{today_str}"
    current_count = local_store.chat_usage_by_user_date.get(usage_key, 0)

    if current_count >= DAILY_CHAT_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"일일 AI 챗봇 사용 한도({DAILY_CHAT_LIMIT}회)를 초과했습니다. 내일 다시 이용해 주세요.",
        )

    # 사용량 1 증가 기록
    new_count = current_count + 1
    local_store.chat_usage_by_user_date[usage_key] = new_count

    if settings.is_supabase_configured:
        try:
            supabase_rest_request(
                "POST",
                "chat_usage_logs",
                payload={
                    "user_id": current_user.id,
                    "usage_date": today_str,
                    "count": new_count,
                },
                extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
        except Exception:
            pass

    room_ctx = None
    if req.room_code:
        room_ctx = local_store.rooms_by_code.get(req.room_code.strip().upper())

    result = generate_guarded_reply(
        user_message=msg,
        room_context=room_ctx,
        history=(req.history or [])[-6:],
    )

    is_in_scope = bool(result.get("is_in_scope", False))
    reply_text = (
        str(result.get("reply") or "").strip()
        if is_in_scope
        else OUT_OF_SCOPE_REFUSAL_MESSAGE
    )
    if not is_in_scope:
        reply_text = OUT_OF_SCOPE_REFUSAL_MESSAGE

    return {
        "status": "success",
        "is_in_scope": is_in_scope,
        "reply": reply_text,
        "usage": {
            "used_today": new_count,
            "daily_limit": DAILY_CHAT_LIMIT,
            "remaining": max(0, DAILY_CHAT_LIMIT - new_count),
        },
    }
