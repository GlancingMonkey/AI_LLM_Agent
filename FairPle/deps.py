"""
페어플 (FairPle) — 보안 미들웨어, JWT 인증, RBAC 및 통합 데이터 저장소 (server/deps.py)
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from config import settings


class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    is_admin: bool = False
    created_at: Optional[str] = None


# ============================================================================
# 통합 데이터 저장소 (인메모리 + Supabase PostgREST 자동 동기화)
# ============================================================================
class LocalMemoryStore:
    def __init__(self) -> None:
        self.users_by_email: Dict[str, Dict[str, Any]] = {}
        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.rooms_by_code: Dict[str, Dict[str, Any]] = {}
        self.rooms_by_id: Dict[str, Dict[str, Any]] = {}
        self.room_members: Dict[str, List[Dict[str, Any]]] = {}  # room_id -> list
        self.tasks_by_id: Dict[str, Dict[str, Any]] = {}
        self.bids_by_room: Dict[str, List[Dict[str, Any]]] = {}  # room_id -> list
        self.submissions_by_id: Dict[str, Dict[str, Any]] = {}
        self.peer_approvals_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        self.takeover_logs_by_room: Dict[str, List[Dict[str, Any]]] = {}
        self.report_snapshots_by_hash: Dict[str, Dict[str, Any]] = {}
        self.chat_usage_by_user_date: Dict[str, int] = {}
        self.admin_audit_logs: List[Dict[str, Any]] = []
        self.last_supabase_error: Optional[str] = None
        self._seed_default_demo_room()

    def _seed_default_demo_room(self) -> None:
        now = datetime.now(timezone.utc)
        room_id = "00000000-0000-4000-8000-000000000304"
        code = "MGMT-304"

        demo_members = [
            {
                "id": "m-1",
                "room_id": room_id,
                "user_id": "u-minsu",
                "display_name": "김민수",
                "has_submitted_bid": True,
                "joined_at": (now - timedelta(hours=10)).isoformat(),
            },
            {
                "id": "m-2",
                "room_id": room_id,
                "user_id": "u-jieun",
                "display_name": "이지은",
                "has_submitted_bid": True,
                "joined_at": (now - timedelta(hours=9)).isoformat(),
            },
            {
                "id": "m-3",
                "room_id": room_id,
                "user_id": "u-seojun",
                "display_name": "박서준",
                "has_submitted_bid": True,
                "joined_at": (now - timedelta(hours=8)).isoformat(),
            },
            {
                "id": "m-4",
                "room_id": room_id,
                "user_id": "u-hyunwoo",
                "display_name": "최현우",
                "has_submitted_bid": True,
                "joined_at": (now - timedelta(hours=7)).isoformat(),
            },
        ]

        room_doc = {
            "id": room_id,
            "invite_code": code,
            "title": "경영학원론 3조 팀플 (ESG 경영 혁신사례 발표)",
            "member_limit": 4,
            "status": "AUCTION_OPEN",
            "auction_ends_at": (now + timedelta(hours=14, minutes=28)).isoformat(),
            "final_due_at": (now + timedelta(days=5)).isoformat(),
            "is_archived": False,
            "created_by": "u-jieun",
            "created_at": (now - timedelta(hours=10)).isoformat(),
        }
        self.rooms_by_code[code] = room_doc
        self.rooms_by_id[room_id] = room_doc
        self.room_members[room_id] = demo_members

        t1_id = "task-mgmt-01"
        t2_id = "task-mgmt-02"
        t3_id = "task-mgmt-03"
        t4_id = "task-mgmt-04"

        demo_tasks = [
            {
                "id": t1_id,
                "room_id": room_id,
                "title": "국내외 ESG 경영 사례 3건 분석 리서치",
                "category": "research",
                "unit_label": "건",
                "quantity": 3,
                "base_points": 15,
                "awarded_points": 18,
                "assignee_id": "u-minsu",
                "assignee_name": "김민수",
                "status": "IN_REVIEW",
                "requirements": "글로벌 2곳, 국내 1곳 성공사례 심층 분석 및 출처 URL 필수 기재",
                "due_at": (now + timedelta(days=2)).isoformat(),
                "created_at": now.isoformat(),
            },
            {
                "id": t2_id,
                "room_id": room_id,
                "title": "발표용 PPT 슬라이드 10장 제작 및 디자인 통일",
                "category": "ppt",
                "unit_label": "장",
                "quantity": 10,
                "base_points": 40,
                "awarded_points": 40,
                "assignee_id": "u-seojun",
                "assignee_name": "박서준",
                "status": "ASSIGNED",
                "requirements": "16:9 와이드 템플릿 적용 및 인포그래픽 다이어그램 10장 완성본",
                "due_at": (now - timedelta(hours=6)).isoformat(),  # 마감 6시간 경과 (인수 가능 상태!)
                "created_at": now.isoformat(),
            },
            {
                "id": t3_id,
                "room_id": room_id,
                "title": "최종 발표 스크립트 작성 및 5분 실전 발표",
                "category": "script",
                "unit_label": "분",
                "quantity": 5,
                "base_points": 45,
                "awarded_points": 27,
                "assignee_id": "u-hyunwoo",
                "assignee_name": "최현우",
                "status": "APPROVED",
                "requirements": "A4 3매 분량 낭독 대본 작성 및 5분 실전 스피치 + Q&A 대응",
                "due_at": (now + timedelta(days=3)).isoformat(),
                "created_at": now.isoformat(),
            },
            {
                "id": t4_id,
                "room_id": room_id,
                "title": "팀플 주제 선정 및 발표 목차 구조화 (기획 파트)",
                "category": "research",
                "unit_label": "건",
                "quantity": 3,
                "base_points": 15,
                "awarded_points": 15,
                "assignee_id": "u-jieun",
                "assignee_name": "이지은",
                "status": "APPROVED",
                "requirements": "전체 발표 흐름(서론-본론-결론) 정의 및 세부 아젠다 수립",
                "due_at": (now - timedelta(days=1)).isoformat(),
                "created_at": now.isoformat(),
            },
        ]
        for t in demo_tasks:
            self.tasks_by_id[t["id"]] = t

        sub1_id = "sub-mgmt-01"
        self.submissions_by_id[sub1_id] = {
            "id": sub1_id,
            "task_id": t1_id,
            "submitted_by": "u-minsu",
            "external_url": "https://docs.google.com/document/d/demo-esg-research",
            "summary": "애플 탄소중립 공급망, 파타고니아 순환경제, SK E&S 수소 포트폴리오 3건 비교 요약본(12p)",
            "proof_image_path": None,
            "is_late": False,
            "submitted_at": (now - timedelta(hours=1)).isoformat(),
        }
        self.peer_approvals_by_sub[sub1_id] = [
            {
                "id": "app-1",
                "submission_id": sub1_id,
                "reviewer_id": "u-hyunwoo",
                "decision": "APPROVED",
                "reason_code": None,
                "created_at": (now - timedelta(minutes=30)).isoformat(),
            }
        ]
        self.bids_by_room[room_id] = []
        self.takeover_logs_by_room[room_id] = []


local_store = LocalMemoryStore()


# ============================================================================
# 경량 HMAC-SHA256 JWT 헬퍼
# ============================================================================
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user_id: str, email: str, name: str, expires_in: int = 86400 * 7) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email.lower().strip(),
        "name": name,
        "iat": now,
        "exp": now + expires_in,
    }
    h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = hmac.new(
        settings.LOCAL_JWT_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"


def decode_local_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h_b64, p_b64, s_b64 = parts
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        expected_sig = hmac.new(
            settings.LOCAL_JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(_b64url_encode(expected_sig), s_b64):
            return None
        payload = json.loads(_b64url_decode(p_b64).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ============================================================================
# Supabase REST / Auth 호출 헬퍼 (eyJ... JWT 키 및 신규 sb_secret_/sb_publishable_ 키 모두 호환)
# ============================================================================
def _build_supabase_headers(
    api_key: str,
    bearer_token: Optional[str] = None,
    include_bearer_for_apikey: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "FairPle-CloudRun-Server/1.5.2",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif include_bearer_for_apikey:
        headers["Authorization"] = f"Bearer {api_key}"

    if extra_headers:
        headers.update(extra_headers)
    return headers


def supabase_rest_request(
    method: str,
    table_or_path: str,
    payload: Optional[Any] = None,
    query: str = "",
    use_service_role: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    if not settings.is_supabase_configured:
        return None

    import httpx

    api_key = (
        settings.SUPABASE_SERVICE_ROLE_KEY
        if use_service_role and settings.SUPABASE_SERVICE_ROLE_KEY
        else (settings.SUPABASE_ANON_KEY or settings.SUPABASE_SERVICE_ROLE_KEY)
    )
    url = f"{settings.SUPABASE_URL}/rest/v1/{table_or_path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"

    merged_extra = {"Prefer": "return=representation"}
    if extra_headers:
        merged_extra.update(extra_headers)

    # 1차 시도: eyJ... 키이면 Authorization Bearer 포함, sb_... 신규 키이면 apikey 단독 먼저 시도 후 폴백
    is_jwt_key = api_key.startswith("eyJ")
    headers_primary = _build_supabase_headers(
        api_key=api_key,
        include_bearer_for_apikey=is_jwt_key,
        extra_headers=merged_extra,
    )
    resp = httpx.request(
        method=method.upper(),
        url=url,
        json=payload if payload is not None else None,
        headers=headers_primary,
        timeout=10.0,
    )

    # 401/403 발생 시 반대 헤더 조합(Bearer 포함/제외) 또는 anon 키로 2차 재시도
    if resp.status_code in (401, 403):
        headers_alt = _build_supabase_headers(
            api_key=api_key,
            include_bearer_for_apikey=not is_jwt_key,
            extra_headers=merged_extra,
        )
        resp = httpx.request(
            method=method.upper(),
            url=url,
            json=payload if payload is not None else None,
            headers=headers_alt,
            timeout=10.0,
        )

    if resp.status_code in (401, 403) and use_service_role and settings.SUPABASE_ANON_KEY:
        anon_key = settings.SUPABASE_ANON_KEY
        headers_anon = _build_supabase_headers(
            api_key=anon_key,
            include_bearer_for_apikey=anon_key.startswith("eyJ"),
            extra_headers=merged_extra,
        )
        resp = httpx.request(
            method=method.upper(),
            url=url,
            json=payload if payload is not None else None,
            headers=headers_anon,
            timeout=10.0,
        )

    if resp.status_code >= 400:
        local_store.last_supabase_error = (
            f"REST {method} {table_or_path} HTTP {resp.status_code}: {resp.text[:300]}"
        )
        resp.raise_for_status()

    local_store.last_supabase_error = None
    return resp.json() if resp.text else None


def supabase_auth_request(
    method: str,
    auth_path: str,
    payload: Optional[Any] = None,
    use_service_role: bool = False,
    bearer_token: Optional[str] = None,
) -> Any:
    if not settings.is_supabase_configured:
        return None

    import httpx

    api_key = (
        settings.SUPABASE_SERVICE_ROLE_KEY
        if (use_service_role and settings.SUPABASE_SERVICE_ROLE_KEY)
        else (settings.SUPABASE_ANON_KEY or settings.SUPABASE_SERVICE_ROLE_KEY)
    )
    url = f"{settings.SUPABASE_URL}/auth/v1/{auth_path.lstrip('/')}"
    is_jwt_key = api_key.startswith("eyJ")

    headers = _build_supabase_headers(
        api_key=api_key,
        bearer_token=bearer_token,
        include_bearer_for_apikey=is_jwt_key if not bearer_token else False,
    )

    resp = httpx.request(
        method=method.upper(),
        url=url,
        json=payload if payload is not None else None,
        headers=headers,
        timeout=10.0,
    )
    if resp.status_code in (401, 403) and not bearer_token:
        headers_alt = _build_supabase_headers(
            api_key=api_key,
            include_bearer_for_apikey=not is_jwt_key,
        )
        resp = httpx.request(
            method=method.upper(),
            url=url,
            json=payload if payload is not None else None,
            headers=headers_alt,
            timeout=10.0,
        )

    if resp.status_code >= 400:
        local_store.last_supabase_error = (
            f"AUTH {method} {auth_path} HTTP {resp.status_code}: {resp.text[:300]}"
        )
        resp.raise_for_status()
    return resp.json() if resp.text else None


def supabase_auth_get_user(access_token: str) -> Optional[Dict[str, Any]]:
    if not settings.is_supabase_configured:
        return None
    try:
        return supabase_auth_request("GET", "user", bearer_token=access_token)
    except Exception:
        return None


def record_admin_audit_log(
    admin_email: str,
    action: str,
    target_type: str,
    target_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "id": str(uuid.uuid4()),
        "admin_email": admin_email,
        "action": action,
        "target_type": target_type,
        "target_id": target_id,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    local_store.admin_audit_logs.append(entry)
    if settings.is_supabase_configured:
        try:
            supabase_rest_request("POST", "admin_audit_logs", payload=entry)
        except Exception:
            pass
    return entry


# ============================================================================
# 1. X-GAS-Proxy-Secret 검증 의존성
# ============================================================================
def verify_gas_proxy_secret(
    request: Request,
    x_gas_proxy_secret: Optional[str] = Header(default=None, alias="X-GAS-Proxy-Secret"),
) -> None:
    expected = settings.INTERNAL_PROXY_SECRET
    if not expected or expected == "fairple-gas-proxy-secret-2026":
        if x_gas_proxy_secret and expected and x_gas_proxy_secret != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="유효하지 않은 X-GAS-Proxy-Secret 헤더입니다.",
            )
        return

    if x_gas_proxy_secret != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GAS 프록시 보안 시크릿(X-GAS-Proxy-Secret)이 일치하지 않습니다.",
        )


# ============================================================================
# 2. 현재 로그인 사용자 인증 의존성 (get_current_user)
# ============================================================================
def get_current_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> UserProfile:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다 (Authorization Bearer 토큰 누락).",
        )

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 토큰입니다.",
        )

    decoded = decode_local_token(token)
    if decoded:
        email = decoded.get("email", "").lower().strip()
        user_id = decoded.get("sub", "")
        name = decoded.get("name", email.split("@")[0])
        return UserProfile(
            id=user_id,
            email=email,
            name=name,
            is_admin=email in settings.ADMIN_EMAILS,
        )

    sb_user = supabase_auth_get_user(token)
    if sb_user and sb_user.get("email"):
        email = sb_user["email"].lower().strip()
        user_id = sb_user.get("id", str(uuid.uuid4()))
        user_meta = sb_user.get("user_metadata") or {}
        name = user_meta.get("name") or email.split("@")[0]

        try:
            rows = supabase_rest_request(
                "GET",
                "users",
                query=f"email=eq.{urllib.parse.quote(email)}&select=id,email,name",
            )
            if isinstance(rows, list) and rows:
                user_id = rows[0].get("id", user_id)
                name = rows[0].get("name", name)
        except Exception:
            pass

        return UserProfile(
            id=user_id,
            email=email,
            name=name,
            is_admin=email in settings.ADMIN_EMAILS,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="만료되었거나 유효하지 않은 인증 토큰입니다. 다시 로그인해 주세요.",
    )


def get_optional_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Optional[UserProfile]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        return get_current_user(authorization)
    except Exception:
        return None


# ============================================================================
# 3. 관리자 RBAC 3중 방어 의존성 (require_admin)
# ============================================================================
def require_admin(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    email_lower = current_user.email.lower().strip()
    if email_lower not in settings.ADMIN_EMAILS:
        record_admin_audit_log(
            admin_email=email_lower,
            action="UNAUTHORIZED_ADMIN_ACCESS_BLOCKED",
            target_type="admin_endpoint",
            target_id="require_admin",
            metadata={"reason": "Email not in ADMIN_EMAILS whitelist"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 없습니다. 허가된 관리자 이메일(ADMIN_EMAILS)만 접근할 수 있습니다.",
        )
    return current_user
