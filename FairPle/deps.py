"""
페어플 (FairPle) — 보안 미들웨어, JWT 인증 및 관리자 RBAC 의존성 (server/deps.py)
- X-GAS-Proxy-Secret 검증
- Supabase Auth / JWT 기반 get_current_user
- ADMIN_EMAILS 기반 3중 방어 require_admin + admin_audit_logs 기록
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
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
# 인메모리 폴백 저장소 (Supabase 미연결 로컬 테스트 / CI 검증 시 자동 동작)
# ============================================================================
class LocalMemoryStore:
    def __init__(self) -> None:
        self.users_by_email: Dict[str, Dict[str, Any]] = {}
        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.admin_audit_logs: List[Dict[str, Any]] = []


local_store = LocalMemoryStore()


# ============================================================================
# 경량 HMAC-SHA256 JWT 헬퍼 (외부 패키지 버전 충돌 없이 표준 JWT 생성·검증)
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
# Supabase REST / Auth 호출 헬퍼
# ============================================================================
def supabase_rest_request(
    method: str,
    table_or_path: str,
    payload: Optional[Any] = None,
    query: str = "",
    use_service_role: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Supabase PostgREST API 직접 호출 헬퍼"""
    if not settings.is_supabase_configured:
        return None

    api_key = (
        settings.SUPABASE_SERVICE_ROLE_KEY
        if use_service_role and settings.SUPABASE_SERVICE_ROLE_KEY
        else settings.SUPABASE_ANON_KEY
    )
    url = f"{settings.SUPABASE_URL}/rest/v1/{table_or_path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if extra_headers:
        headers.update(extra_headers)

    data_bytes = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else None


def supabase_auth_get_user(access_token: str) -> Optional[Dict[str, Any]]:
    """Supabase Auth /auth/v1/user 엔드포인트로 JWT 유효성 검증"""
    if not settings.is_supabase_configured:
        return None

    api_key = settings.SUPABASE_ANON_KEY or settings.SUPABASE_SERVICE_ROLE_KEY
    url = f"{settings.SUPABASE_URL}/auth/v1/user"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {access_token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def record_admin_audit_log(
    admin_email: str,
    action: str,
    target_type: str,
    target_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """admin_audit_logs 테이블 및 로컬 스토어에 감사 로그 기록"""
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


# ============================================================================
# 1. X-GAS-Proxy-Secret 검증 의존성
# ============================================================================
def verify_gas_proxy_secret(
    request: Request,
    x_gas_proxy_secret: Optional[str] = Header(default=None, alias="X-GAS-Proxy-Secret"),
) -> None:
    """INTERNAL_PROXY_SECRET이 설정된 경우 X-GAS-Proxy-Secret 헤더 일치 여부를 검증"""
    expected = settings.INTERNAL_PROXY_SECRET
    if not expected or expected == "fairple-gas-proxy-secret-2026":
        # 기본 예시값이거나 미설정 시 로컬/개발 편의를 위해 헤더가 있으면 대조하고 없으면 통과
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

    # 1차: 로컬 서명 토큰 확인
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

    # 2차: Supabase Auth JWT 검증
    sb_user = supabase_auth_get_user(token)
    if sb_user and sb_user.get("email"):
        email = sb_user["email"].lower().strip()
        user_id = sb_user.get("id", str(uuid.uuid4()))
        user_meta = sb_user.get("user_metadata") or {}
        name = user_meta.get("name") or email.split("@")[0]

        # public.users 테이블에서 이름 조회 시도
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
