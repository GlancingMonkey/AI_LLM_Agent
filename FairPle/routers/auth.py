"""
페어플 (FairPle) — 이메일 회원가입·로그인·세션 확인 라우터 (server/routers/auth.py)
- Supabase Auth (/auth/v1/admin/users + /auth/v1/signup) 및 public.users 테이블 양방향 확실 동기화
"""

import hashlib
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from deps import (
    UserProfile,
    create_access_token,
    get_current_user,
    local_store,
    supabase_auth_request,
    supabase_rest_request,
    verify_gas_proxy_secret,
)

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    dependencies=[Depends(verify_gas_proxy_secret)],
)


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)
    name: str = Field(..., min_length=1, max_length=50)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)


def _hash_password(password: str, email: str) -> str:
    salted = f"fairple:{email.lower().strip()}:{password}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def _upsert_public_user(user_id: str, email: str, name: str) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    clean_email = email.lower().strip()
    clean_name = name.strip()
    user_record = {
        "id": user_id,
        "email": clean_email,
        "name": clean_name,
        "created_at": now_iso,
    }

    if settings.is_supabase_configured:
        try:
            existing = supabase_rest_request(
                "GET",
                "users",
                query=f"email=eq.{urllib.parse.quote(clean_email)}&select=id,email,name,created_at",
            )
            if isinstance(existing, list) and existing:
                return existing[0]

            created = supabase_rest_request(
                "POST",
                "users",
                payload=user_record,
                extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
            if isinstance(created, list) and created:
                return created[0]
        except Exception as exc:
            local_store.last_supabase_error = f"public.users upsert error: {exc}"

    return user_record


def _ensure_supabase_auth_user(email: str, password: Optional[str], name: str) -> Optional[str]:
    """
    Supabase Auth(auth.users)에 계정을 생성하여 Supabase 대시보드의 Authentication > Users에
    즉시 표시되도록 보장합니다.
    1차: SUPABASE_SERVICE_ROLE_KEY가 있으면 /auth/v1/admin/users (email_confirm=True) 호출
    2차: /auth/v1/signup 호출
    """
    if not settings.is_supabase_configured:
        return None

    clean_email = email.lower().strip()
    pwd = password or "FairPleTemp123!"

    # 1차: Admin API (/auth/v1/admin/users) - 이메일 확인 대기나 발송 한도(429) 없이 즉시 생성
    if settings.SUPABASE_SERVICE_ROLE_KEY:
        try:
            admin_res = supabase_auth_request(
                "POST",
                "admin/users",
                payload={
                    "email": clean_email,
                    "password": pwd,
                    "email_confirm": True,
                    "user_metadata": {"name": name},
                },
                use_service_role=True,
            )
            if isinstance(admin_res, dict) and admin_res.get("id"):
                return str(admin_res["id"])
        except Exception:
            pass

    # 2차: 일반 /auth/v1/signup 호출
    try:
        signup_res = supabase_auth_request(
            "POST",
            "signup",
            payload={
                "email": clean_email,
                "password": pwd,
                "data": {"name": name},
            },
            use_service_role=False,
        )
        if isinstance(signup_res, dict):
            sb_user = signup_res.get("user") or signup_res
            if isinstance(sb_user, dict) and sb_user.get("id"):
                return str(sb_user["id"])
    except Exception:
        pass

    return None


@router.post("/signup")
def signup(req: SignupRequest) -> Dict[str, Any]:
    email = req.email.lower().strip()
    name = req.name.strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="올바른 이메일 주소 형식을 입력해 주세요.",
        )

    # 1. Supabase Auth (Authentication > Users) 및 public.users (Table Editor > users) 동시 생성
    sb_uid = _ensure_supabase_auth_user(email=email, password=req.password, name=name)
    user_id = sb_uid or str(uuid.uuid4())

    user_doc = _upsert_public_user(user_id=user_id, email=email, name=name)
    user_id = str(user_doc.get("id") or user_id)

    local_store.users_by_email[email] = {
        "id": user_id,
        "email": email,
        "name": name,
        "password_hash": _hash_password(req.password, email),
        "created_at": user_doc.get("created_at"),
    }
    local_store.users_by_id[user_id] = local_store.users_by_email[email]

    access_token = create_access_token(user_id=user_id, email=email, name=name)

    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
        "supabase_synced": settings.is_supabase_configured,
        "user": {
            "id": user_id,
            "email": email,
            "name": name,
            "is_admin": email in settings.ADMIN_EMAILS,
        },
    }


@router.post("/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    email = req.email.lower().strip()
    access_token: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None

    # 1. Supabase Auth 로그인 시도
    if settings.is_supabase_configured:
        try:
            sb_data = supabase_auth_request(
                "POST",
                "token?grant_type=password",
                payload={"email": email, "password": req.password},
            )
            if isinstance(sb_data, dict):
                access_token = sb_data.get("access_token")
                sb_user = sb_data.get("user") or {}
                user_id = sb_user.get("id")
                name = (sb_user.get("user_metadata") or {}).get("name")
        except Exception:
            pass

    # 2. 로컬 스토어 또는 public.users 테이블 조회
    local_user = local_store.users_by_email.get(email)
    if not access_token:
        if local_user:
            expected_hash = _hash_password(req.password, email)
            if local_user.get("password_hash") != expected_hash:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="이메일 또는 비밀번호가 올바르지 않습니다.",
                )
            user_id = local_user["id"]
            name = local_user["name"]
            access_token = create_access_token(user_id=user_id, email=email, name=name)
        else:
            if settings.is_supabase_configured:
                try:
                    rows = supabase_rest_request(
                        "GET",
                        "users",
                        query=f"email=eq.{urllib.parse.quote(email)}&select=id,email,name",
                    )
                    if isinstance(rows, list) and rows:
                        user_id = rows[0]["id"]
                        name = rows[0]["name"]
                        access_token = create_access_token(
                            user_id=user_id, email=email, name=name
                        )
                except Exception:
                    pass

            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="등록되지 않은 이메일이거나 비밀번호가 일치하지 않습니다.",
                )

    final_name = name or email.split("@")[0]
    # 만약 이전에 인메모리로만 가입되어 Supabase에 없었던 계정이라면 로그인 시 자동으로 Supabase에도 동기화!
    sb_uid = _ensure_supabase_auth_user(email=email, password=req.password, name=final_name)
    final_id = str(user_id or sb_uid or uuid.uuid4())
    user_doc = _upsert_public_user(user_id=final_id, email=email, name=final_name)
    final_id = str(user_doc.get("id") or final_id)

    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
        "supabase_synced": settings.is_supabase_configured,
        "user": {
            "id": final_id,
            "email": email,
            "name": final_name,
            "is_admin": email in settings.ADMIN_EMAILS,
        },
    }


@router.get("/me")
def get_me(current_user: UserProfile = Depends(get_current_user)) -> Dict[str, Any]:
    # 이미 로그인된 세션이 /api/auth/me를 호출할 때도 Supabase public.users 및 auth.users에 자동 백필 동기화!
    if settings.is_supabase_configured:
        _ensure_supabase_auth_user(
            email=current_user.email,
            password=None,
            name=current_user.name,
        )
        _upsert_public_user(
            user_id=current_user.id,
            email=current_user.email,
            name=current_user.name,
        )

    return {
        "status": "success",
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "is_admin": current_user.is_admin,
        },
    }
