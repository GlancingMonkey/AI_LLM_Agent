"""
페어플 (FairPle) — 이메일 회원가입·로그인·세션 확인 라우터 (server/routers/auth.py)
- POST /api/auth/signup
- POST /api/auth/login
- GET  /api/auth/me
"""

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
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
    user_record = {
        "id": user_id,
        "email": email.lower().strip(),
        "name": name.strip(),
        "created_at": now_iso,
    }

    if settings.is_supabase_configured:
        try:
            existing = supabase_rest_request(
                "GET",
                "users",
                query=f"email=eq.{urllib.parse.quote(user_record['email'])}&select=id,email,name,created_at",
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
        except Exception:
            pass

    return user_record


@router.post("/signup")
def signup(req: SignupRequest) -> Dict[str, Any]:
    email = req.email.lower().strip()
    name = req.name.strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="올바른 이메일 주소 형식을 입력해 주세요.",
        )

    user_id = str(uuid.uuid4())
    access_token: Optional[str] = None

    # 1. Supabase Auth가 구성된 경우 /auth/v1/signup 호출 시도
    if settings.is_supabase_configured:
        api_key = settings.SUPABASE_ANON_KEY or settings.SUPABASE_SERVICE_ROLE_KEY
        signup_url = f"{settings.SUPABASE_URL}/auth/v1/signup"
        payload = {
            "email": email,
            "password": req.password,
            "data": {"name": name},
        }
        http_req = urllib.request.Request(
            signup_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                sb_data = json.loads(resp.read().decode("utf-8"))
                sb_user = sb_data.get("user") or sb_data
                if sb_user and sb_user.get("id"):
                    user_id = sb_user["id"]
                access_token = sb_data.get("access_token")
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            if "already registered" in err_body.lower() or http_err.code == 422:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="이미 가입된 이메일입니다. 로그인해 주세요.",
                )
        except Exception:
            pass

    # 2. 로컬 스토어 중복 검사 및 저장
    if email in local_store.users_by_email and not access_token:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 가입된 이메일입니다. 로그인해 주세요.",
        )

    user_doc = _upsert_public_user(user_id=user_id, email=email, name=name)
    user_id = user_doc.get("id", user_id)

    local_store.users_by_email[email] = {
        "id": user_id,
        "email": email,
        "name": name,
        "password_hash": _hash_password(req.password, email),
        "created_at": user_doc.get("created_at"),
    }
    local_store.users_by_id[user_id] = local_store.users_by_email[email]

    # Supabase 이메일 인증 대기 상태이거나 로컬 모드일 때도 즉시 로그인 가능하도록 JWT 발급
    if not access_token:
        access_token = create_access_token(user_id=user_id, email=email, name=name)

    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
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

    # 1. Supabase Auth /auth/v1/token?grant_type=password 시도
    if settings.is_supabase_configured:
        api_key = settings.SUPABASE_ANON_KEY or settings.SUPABASE_SERVICE_ROLE_KEY
        token_url = f"{settings.SUPABASE_URL}/auth/v1/token?grant_type=password"
        payload = {"email": email, "password": req.password}
        http_req = urllib.request.Request(
            token_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                sb_data = json.loads(resp.read().decode("utf-8"))
                access_token = sb_data.get("access_token")
                sb_user = sb_data.get("user") or {}
                user_id = sb_user.get("id")
                name = (sb_user.get("user_metadata") or {}).get("name")
        except Exception:
            pass

    # 2. 로컬 스토어 확인
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
            # Supabase public.users에 이미 등록된 사용자인지 조회
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
                        access_token = create_access_token(user_id=user_id, email=email, name=name)
                except Exception:
                    pass

            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="등록되지 않은 이메일이거나 비밀번호가 일치하지 않습니다.",
                )

    final_name = name or email.split("@")[0]
    final_id = user_id or str(uuid.uuid4())
    _upsert_public_user(user_id=final_id, email=email, name=final_name)

    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": final_id,
            "email": email,
            "name": final_name,
            "is_admin": email in settings.ADMIN_EMAILS,
        },
    }


@router.get("/me")
def get_me(current_user: UserProfile = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "status": "success",
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "is_admin": current_user.is_admin,
        },
    }
