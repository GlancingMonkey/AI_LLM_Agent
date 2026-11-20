"""
Phase A 검증 테스트: GET /health, POST /api/auth/signup, POST /api/auth/login, GET /api/auth/me
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from deps import get_current_user, require_admin
from routers import auth


def test_health_check():
    res = main.health_check()
    assert res["status"] == "ok"
    assert res["service"] == "fairple-server"


def test_auth_signup_login_me():
    signup_res = auth.signup(
        auth.SignupRequest(
            email="jieun.lee@univ.ac.kr",
            password="password123",
            name="이지은",
        )
    )
    assert signup_res["status"] == "success"
    token = signup_res["access_token"]
    assert token

    login_res = auth.login(
        auth.LoginRequest(
            email="jieun.lee@univ.ac.kr",
            password="password123",
        )
    )
    assert login_res["status"] == "success"
    assert login_res["user"]["name"] == "이지은"

    user_profile = get_current_user(f"Bearer {token}")
    me_res = auth.get_me(user_profile)
    assert me_res["status"] == "success"
    assert me_res["user"]["email"] == "jieun.lee@univ.ac.kr"
