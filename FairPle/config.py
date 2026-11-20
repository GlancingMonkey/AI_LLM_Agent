"""
페어플 (FairPle) — 서버 환경변수 및 설정 관리 (server/config.py)
"""

import os
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "").strip()
        self.SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.GEMINI_API_KEY: str = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.INTERNAL_PROXY_SECRET: str = os.getenv("INTERNAL_PROXY_SECRET", "").strip()

        raw_admins = os.getenv("ADMIN_EMAILS", "admin@example.com,manager@univ.ac.kr")
        self.ADMIN_EMAILS: Set[str] = {
            email.strip().lower()
            for email in raw_admins.split(",")
            if email.strip()
        }

        raw_origins = os.getenv(
            "ALLOWED_ORIGINS",
            "https://script.google.com,https://script.googleusercontent.com,http://localhost:8080,*",
        )
        self.ALLOWED_ORIGINS: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

        self.LOCAL_JWT_SECRET: str = (
            self.INTERNAL_PROXY_SECRET
            or self.SUPABASE_SERVICE_ROLE_KEY
            or "fairple-local-dev-jwt-secret-key"
        )

    @property
    def is_supabase_configured(self) -> bool:
        return bool(
            self.SUPABASE_URL
            and "your-project-id" not in self.SUPABASE_URL
            and (self.SUPABASE_ANON_KEY or self.SUPABASE_SERVICE_ROLE_KEY)
        )


settings = Settings()
