"""
페어플 (FairPle) — Cloud Run FastAPI 메인 엔트리포인트 (server/main.py)
- Phase A ~ D 전체 라우터 통합: auth, rooms, auction, tasks, report, chat, admin
"""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import admin, auction, auth, chat, report, rooms, tasks

app = FastAPI(
    title="FairPle (페어플) — 대학 팀플 공정 역할 분배 및 기여도 증빙 API",
    version="1.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if "*" not in settings.ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase A ~ D 전체 라우터 마운트
app.include_router(auth.router)
app.include_router(rooms.router)
app.include_router(auction.router)
app.include_router(tasks.router)
app.include_router(report.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "fairple-server",
        "version": "1.5.0",
        "phase": "Phase A-D Complete",
        "supabase_configured": settings.is_supabase_configured,
        "gemini_model": settings.GEMINI_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
