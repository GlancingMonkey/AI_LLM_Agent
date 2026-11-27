"""
서울시 도시계획 LangGraph Agentic RAG FastAPI 서버 (main.py)

- 동일 폴더의 rag_agent.py 모듈을 직접 임포트 (from rag_agent import rag_service)
- GAS 웹앱 시작 시 연결 확인용 GET /health 및 CORS 설정
- PDF 파일 업로드(Base64 및 Multipart) -> 텍스트 변환 -> 청크 분할 -> ChromaDB 인덱싱
- LangGraph 기반 라우팅(route_question) 및 RAG 추론(retrieve -> grade_documents -> generate / casual_talk)
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 동일 폴더 내 파이썬 모듈 직접 참조 (same-folder-py-import 규칙 준수)
from rag_agent import rag_service

load_dotenv()

app = FastAPI(
    title="Seoul Urban Plan Agentic RAG API",
    description="LangGraph + ChromaDB + Google Gemini 기반 서울시 계획 Agentic RAG 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://script.google.com",
        "https://script.googleusercontent.com",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PdfBase64UploadRequest(BaseModel):
    filename: str = Field(..., description="업로드할 PDF 파일명")
    content_base64: str = Field(..., description="Base64 인코딩된 PDF 바이트 문자열")
    chunk_size: int = Field(600, ge=100, le=4000, description="청크 크기")
    chunk_overlap: int = Field(100, ge=0, le=1000, description="청크 오버랩")


class TextUploadRequest(BaseModel):
    filename: str = Field("서울시_도시계획_문서.pdf", description="문서 제목/파일명")
    text: str = Field(..., description="인덱싱할 텍스트 원문")
    chunk_size: int = Field(600, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=1000)


class QueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """GAS 웹앱 시작 시 자동 연결 확인 및 상태 점검 엔드포인트"""
    status_info = rag_service.get_status()
    return {
        "status": "ok",
        "service": "seoul-agentic-rag",
        "model": status_info["model"],
        "api_key_configured": status_info["api_key_configured"],
        "total_chunks": status_info["total_chunks"],
        "sources_count": len(status_info["sources"]),
    }


@app.get("/status")
def get_rag_status() -> Dict[str, Any]:
    """ChromaDB 인덱싱 현황 및 LangGraph 파이프라인 구조 조회"""
    return rag_service.get_status()


@app.post("/upload-pdf-base64")
def upload_pdf_base64(payload: PdfBase64UploadRequest) -> Dict[str, Any]:
    """GAS 클라이언트에서 Base64로 전달된 PDF 파일을 텍스트 변환 및 청크 분할하여 ChromaDB에 저장"""
    try:
        raw_b64 = payload.content_base64.strip()
        if "," in raw_b64 and raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1]
        pdf_bytes = base64.b64decode(raw_b64)
        return rag_service.load_and_split_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=payload.filename,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류가 발생했습니다: {exc}") from exc


@app.post("/upload-pdf")
def upload_pdf_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(600),
    chunk_overlap: int = Form(100),
) -> Dict[str, Any]:
    """Multipart PDF 파일 업로드 및 ChromaDB 인덱싱"""
    try:
        pdf_bytes = file.file.read()
        filename = file.filename or "uploaded.pdf"
        return rag_service.load_and_split_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 업로드 처리 실패: {exc}") from exc


@app.post("/upload-text")
def upload_text_document(payload: TextUploadRequest) -> Dict[str, Any]:
    """텍스트 문서를 청크 단위로 쪼개어 ChromaDB에 인덱싱"""
    try:
        return rag_service.load_and_split_text(
            text=payload.text,
            filename=payload.filename,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"문서 인덱싱 실패: {exc}") from exc


@app.post("/reset-db")
def reset_vector_db() -> Dict[str, Any]:
    """ChromaDB 벡터 저장소 초기화"""
    return rag_service.reset_vectorstore()


@app.post("/query")
def query_agentic_rag(payload: QueryRequest) -> Dict[str, Any]:
    """LangGraph Agentic RAG 파이프라인 실행 (route_question -> retrieve -> grade_documents -> generate / casual_talk)"""
    try:
        return rag_service.run_query(payload.question)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"질의 실행 중 오류 발생: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
