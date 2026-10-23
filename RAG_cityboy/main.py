"""
Google Cloud Run 배포용 RAG AI 서버
- 모델: Google Gemini 3.5 Flash Lite (답변 생성) & OpenAI text-embedding-3-small (임베딩)
- 벡터 저장소: Supabase (pgvector)
- 질의 확장: LangChain Query Augmentation Chain (MessagesPlaceholder 기반)
- 실시간 스트리밍: SSE (Server-Sent Events) StreamHandler
"""

import os
import io
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import pypdf
from dotenv import load_dotenv

# LangChain 핵심 컴포넌트 임포트
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client

# 환경 변수 로드 (.env 파일이 있을 경우)
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG-Server")

# FastAPI 앱 생성
app = FastAPI(
    title="City Policy RAG AI Server",
    description="도시 정책 문서를 기반으로 질의 확장 및 실시간 스트리밍 RAG 답변을 제공하는 AI 서버",
    version="1.0.0"
)

# CORS 설정 (Google Apps Script 및 외부 브라우저 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 환경 변수 및 Supabase 클라이언트 헬퍼 함수
# ==============================================================================
def get_google_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google AI API 키(GEMINI_API_KEY 또는 GOOGLE_API_KEY)가 환경 변수에 설정되지 않았습니다."
        )
    return key

def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API 키(OPENAI_API_KEY)가 설정되지 않았습니다. 임베딩 벡터 생성을 위해 필요합니다."
        )
    return key

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase 연동 정보(SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY)가 누락되었습니다."
        )
    return create_client(url, key)

# ==============================================================================
# Pydantic 데이터 모델
# ==============================================================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 발화자 ('user' 또는 'assistant')")
    content: str = Field(..., description="대화 내용")

class ChatStreamRequest(BaseModel):
    query: str = Field(..., description="사용자의 현재 질문")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="이전 대화 이력")
    filename: Optional[str] = Field(None, description="특정 파일명 필터")

# ==============================================================================
# 1. 헬스 체크 엔드포인트
# ==============================================================================
@app.get("/health", summary="서버 및 API 연동 상태 점검")
def health_check():
    return {
        "status": "healthy",
        "service": "City Policy RAG AI Server",
        "llm_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "embedding_model": "text-embedding-3-small",
        "api_readiness": {
            "google_gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
            "openai_embedding": bool(os.getenv("OPENAI_API_KEY")),
            "supabase_vector_db": bool(os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")))
        }
    }

# ==============================================================================
# 2. PDF 텍스트 변환 엔드포인트
# ==============================================================================
@app.post("/extract-pdf", summary="업로드된 PDF 파일의 텍스트 추출")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"선택한 파일('{file.filename}')은 PDF 형식이 아닙니다."
        )

    try:
        content = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(content))
        total_pages = len(pdf_reader.pages)
        
        full_text_list = []
        for idx, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text_list.append(f"--- [페이지 {idx}] ---\n" + page_text.strip())

        full_text = "\n\n".join(full_text_list)
        if not full_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF 파일에서 텍스트를 추출할 수 없습니다. 스캔된 이미지 문서인지 확인해 주세요."
            )

        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": total_pages,
            "total_characters": len(full_text),
            "preview_text": full_text
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF 추출 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF 파일 분석 중 오류가 발생했습니다: {str(e)}"
        )

# ==============================================================================
# 3. 텍스트 청킹, OpenAI 임베딩 및 Supabase 벡터 DB 적재 엔드포인트
# ==============================================================================
@app.post("/create-rag", summary="문서를 청크로 분할하여 Supabase 벡터 DB에 저장")
async def create_rag(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    filename: Optional[str] = Form(None)
):
    target_text = ""
    target_filename = filename or "document.pdf"

    if file is not None:
        target_filename = file.filename
        try:
            content = await file.read()
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            extracted_list = [
                f"[Page {i}] {p.extract_text() or ''}"
                for i, p in enumerate(pdf_reader.pages, start=1)
                if (p.extract_text() or '').strip()
            ]
            target_text = "\n\n".join(extracted_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 파일 읽기 실패: {str(e)}")
    elif raw_text:
        target_text = raw_text.strip()
    else:
        raise HTTPException(status_code=400, detail="PDF 파일 또는 텍스트 내용이 전달되지 않았습니다.")

    if not target_text:
        raise HTTPException(status_code=400, detail="임베딩할 본문 텍스트가 비어 있습니다.")

    # 1) 청크 단위 분할 (800자, 오버랩 150자)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(target_text)

    # 2) OpenAI 임베딩 모델 (text-embedding-3-small, 1536차원)
    openai_key = get_openai_api_key()
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key
    )

    try:
        embeddings = embeddings_model.embed_documents(chunks)
    except Exception as e:
        logger.error(f"임베딩 생성 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"OpenAI 임베딩 API 호출 실패: {str(e)}")

    # 3) Supabase documents 테이블 적재
    supabase = get_supabase_client()
    records = [
        {
            "content": chunk,
            "metadata": {
                "source": target_filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            },
            "embedding": emb
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    try:
        # 50개 단위 배치 저장
        for i in range(0, len(records), 50):
            supabase.table("documents").insert(records[i:i + 50]).execute()

        return {
            "status": "success",
            "message": f"{len(chunks)}개 청크가 Supabase 벡터 DB에 성공적으로 저장되었습니다.",
            "filename": target_filename,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        logger.error(f"Supabase 저장 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Supabase 벡터 DB 저장 실패: {str(e)}")

# ==============================================================================
# 4. 질의 확장(Query Augmentation) 및 실시간 스트리밍 RAG 답변 엔드포인트
# ==============================================================================
@app.post("/chat/stream", summary="질의 확장 및 실시간 스트리밍 답변 생성")
async def chat_stream(request_data: ChatStreamRequest):
    google_api_key = get_google_api_key()
    openai_key = get_openai_api_key()
    supabase = get_supabase_client()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 대화 기록(Chat History)을 LangChain 메시지 객체로 변환
            history_messages = [
                HumanMessage(content=m.content) if m.role.lower() == "user" else AIMessage(content=m.content)
                for m in request_data.chat_history
            ]

            # 2. 질의 확장 체인(Query Augmentation Chain) 구성
            augmentation_prompt = ChatPromptTemplate.from_messages([
                ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])

            model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            augmentation_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                temperature=0.2
            )

            query_augmentation_chain = augmentation_prompt | augmentation_llm | StrOutputParser()

            # 체인 실행: 현재 질문과 과거 대화 기록 전달
            if history_messages:
                augmented_query = await query_augmentation_chain.ainvoke({
                    "chat_history": history_messages,
                    "query": request_data.query
                })
                augmented_query = augmented_query.strip()
            else:
                augmented_query = request_data.query

            # 클라이언트에 확장된 질문 이벤트 전송
            yield json.dumps({
                "event": "augmented_query",
                "original_query": request_data.query,
                "augmented_query": augmented_query
            }, ensure_ascii=False) + "\n\n"

            # 3. 증강된 질의로 Supabase 벡터 DB 유사도 검색 (Retriever)
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_key
            )
            query_embedding = embedding_model.embed_query(augmented_query)

            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": 4,
                "filter": {"source": request_data.filename} if request_data.filename else {}
            }
            search_res = supabase.rpc("match_documents", rpc_params).execute()
            retrieved_chunks = search_res.data or []

            # 출처 및 검색 청크 목록 분리 전송
            sources_payload = []
            context_blocks = []
            for item in retrieved_chunks:
                source_file = item.get("metadata", {}).get("source", "문서")
                chunk_idx = item.get("metadata", {}).get("chunk_index", 0)
                sim_score = round(float(item.get("similarity", 0.0)), 4)
                content = item.get("content", "")

                sources_payload.append({
                    "source": source_file,
                    "chunk_index": chunk_idx,
                    "similarity": sim_score,
                    "content": content
                })
                context_blocks.append(f"[출처: {source_file} (청크 #{chunk_idx}, 유사도 {sim_score})]\n{content}")

            yield json.dumps({
                "event": "sources",
                "retrieved_count": len(retrieved_chunks),
                "sources": sources_payload
            }, ensure_ascii=False) + "\n\n"

            # 4. 도시 정책 전문가 프롬프트 및 실시간 토큰 스트리밍 생성
            context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "관련된 참조 문서 내용을 찾을 수 없습니다."

            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 반드시 주어진 [참조 문서]의 내용만을 근거로 삼아 전문적이고 명확하게 한국어로 답변하라. 답변 시 참조한 출처를 명확히 밝히고, 문서에 없는 내용은 추측하지 말라."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "다음 [참조 문서]를 기반으로 아래 질문에 대해 답변하라.\n\n[참조 문서]:\n{context}\n\n[질문]:\n{question}")
            ])

            streaming_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                temperature=0.3,
                streaming=True
            )

            rag_chain = rag_prompt | streaming_llm

            # 실시간 토큰 스트리밍 전송 (StreamHandler)
            async for chunk in rag_chain.astream({
                "context": context_text,
                "question": augmented_query,
                "chat_history": history_messages
            }):
                token_text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token_text:
                    yield json.dumps({
                        "event": "token",
                        "token": token_text
                    }, ensure_ascii=False) + "\n\n"

            # 스트리밍 완료 이벤트
            yield json.dumps({
                "event": "done",
                "status": "completed"
            }, ensure_ascii=False) + "\n\n"

        except Exception as e:
            logger.error(f"스트리밍 처리 오류: {str(e)}", exc_info=True)
            yield json.dumps({
                "event": "error",
                "message": f"질의 처리 중 오류 발생: {str(e)}"
            }, ensure_ascii=False) + "\n\n"

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main.py:app", host="0.0.0.0", port=port, reload=True)
