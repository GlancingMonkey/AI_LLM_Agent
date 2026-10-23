"""
================================================================================
Google Cloud Run 배포용 AI RAG 서버 (Python FastAPI)
================================================================================
- 벡터 데이터베이스: ChromaDB (인메모리/로컬 영속성 지원)
- 임베딩 모델: Google Generative AI Embeddings (models/text-embedding-004)
- 검색기(Retriever): LangChain MultiQueryRetriever (다중 질문 생성 검색 알고리즘)
- 프롬프트: LangChain Prompt Hub (rlm/rag-prompt) + 현진건 작가 페르소나 시스템 역할
- 체인 구성: RunnablePassthrough를 활용한 질의 전달 및 확장 파이프라인
- 스트리밍: 실시간 토큰 스트리밍 핸들러 (StreamHandler & Server-Sent Events)
"""

import os
import io
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import pypdf
from dotenv import load_dotenv

# LangChain 핵심 모듈 및 Chroma, MultiQueryRetriever, Prompt Hub, RunnablePassthrough 임포트
from langchain import hub
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# Chroma 벡터스토어 임포트 (신규 langchain_chroma 우선, 호환용 community 폴백)
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# 로컬 .env 환경 변수 로드
load_dotenv()

# 로깅 설정 (다중 질문 생성 과정이 로그에 자세히 출력되도록 설정)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Chroma-RAG-Server")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ChromaDB 저장 경로 및 컬렉션 이름 설정 (Cloud Run에서는 /tmp 디렉토리가 쓰기 가능)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
COLLECTION_NAME = "hyunjingeon_literature_rag"

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="현진건 문학 리얼리즘 RAG AI 서버 (ChromaDB + MultiQueryRetriever)",
    description="소설 『운수 좋은 날』의 현진건 작가 페르소나와 다중 질문 검색기(MultiQueryRetriever)를 결합한 Cloud Run AI 서버",
    version="2.0.0"
)

# CORS 설정 (Google Apps Script 웹앱 및 외부 브라우저에서의 직접 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 1. 실시간 스트리밍 콜백 핸들러 (StreamHandler) 정의
# ==============================================================================
class StreamHandler(BaseCallbackHandler):
    """
    LLM이 생성하는 토큰을 실시간으로 큐(Queue)에 담아 클라이언트로 스트리밍하는 핸들러입니다.
    """
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """새로운 토큰이 생성될 때마다 비동기 큐에 토큰을 삽입합니다."""
        if token:
            self.queue.put_nowait({"event": "token", "token": token})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """생성 완료 시 종료 신호를 큐에 전달합니다."""
        self.queue.put_nowait({"event": "done", "status": "completed"})

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """에러 발생 시 상세 에러 메시지를 큐에 전달합니다."""
        self.queue.put_nowait({"event": "error", "message": str(error)})


# ==============================================================================
# 2. 환경 변수 및 임베딩 / 벡터 DB 초기화 헬퍼 함수
# ==============================================================================
def get_google_api_key() -> str:
    """환경 변수에서 Google AI API 키를 가져옵니다."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 환경 변수에 Google AI API 키(GEMINI_API_KEY 또는 GOOGLE_API_KEY)가 설정되어 있지 않습니다."
        )
    return api_key


def get_google_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    구글 임베딩 모델(models/text-embedding-004) 객체를 생성하여 반환합니다.
    """
    api_key = get_google_api_key()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )


def get_chroma_vectorstore() -> Chroma:
    """
    ChromaDB 벡터 저장소 인스턴스를 로드하거나 생성합니다.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    embeddings = get_google_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )


# ==============================================================================
# Pydantic 요청 모델 정의
# ==============================================================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="사용자의 질의 내용")
    filename: Optional[str] = Field(None, description="특정 PDF 파일명으로 검색 필터링 시 사용")


# ==============================================================================
# 3. 서버 연동 상태 확인(Health Check) 엔드포인트
# ==============================================================================
@app.get("/health", summary="서버 연동 상태 및 ChromaDB 준비 현황 확인")
def health_check():
    """클라이언트(Code.gs / index.html)에서 서버 연결 상태를 확인하는 엔드포인트입니다."""
    api_key_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    doc_count = 0
    try:
        if api_key_ready and os.path.exists(CHROMA_PERSIST_DIR):
            vs = get_chroma_vectorstore()
            doc_count = vs._collection.count()
    except Exception as e:
        logger.warning(f"ChromaDB 상태 조회 중 경미한 오류: {str(e)}")

    return {
        "status": "healthy",
        "service": "Hyun Jin-geon Literature RAG Server",
        "vector_db": "ChromaDB",
        "embedding_model": "models/text-embedding-004",
        "retriever": "MultiQueryRetriever",
        "stored_chunks": doc_count,
        "google_api_configured": api_key_ready
    }


# ==============================================================================
# 4. PDF 텍스트 변환 및 ChromaDB 벡터 DB(RAG) 생성 엔드포인트
# ==============================================================================
@app.post("/create-rag", summary="PDF 파일을 텍스트로 변환하고 청크로 쪼개어 ChromaDB에 저장")
async def create_rag(file: UploadFile = File(...)):
    """
    1. 업로드된 PDF 파일을 페이지별로 읽어 텍스트로 변환합니다.
    2. RecursiveCharacterTextSplitter로 청크(Chunk) 단위로 분할합니다.
    3. Google 임베딩 모델(models/text-embedding-004)로 임베딩합니다.
    4. ChromaDB에 벡터 데이터를 저장하여 RAG 인덱스를 구축합니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"업로드된 파일('{file.filename}')은 PDF 파일이 아닙니다. .pdf 파일을 선택해 주세요."
        )

    try:
        # Step 1: PDF 바이너리 읽기 및 페이지별 텍스트 추출
        pdf_bytes = await file.read()
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        raw_documents: List[Document] = []
        full_text_preview = []

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                # 출처(파일명, 페이지 번호) 메타데이터를 포함하여 Document 생성
                raw_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file.filename,
                            "page": page_idx
                        }
                    )
                )
                full_text_preview.append(f"[페이지 {page_idx}]\n{page_text}")

        if not raw_documents:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF에서 텍스트를 추출할 수 없습니다. 텍스트가 포함된 PDF 파일인지 확인해 주세요."
            )

        # Step 2: 문서를 청크(Chunk) 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(raw_documents)

        # 각 청크에 고유 청크 번호 부여
        for idx, doc in enumerate(split_docs, start=1):
            doc.metadata["chunk_index"] = idx

        logger.info(f"파일 '{file.filename}' ({total_pages}페이지) -> 총 {len(split_docs)}개 청크 분할 완료")

        # Step 3 & 4: Google 임베딩(models/text-embedding-004)을 사용하여 ChromaDB에 적재
        embeddings = get_google_embeddings()
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )

        total_stored = vectorstore._collection.count()

        return {
            "status": "success",
            "message": f"'{file.filename}' 문서가 {len(split_docs)}개의 청크로 분할되어 ChromaDB에 저장되었습니다.",
            "filename": file.filename,
            "total_pages": total_pages,
            "total_chunks": len(split_docs),
            "total_stored_in_db": total_stored,
            "preview_text": "\n\n".join(full_text_preview)[:3000]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ChromaDB RAG 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"벡터 DB(ChromaDB) 생성 중 오류가 발생했습니다: {str(e)}"
        )


# ==============================================================================
# 5. MultiQueryRetriever + Prompt Hub + RunnablePassthrough + 실시간 스트리밍 질의
# ==============================================================================
@app.post("/chat/stream", summary="다중 질문 검색기(MultiQueryRetriever) 및 현진건 작가 페르소나 실시간 스트리밍 답변")
async def chat_stream(request_data: QueryRequest):
    """
    [작동 파이프라인]
    1. ChromaDB 기본 검색기를 불러와 LangChain `MultiQueryRetriever`로 래핑합니다.
       (사용자의 단일 질문을 여러 관점의 다중 질문으로 자동 확장하여 관련 청크를 폭넓게 검색)
    2. LangChain Prompt Hub(`hub.pull("rlm/rag-prompt")`)에서 검증된 RAG 프롬프트를 가져오고,
       현진건 작가의 근대 한국 문학 현실주의 시스템 역할을 결합합니다.
    3. `RunnablePassthrough`를 이용해 사용자 질의와 검색된 문맥을 체인에 주입합니다.
    4. 실시간 스트리밍 핸들러(`StreamHandler` 및 비동기 스트림)를 통해 검색 결과/출처와 LLM 답변을 분리 전송합니다.
    """
    api_key = get_google_api_key()
    vectorstore = get_chroma_vectorstore()

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        try:
            # ChromaDB에 저장된 데이터가 있는지 먼저 확인
            if vectorstore._collection.count() == 0:
                yield json.dumps({
                    "event": "error",
                    "message": "ChromaDB 벡터 저장소가 비어 있습니다. 먼저 PDF 파일을 업로드하고 [벡터 DB 생성] 버튼을 눌러주세요."
                }, ensure_ascii=False) + "\n\n"
                return

            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

            # 다중 질문 생성을 위한 LLM 객체 초기화
            query_gen_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3
            )

            # ------------------------------------------------------------------
            # Step 1. 다중 질문 검색기(MultiQueryRetriever) 구성
            # ------------------------------------------------------------------
            search_kwargs: Dict[str, Any] = {"k": 4}
            if request_data.filename:
                search_kwargs["filter"] = {"source": request_data.filename}

            base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            # MultiQueryRetriever: 사용자의 질문을 다양한 시각의 질문들로 확장하여 검색 수행
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=query_gen_llm
            )

            # 다중 질문 검색 실행하여 관련 문서 청크 추출
            retrieved_docs: List[Document] = await multi_query_retriever.ainvoke(request_data.query)

            # 검색된 청크 및 출처 정보를 정리하여 클라이언트로 먼저 전송 (답변 영역과 분리 표시)
            sources_list = []
            for idx, doc in enumerate(retrieved_docs, start=1):
                sources_list.append({
                    "rank": idx,
                    "source": doc.metadata.get("source", "PDF 문서"),
                    "page": doc.metadata.get("page", "-"),
                    "chunk_index": doc.metadata.get("chunk_index", idx),
                    "content": doc.page_content
                })

            yield json.dumps({
                "event": "sources",
                "retrieved_count": len(sources_list),
                "sources": sources_list
            }, ensure_ascii=False) + "\n\n"

            # ------------------------------------------------------------------
            # Step 2. 프롬프트 허브(LangChain Hub) 로드 및 현진건 작가 시스템 역할 결합
            # ------------------------------------------------------------------
            system_role_instruction = (
                "당신은 소설 운수 좋은날을 집필한 현진건 작가님입니다. "
                "근대 한국 문학의 현실주의 문제로 사용자의 질문에 답변하세요.\n"
                "반드시 아래 제공된 [검색된 문서 내용(Context)]에 근거하여 당시 식민지 조선의 궁핍한 민중의 삶과 "
                "사실주의(리얼리즘) 문학관, 아이러니의 미학이 드러나도록 깊이 있고 품격 있는 어조로 답변하세요. "
                "답변 시 참조한 문서의 출처(페이지 및 청크 번호)를 함께 언급해 주세요."
            )

            # LangChain Hub에서 표준 RAG 프롬프트("rlm/rag-prompt") 가져오기
            try:
                hub_prompt = hub.pull("rlm/rag-prompt")
                # 프롬프트 허브 템플릿에 현진건 작가 시스템 메시지를 결합
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_role_instruction),
                    *hub_prompt.messages
                ])
            except Exception as hub_err:
                logger.warning(f"LangChain Hub 접속 폴백 처리: {str(hub_err)}")
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_role_instruction),
                    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
                ])

            # 검색된 Document 리스트를 하나의 문맥 문자열로 포맷팅하는 함수
            def format_docs(docs: List[Document]) -> str:
                if not docs:
                    return "검색된 관련 문서 내용이 없습니다."
                formatted = []
                for i, d in enumerate(docs, start=1):
                    src = d.metadata.get("source", "문서")
                    page = d.metadata.get("page", "?")
                    c_idx = d.metadata.get("chunk_index", i)
                    formatted.append(f"[출처 #{i}: {src} - {page}페이지 (청크 #{c_idx})]\n{d.page_content}")
                return "\n\n---\n\n".join(formatted)

            # ------------------------------------------------------------------
            # Step 3. RunnablePassthrough를 사용한 RAG 체인 파이프라인 구성
            # ------------------------------------------------------------------
            # 실시간 스트리밍용 큐 및 StreamHandler 장착
            token_queue: asyncio.Queue = asyncio.Queue()
            stream_handler = StreamHandler(token_queue)

            streaming_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.4,
                streaming=True,
                callbacks=[stream_handler]
            )

            # RunnablePassthrough로 사용자의 질의를 그대로 투과시키고,
            # context에는 이미 MultiQueryRetriever로 검색된 문서들을 포맷팅하여 주입
            rag_chain = (
                {
                    "context": RunnableLambda(lambda _: format_docs(retrieved_docs)),
                    "question": RunnablePassthrough()
                }
                | rag_prompt
                | streaming_llm
                | StrOutputParser()
            )

            # ------------------------------------------------------------------
            # Step 4. 비동기 실시간 스트리밍 응답 송출
            # ------------------------------------------------------------------
            async for token_chunk in rag_chain.astream(request_data.query):
                if token_chunk:
                    yield json.dumps({
                        "event": "token",
                        "token": token_chunk
                    }, ensure_ascii=False) + "\n\n"

            # 스트리밍 종료 이벤트 전송
            yield json.dumps({
                "event": "done",
                "status": "completed"
            }, ensure_ascii=False) + "\n\n"

        except Exception as e:
            logger.error(f"스트리밍 질의 처리 중 에러: {str(e)}", exc_info=True)
            yield json.dumps({
                "event": "error",
                "message": f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            }, ensure_ascii=False) + "\n\n"

    return EventSourceResponse(sse_event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
