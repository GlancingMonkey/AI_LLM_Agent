import os
import io
import json
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# 프론트엔드(GAS)에서의 직접 호출을 허용하기 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드 (Cloud Run 환경 변수에서 모델명도 유연하게 변경 가능하도록 설정)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # 필요시 "gemini-3.8-flash-lite" 등으로 변경

# Cloud Run 임시 폴더 경로
CHROMA_PERSIST_DIR = "/tmp/chromadb"

# 1. 임베딩 모델 지연 로딩 (Cloud Run 시작 시 포트 8080 타임아웃 방지)
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )
    return _embeddings

def get_llm(streaming: bool = True):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        streaming=streaming
    )

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def health_check():
    """Cloud Run 헬스체크용 기본 엔드포인트"""
    return {"status": "ok", "model": GEMINI_MODEL}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트를 추출하고 Chroma DB에 저장합니다."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        # 텍스트를 청크 단위로 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        metadatas = [{"source": file.filename} for _ in chunks]

        # 기존 DB 폴더가 있으면 충돌 방지를 위해 정리 후 새로 생성
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)

        embeddings = get_embeddings()
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=CHROMA_PERSIST_DIR
        )

        return {"message": "벡터 DB 생성 완료", "chunks_count": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류: {str(e)}")

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """사용자의 질의를 받아 MultiQueryRetriever로 검색 후 현진건 스타일로 답변을 스트리밍합니다."""

    async def generate():
        try:
            # DB 존재 여부 먼저 확인 (Cloud Run 재시작으로 초기화되었을 경우 방어)
            if not os.path.exists(CHROMA_PERSIST_DIR):
                err_msg = "⚠️ 아직 업로드된 문서가 없소이다. 먼저 PDF 파일을 올리고 [벡터 DB 생성]을 눌러주시오."
                yield f"data: {json.dumps({'type': 'chunk', 'content': err_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            embeddings = get_embeddings()
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # 다중 질문 생성용 LLM(비스트리밍)과 답변 생성용 LLM(스트리밍) 분리
            query_llm = get_llm(streaming=False)
            chat_llm = get_llm(streaming=True)

            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=query_llm
            )

            # 1. 문서 검색 (딱 1번만 실행하여 API 호출 횟수 절약)
            docs = await retriever_from_llm.ainvoke(request.query)
            context_data = [d.page_content for d in docs]
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

            # 검색된 문서(Context)와 출처를 클라이언트에 먼저 전송
            yield f"data: {json.dumps({'type': 'context', 'sources': sources, 'context': context_data}, ensure_ascii=False)}\n\n"

            # 2. 검색된 문서를 바탕으로 현진건 페르소나 답변 스트리밍
            prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. 근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요. 주어진 참고 문서에 없는 내용은 모른다고 답해야 합니다.\n\n[참고 문서]\n{context}"),
                ("user", "{question}")
            ])

            formatted_context = "\n\n".join(context_data)
            answer_chain = prompt | chat_llm | StrOutputParser()

            async for chunk in answer_chain.astream({
                "context": formatted_context,
                "question": request.query
            }):
                if chunk:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            traceback.print_exc()
            # 에러 발생 시 무한 로딩에 빠지지 않도록 채팅창에 에러 원인을 바로 전송
            error_text = f"\n\n❌ [서버 오류 발생]: {str(e)}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': error_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
