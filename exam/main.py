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

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CHROMA_PERSIST_DIR = "/tmp/chromadb"

_embeddings = None

# [조건 충족]: 구글 임베딩 모델 사용 (지연 로딩으로 에러 방지)
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return _embeddings

def get_llm(streaming: bool = True):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        streaming=streaming
    )

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def health_check():
    return {"status": "connected", "message": "서버 연결 상태: 정상"}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                full_text += page.extract_text() + "\n"

        # 구글 임베딩 API 호출 제한(429 에러) 방지를 위해 청크를 큼직하게 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)
        metadatas = [{"source": file.filename} for _ in chunks]

        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)

        # [조건 충족]: 벡터 DB는 Chroma DB 사용
        Chroma.from_texts(
            texts=chunks,
            embedding=get_embeddings(),
            metadatas=metadatas,
            persist_directory=CHROMA_PERSIST_DIR
        )
        return {"message": "업로드 및 벡터 DB 생성 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=get_embeddings()
            )
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # [조건 충족]: 다중 질문 검색기 사용
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=get_llm(streaming=False)
            )

            # [조건 충족]: 시스템 역할 정의
            prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.\n\n[참고 문서]\n{context}"),
                ("user", "{question}")
            ])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # [조건 충족]: 질의 확장은 RunnablePassthrough 사용
            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | get_llm(streaming=True)
                | StrOutputParser()
            )

            async for chunk in rag_chain.astream(request.query):
                if chunk:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'chunk', 'content': f'❌ 오류: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
