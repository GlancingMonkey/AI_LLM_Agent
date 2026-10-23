import os
import io
import json
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
# 기존 GoogleGenerativeAIEmbeddings는 지우고 아래 줄을 추가합니다.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

# 환경 변수에서 구글 API 키 로드
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Cloud Run은 컨테이너 종료 시 데이터가 초기화되므로 임시 폴더를 사용
CHROMA_PERSIST_DIR = "/tmp/chromadb"

# 에러가 나던 embeddings = GoogleGenerativeAIEmbeddings(...) 코드를 완전히 지우고 아래로 교체합니다.
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # 또는 "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    streaming=True
)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트를 추출하고 Chroma DB에 저장합니다."""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
            
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                full_text += page.extract_text() + "\n"

        # 텍스트 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        metadatas = [{"source": file.filename} for _ in chunks]

        # Chroma 벡터 DB에 저장 (기존 데이터 초기화 후 덮어쓰기)
        vector_store = Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings, 
            metadatas=metadatas,
            persist_directory=CHROMA_PERSIST_DIR
        )

        return {"message": "벡터 DB 생성 완료", "chunks_count": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """사용자의 질의를 받아 MultiQueryRetriever로 검색 후 현진건 스타일로 답변을 스트리밍합니다."""
    try:
        # 저장된 Chroma DB 불러오기
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 다중 질문 검색기(MultiQueryRetriever) 적용 - 유사한 질문을 여러 개 생성하여 검색 품질 향상
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=llm
        )

        # 시스템 역할 부여 프롬프트 (현진건 작가 페르소나)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. 근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요. 주어진 참고 문서에 없는 내용은 모른다고 답해야 합니다.\n\n[참고 문서]\n{context}"),
            ("user", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 질의 확장 및 답변 생성 체인 (RunnablePassthrough 사용)
        rag_chain = (
            {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async def generate():
            # 1. 먼저 검색된 문서(Context)와 출처를 클라이언트에 전송
            docs = retriever_from_llm.invoke(request.query)
            context_data = [d.page_content for d in docs]
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            
            yield f"data: {json.dumps({'type': 'context', 'sources': sources, 'context': context_data})}\n\n"
            
            # 2. LLM의 답변을 실시간 스트리밍(StreamHandler 역할)
            async for chunk in rag_chain.astream(request.query):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
