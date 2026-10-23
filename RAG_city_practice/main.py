import os
import io
import json
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from supabase import create_client, Client

# OpenAIEmbeddings 삭제 및 GoogleGenerativeAIEmbeddings 추가
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 구글 임베딩 모델로 교체 (768 차원)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0, 
    streaming=True
)

class ChatRequest(BaseModel):
    query: str
    chat_history: list[dict]

@app.post("/api/v2/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
            
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                full_text += page.extract_text() + "\n"
                
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="텍스트를 추출할 수 없습니다.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        metadatas = [{"source": file.filename} for _ in chunks]

        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="policy_docs_v2",
            query_name="search_policy_docs_v2"
        )
        vector_store.add_texts(texts=chunks, metadatas=metadatas)

        return {"message": "구글 임베딩 RAG 벡터 DB 생성 완료", "chunks_count": len(chunks), "extracted_text": full_text[:1000] + "..."}
        
    except Exception as e:
        error_msg = f"에러 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v2/chat")
async def chat_stream(request: ChatRequest):
    try:
        history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history.append(AIMessage(content=msg["content"]))

        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 '이', '저', '그'와 같은 표현을 명확한 명사로 표현하라. 단, 대화 기록이 없다면 사용자의 질문을 그대로 반환하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{query}")
        ])
        
        query_augmentation_chain = condense_prompt | llm | StrOutputParser()
        augmented_query = query_augmentation_chain.invoke({
            "query": request.query,
            "chat_history": history
        })

        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="policy_docs_v2",
            query_name="search_policy_docs_v2"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(augmented_query)
        
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n[참고 문서]\n{context}\n\n위 참고 문서를 바탕으로 사용자의 질문에 전문적으로 답변해. 문서에 없는 내용은 모른다고 답해."),
            ("user", "{query}")
        ])
        qa_chain = qa_prompt | llm | StrOutputParser()

        async def generate():
            try:
                meta_info = {
                    "type": "metadata",
                    "augmented_query": augmented_query,
                    "sources": sources,
                    "context_preview": [d.page_content[:100] + "..." for d in docs]
                }
                yield f"data: {json.dumps(meta_info)}\n\n"
                
                async for chunk in qa_chain.astream({"context": context, "query": augmented_query}):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as stream_e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(stream_e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        error_msg = f"에러 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)
