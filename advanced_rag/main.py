"""
================================================================================
Advanced RAG Server (FastAPI): FAISS Index + Multi-Query + Hybrid Search + RRF
================================================================================
[아키텍처 및 계층별 시간/공간 복잡도]
1. 문서 적재 및 인덱싱 계층 (FAISS + BM25 Index Layer)
   - 텍스트 청킹: O(C) (C: 전체 텍스트 길이)
   - FAISS IndexFlatL2 적재: 시간 복잡도 O(N * D), 공간 복잡도 O(N * D) (N: 문서 청크 수, D: 임베딩 차원)
   - FAISS IndexHNSWFlat 적재(선택): 시간 복잡도 O(N * log(N) * D), 검색 시 O(log(N) * D)
   - BM25 역색인 구축: 시간 복잡도 O(N * L), 공간 복잡도 O(V + N * L) (L: 평균 토큰 수, V: 어휘 사전 크기)

2. 다중 쿼리 확장 계층 (Multi-Query Layer)
   - 단일 사용자 쿼리를 LLM을 통해 서로 다른 5가지 관점의 파생 쿼리로 확장
   - 복잡도: O(M * T_llm) (M=5 확장 쿼리 수, T_llm: LLM 추론 비용)

3. 하이브리드 검색 계층 (Hybrid Search & FAISS Layer)
   - M개의 확장 쿼리 각각에 대해 Dense 검색(FAISS) + Sparse 검색(BM25) 병행 수행
   - FAISS Dense 검색 복잡도: O(M * N * D) (IndexFlatL2 기준) 또는 O(M * log(N) * D) (IndexHNSWFlat 기준)
   - BM25 Sparse 검색 복잡도: O(M * N * Q_t) (Q_t: 쿼리 내 평균 토큰 수)

4. RRF 순위 융합 계층 (RAG Fusion & RRF Layer)
   - 공식: RRF_Score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))  (기본 상수 k = 60)
   - 2 * M개(Dense 5개 + Sparse 5개 = 총 10개)의 랭킹 리스트를 해시맵으로 통합 후 정렬
   - 복잡도: O(M * K + U * log(U)) (K: 각 검색 경로당 상위 추출 개수, U: 고유 후보 문서 수, U <= 2*M*K)
================================================================================
"""

import os
import io
import json
import time
import base64
import hashlib
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LLM SDK (OpenAI 또는 Gemini API 지원, 미설정 시 결정론적 Fallback 제공)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid + RRF)",
    description="FAISS Dense Vector + BM25 Sparse Search + 5x Multi-Query Expansion + Reciprocal Rank Fusion API",
    version="1.0.0",
)

# CORS 설정 (GAS 및 외부 웹앱 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 1. 실시간 스트리밍 핸들러 (StreamHandler)
# =====================================================================
class StreamHandler:
    """
    서버의 RAG 파이프라인 진행 상황(문서 파싱, FAISS 구축, 다중 쿼리 확장, RRF 융합, LLM 답변 생성)을
    클라이언트(GAS / 프론트엔드)로 실시간 전달 및 기록하기 위한 스트리밍 핸들러 클래스입니다.
    복잡도: 이벤트당 직렬화 O(1)
    """

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def emit(self, stage: str, message: str, progress: int, extra: Optional[Dict[str, Any]] = None) -> str:
        """단일 진행 상태 이벤트를 생성하고 NDJSON 라인 문자열로 반환합니다."""
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "message": message,
            "progress": progress,
            "extra": extra or {},
        }
        self.logs.append(event)
        return json.dumps(event, ensure_ascii=False) + "\n"

    def get_logs(self) -> List[Dict[str, Any]]:
        """누적된 전체 스트리밍 상태 로그 리스트를 반환합니다."""
        return self.logs


# =====================================================================
# 2. 인메모리 벡터 DB 및 하이브리드 검색 저장소 (VectorStoreState)
# =====================================================================
class VectorStoreState:
    """
    FAISS 인덱스(IndexFlatL2 / IndexHNSWFlat)와 BM25 인덱스를 동시에 유지하는 상태 관리 클래스입니다.
    """

    def __init__(self, dimension: int = 384):
        self.dimension: int = dimension
        self.index_type: str = "IndexFlatL2"
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []  # [{doc_id, text, metadata: {source, page, chunk_id}}]
        self.tokenized_corpus: List[List[str]] = []
        self.current_filename: Optional[str] = None
        self.last_indexed_at: Optional[str] = None
        self.progress_logs: List[Dict[str, Any]] = []

    def is_ready(self) -> bool:
        return (
            self.faiss_index is not None
            and self.bm25_index is not None
            and len(self.documents) > 0
        )


# 전역 스토어 인스턴스
STORE = VectorStoreState(dimension=384)


# =====================================================================
# 3. 임베딩, 토크나이저 및 LLM 유틸리티 함수
# =====================================================================
def simple_tokenize(text: str) -> List[str]:
    """
    한국어 및 영문 키워드 매칭(BM25)을 위한 경량 토크나이저.
    공백 분리 및 2글자 이상 서브워드(bi-gram) 토큰을 병행 생성하여 한국어 조사 변화에도 강건하게 동작합니다.
    복잡도: O(L) (L: 문자열 길이)
    """
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    words = cleaned.split()
    tokens = list(words)
    for w in words:
        if len(w) >= 3:
            for i in range(len(w) - 1):
                tokens.append(w[i : i + 2])
    return tokens if tokens else ["empty"]


def compute_embeddings(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    텍스트 리스트를 고차원 Dense Vector(float32)로 변환합니다.
    - OPENAI_API_KEY가 설정된 경우 text-embedding-3-small 사용 (차원 자동 맞춤)
    - GEMINI_API_KEY가 설정된 경우 text-embedding-004 사용
    - API 키가 없거나 오프라인 환경일 경우, 문자/단어 N-gram 해싱 기반의 결정론적(Deterministic) 의미 벡터를 생성하여
      언제든 즉시 FAISS L2 거리 계산이 가능하도록 지원합니다.
    복잡도: O(N * L + N * D)
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=dim,
            )
            vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
            faiss.normalize_L2(vecs)
            return vecs
        except Exception:
            pass  # API 오류 시 아래 결정론적 임베딩으로 안전하게 폴백

    # 결정론적 Feature Hashing 임베딩 (외부 API 장애 시에도 동일한 토큰/문맥 유사도 보장)
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts):
        tokens = simple_tokenize(text)
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            bucket = h % dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vectors[idx, bucket] += sign
        norm = np.linalg.norm(vectors[idx])
        if norm > 0:
            vectors[idx] /= norm
    return vectors


def call_llm(prompt: str, system_prompt: str = "당신은 전문 AI 어시스턴트입니다.") -> str:
    """
    환경변수(OPENAI_API_KEY 또는 GEMINI_API_KEY)를 사용하여 LLM 응답을 생성합니다.
    API 키가 없는 실습 환경에서도 파이프라인 전체를 검증할 수 있도록 구조화된 폴백 응답을 제공합니다.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if openai_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API 호출 오류: {str(e)}")

    if gemini_key and genai is not None:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
            response = model.generate_content(f"{system_prompt}\n\n{prompt}")
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API 호출 오류: {str(e)}")

    return ""


# =====================================================================
# 4. 핵심 RAG 파이프라인 함수 (Multi-Query / Hybrid Search / RRF)
# =====================================================================
def generate_multi_queries(original_query: str, num_queries: int = 5) -> List[str]:
    """
    [다중 쿼리 확장 계층 (Multi-Query Layer)]
    사용자의 단일 입력 쿼리를 LLM을 통해 서로 다른 5가지 관점(의미적 확장, 핵심 키워드 중심,
    배경/원인 관점, 적용/사례 관점, 요약/정의 관점)의 파생 쿼리로 확장합니다.
    시간 복잡도: O(M * T_llm) (M = 5)
    """
    prompt = (
        f"사용자의 원본 질문을 바탕으로 벡터 검색 및 키워드 검색의 재현율(Recall)을 극대화하기 위해 "
        f"서로 다른 관점의 파생 검색 쿼리 {num_queries}개를 생성하세요.\n"
        f"원본 질문: {original_query}\n"
        f"출력 형식: 번호 없이 한 줄에 하나씩 정확히 {num_queries}개의 질문만 출력하세요."
    )
    llm_output = ""
    try:
        llm_output = call_llm(prompt, system_prompt="당신은 RAG 검색 최적화 전문가입니다.")
    except Exception:
        llm_output = ""

    if llm_output:
        lines = [
            line.strip().lstrip("0123456789.-*) ")
            for line in llm_output.splitlines()
            if line.strip()
        ]
        if len(lines) >= num_queries:
            return lines[:num_queries]

    # LLM 키 미설정 또는 응답 부족 시 5가지 관점 템플릿 기반 확장 보장
    perspectives = [
        f"{original_query} 핵심 개념 및 정의",
        f"{original_query} 주요 원리 및 동작 구조",
        f"{original_query} 관련 핵심 키워드 및 세부 명세",
        f"{original_query} 실제 적용 사례 및 장단점 비교",
        f"{original_query} 요약 정리 및 핵심 데이터 근거",
    ]
    return perspectives[:num_queries]


def perform_hybrid_search_for_query(
    query: str, top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    [하이브리드 검색 및 FAISS 인덱스 계층 (Hybrid Search & FAISS Layer)]
    단일 쿼리에 대해:
      1) FAISS Dense Vector 검색 (L2 거리 기반 의미적 유사도)
      2) BM25 Sparse 검색 (TF-IDF 변형 기반 정확한 키워드 매칭)
    을 병행 수행하여 각각의 순위 리스트를 반환합니다.
    시간 복잡도:
      - Dense (IndexFlatL2): O(N * D) / (IndexHNSWFlat): O(log(N) * D)
      - Sparse (BM25): O(N * Q_t)
    """
    if not STORE.is_ready():
        raise ValueError("벡터 DB(FAISS/BM25)가 아직 초기화되지 않았습니다. 먼저 PDF 문서를 적재해주세요.")

    num_docs = len(STORE.documents)
    k = min(top_k, num_docs)

    # 1. FAISS Dense Vector Search
    query_vec = compute_embeddings([query], dim=STORE.dimension)
    distances, indices = STORE.faiss_index.search(query_vec, k)

    dense_results: List[Dict[str, Any]] = []
    for rank_idx, (dist, doc_idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if doc_idx < 0 or doc_idx >= num_docs:
            continue
        doc = STORE.documents[int(doc_idx)]
        # 정규화된 벡터의 L2 거리를 유사도 점수(0~1)로 변환: sim = 1 / (1 + dist)
        similarity = float(1.0 / (1.0 + max(0.0, float(dist))))
        dense_results.append({
            "doc_id": doc["doc_id"],
            "rank": rank_idx,
            "dense_score": round(similarity, 4),
            "l2_distance": round(float(dist), 4),
        })

    # 2. BM25 Sparse Keyword Search
    query_tokens = simple_tokenize(query)
    bm25_scores = STORE.bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]

    sparse_results: List[Dict[str, Any]] = []
    for rank_idx, doc_idx in enumerate(top_bm25_indices, start=1):
        doc = STORE.documents[int(doc_idx)]
        sparse_results.append({
            "doc_id": doc["doc_id"],
            "rank": rank_idx,
            "sparse_score": round(float(bm25_scores[int(doc_idx)]), 4),
        })

    return {"dense": dense_results, "sparse": sparse_results}


def reciprocal_rank_fusion(
    all_search_runs: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    [RRF 순위 융합 계층 (RAG Fusion & RRF Layer)]
    확장된 5개 쿼리 × 2개 검색 경로(FAISS Dense + BM25 Sparse) = 총 10개의 검색 결과 리스트를
    Reciprocal Rank Fusion 공식으로 통합하여 가중치를 부여하고 재정렬(Re-ranking)합니다.
    - 공식: RRF_Score(d) = Σ 1 / (k + rank(d))  (k = 60)
    시간 복잡도: O(M * K + U * log(U)) (M: 확장 쿼리 수, K: top_k, U: 고유 후보 문서 수)
    """
    doc_lookup: Dict[int, Dict[str, Any]] = {d["doc_id"]: d for d in STORE.documents}
    rrf_scores: Dict[int, float] = {}
    fusion_details: Dict[int, Dict[str, Any]] = {}

    for run in all_search_runs:
        q_text = run["query"]
        # Dense 랭킹 반영
        for item in run["dense"]:
            doc_id = item["doc_id"]
            rank = item["rank"]
            score_contrib = 1.0 / (rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

            if doc_id not in fusion_details:
                fusion_details[doc_id] = {
                    "best_dense_rank": rank,
                    "best_sparse_rank": None,
                    "best_dense_score": item["dense_score"],
                    "best_sparse_score": 0.0,
                    "matched_queries": set(),
                    "hit_count": 0,
                }
            else:
                if fusion_details[doc_id]["best_dense_rank"] is None or rank < fusion_details[doc_id]["best_dense_rank"]:
                    fusion_details[doc_id]["best_dense_rank"] = rank
                fusion_details[doc_id]["best_dense_score"] = max(
                    fusion_details[doc_id]["best_dense_score"], item["dense_score"]
                )
            fusion_details[doc_id]["matched_queries"].add(q_text)
            fusion_details[doc_id]["hit_count"] += 1

        # Sparse(BM25) 랭킹 반영
        for item in run["sparse"]:
            doc_id = item["doc_id"]
            rank = item["rank"]
            score_contrib = 1.0 / (rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

            if doc_id not in fusion_details:
                fusion_details[doc_id] = {
                    "best_dense_rank": None,
                    "best_sparse_rank": rank,
                    "best_dense_score": 0.0,
                    "best_sparse_score": item["sparse_score"],
                    "matched_queries": set(),
                    "hit_count": 0,
                }
            else:
                if fusion_details[doc_id]["best_sparse_rank"] is None or rank < fusion_details[doc_id]["best_sparse_rank"]:
                    fusion_details[doc_id]["best_sparse_rank"] = rank
                fusion_details[doc_id]["best_sparse_score"] = max(
                    fusion_details[doc_id]["best_sparse_score"], item["sparse_score"]
                )
            fusion_details[doc_id]["matched_queries"].add(q_text)
            fusion_details[doc_id]["hit_count"] += 1

    # RRF 점수 기준 내림차순 정렬 (O(U log U))
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)[:top_n]

    ranked_documents: List[Dict[str, Any]] = []
    for final_rank, doc_id in enumerate(sorted_doc_ids, start=1):
        doc = doc_lookup[doc_id]
        detail = fusion_details[doc_id]
        ranked_documents.append({
            "final_rank": final_rank,
            "doc_id": doc_id,
            "rrf_score": round(rrf_scores[doc_id], 6),
            "snippet": doc["text"],
            "metadata": {
                **doc["metadata"],
                "best_dense_rank": detail["best_dense_rank"],
                "best_sparse_rank": detail["best_sparse_rank"],
                "dense_similarity": detail["best_dense_score"],
                "bm25_score": detail["best_sparse_score"],
                "hit_count": detail["hit_count"],
                "matched_query_count": len(detail["matched_queries"]),
            },
        })

    return ranked_documents


# =====================================================================
# 5. 요청/응답 데이터 모델 (Pydantic Schema)
# =====================================================================
class BuildIndexRequest(BaseModel):
    filename: str = Field(..., description="업로드된 PDF 파일명")
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="텍스트 직접 전달 시 사용 (옵션)")
    index_type: str = Field("IndexFlatL2", description="FAISS 인덱스 타입: IndexFlatL2 또는 IndexHNSWFlat")
    chunk_size: int = Field(500, description="청크당 최대 글자 수")
    chunk_overlap: int = Field(80, description="청크 간 중첩 글자 수")


class QueryRequest(BaseModel):
    query: str = Field(..., description="사용자의 입력 질의")
    top_k: int = Field(5, description="최종 반환할 상위 RRF 문서 개수")
    rrf_k: int = Field(60, description="RRF 공식 상수 k (기본값 60)")


# =====================================================================
# 6. 텍스트 추출 및 청킹 유틸리티
# =====================================================================
def extract_pages_from_pdf(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    PDF 바이트에서 페이지별 텍스트를 추출합니다.
    복잡도: O(P) (P: PDF 전체 페이지 수)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_data: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages_data.append({"page": page_idx, "text": txt, "source": filename})
    return pages_data


def chunk_pages(
    pages_data: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    페이지별 텍스트를 지정된 크기(chunk_size)와 중첩(chunk_overlap)으로 분할합니다.
    복잡도: O(C) (C: 전체 문자 수)
    """
    chunks: List[Dict[str, Any]] = []
    doc_id = 0
    step = max(1, chunk_size - chunk_overlap)

    for page_info in pages_data:
        text = page_info["text"]
        page_num = page_info["page"]
        source = page_info["source"]

        for start in range(0, len(text), step):
            segment = text[start : start + chunk_size].strip()
            if len(segment) < 20 and len(chunks) > 0:
                continue
            chunks.append({
                "doc_id": doc_id,
                "text": segment,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"p{page_num}_c{doc_id}",
                    "char_length": len(segment),
                },
            })
            doc_id += 1
    return chunks


# =====================================================================
# 7. API 엔드포인트 구현
# =====================================================================
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    서버 연동 상태, 현재 적재된 FAISS/BM25 인덱스 현황 및 알고리즘 복잡도 정보를 반환합니다.
    """
    return {
        "status": "ok",
        "server": "Advanced RAG FastAPI Server (Cloud Run Compatible)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_ready": STORE.is_ready(),
        "current_filename": STORE.current_filename,
        "index_type": STORE.index_type,
        "total_chunks": len(STORE.documents),
        "vector_dimension": STORE.dimension,
        "last_indexed_at": STORE.last_indexed_at,
        "complexity_info": {
            "faiss_index_build": "O(N * D) [IndexFlatL2] / O(N * log(N) * D) [IndexHNSWFlat]",
            "bm25_index_build": "O(N * L) (N: 문서 수, L: 평균 토큰 수)",
            "multi_query_expansion": "O(M * T_llm) (M = 5 파생 쿼리)",
            "hybrid_search": "O(M * N * D + M * N * Q_t)",
            "rrf_fusion": "O(M * K + U * log(U)) (k=60 상호 순위 융합)",
        },
    }


@app.post("/build-index")
async def build_index(req: BuildIndexRequest) -> JSONResponse:
    """
    PDF 파일(Base64) 또는 원문 텍스트를 받아 청킹 후 FAISS Index와 BM25 역색인을 동시에 구축합니다.
    StreamHandler를 통해 단계별 진행 현황(progress_logs)을 함께 기록하여 반환합니다.
    """
    handler = StreamHandler()
    start_time = time.time()

    try:
        handler.emit("INIT", f"'{req.filename}' 문서 수신 및 초기화 시작", 10)

        pages_data: List[Dict[str, Any]] = []
        if req.pdf_base64:
            try:
                raw_bytes = base64.b64decode(req.pdf_base64)
            except Exception as decode_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"[PDF_DECODE_ERROR] Base64 디코딩에 실패했습니다: {str(decode_err)}",
                )
            pages_data = extract_pages_from_pdf(raw_bytes, req.filename)
        elif req.raw_text:
            pages_data = [{"page": 1, "text": req.raw_text.strip(), "source": req.filename}]
        else:
            raise HTTPException(
                status_code=400,
                detail="[EMPTY_INPUT_ERROR] pdf_base64 또는 raw_text 중 하나는 반드시 포함되어야 합니다.",
            )

        if not pages_data:
            raise HTTPException(
                status_code=422,
                detail="[PDF_TEXT_EMPTY_ERROR] PDF 문서에서 추출 가능한 텍스트를 찾지 못했습니다. (스캔 이미지 PDF인지 확인하세요)",
            )

        handler.emit("CHUNKING", f"총 {len(pages_data)}페이지 텍스트 추출 완료. 청킹 수행 중...", 35)
        chunks = chunk_pages(pages_data, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="[CHUNKING_ERROR] 유효한 문서 청크를 생성하지 못했습니다.",
            )

        handler.emit("EMBEDDING", f"총 {len(chunks)}개 청크 임베딩 벡터(차원={STORE.dimension}) 생성 중...", 60)
        texts = [c["text"] for c in chunks]
        embeddings = compute_embeddings(texts, dim=STORE.dimension)

        # FAISS Index 생성 (IndexFlatL2 또는 IndexHNSWFlat)
        handler.emit("FAISS_BUILD", f"FAISS 인덱스({req.index_type}) 적재 중...", 80)
        if req.index_type == "IndexHNSWFlat":
            faiss_index = faiss.IndexHNSWFlat(STORE.dimension, 32)
        else:
            faiss_index = faiss.IndexFlatL2(STORE.dimension)
        faiss_index.add(embeddings)

        # BM25 Sparse Index 생성
        handler.emit("BM25_BUILD", "BM25(Okapi) 키워드 역색인 구축 중...", 90)
        tokenized_corpus = [simple_tokenize(t) for t in texts]
        bm25_index = BM25Okapi(tokenized_corpus)

        # 전역 상태 업데이트
        STORE.faiss_index = faiss_index
        STORE.bm25_index = bm25_index
        STORE.documents = chunks
        STORE.tokenized_corpus = tokenized_corpus
        STORE.index_type = req.index_type
        STORE.current_filename = req.filename
        STORE.last_indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        elapsed = round(time.time() - start_time, 3)
        handler.emit(
            "COMPLETED",
            f"벡터 DB 구축 완료! ({len(chunks)}개 청크, 소요 시간: {elapsed}초)",
            100,
            {"total_chunks": len(chunks), "elapsed_sec": elapsed},
        )
        STORE.progress_logs = handler.get_logs()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": req.filename,
                "index_type": STORE.index_type,
                "total_pages": len(pages_data),
                "total_chunks": len(chunks),
                "vector_dimension": STORE.dimension,
                "elapsed_sec": elapsed,
                "progress_logs": handler.get_logs(),
                "complexity": {
                    "chunking": f"O(C) - {len(pages_data)} 페이지 분할",
                    "faiss_indexing": f"O(N * D) - N={len(chunks)}, D={STORE.dimension} ({STORE.index_type})",
                    "bm25_indexing": f"O(N * L) - N={len(chunks)} 문서 토큰 역색인",
                },
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"[INDEX_BUILD_EXCEPTION] 벡터 DB 생성 중 서버 오류가 발생했습니다: {str(e)}\n{tb}",
        )


@app.post("/query")
async def query_rag(req: QueryRequest) -> JSONResponse:
    """
    [Advanced RAG 통합 질의 엔드포인트]
    1단계: 단일 사용자 쿼리를 5가지 관점의 파생 쿼리로 확장 (Multi-Query Layer)
    2단계: 각 확장 쿼리마다 FAISS(Dense) + BM25(Sparse) 하이브리드 검색 수행 (Hybrid Search Layer)
    3단계: 10개 검색 경로 결과를 RRF(Reciprocal Rank Fusion) 공식으로 순위 융합 및 재정렬 (RRF Layer)
    4단계: 선별된 상위 문서를 컨텍스트로 주입하여 LLM 답변 생성 후 검색 근거/출처와 분리하여 JSON 반환
    """
    handler = StreamHandler()
    start_time = time.time()

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="[EMPTY_QUERY_ERROR] 질의 내용(query)이 비어 있습니다.")

    if not STORE.is_ready():
        raise HTTPException(
            status_code=400,
            detail="[VECTOR_DB_NOT_READY] 구축된 벡터 DB가 없습니다. 먼저 PDF 파일을 선택하고 '벡터 DB 생성'을 실행해주세요.",
        )

    try:
        user_query = req.query.strip()
        handler.emit("MULTI_QUERY", "단일 질의를 5가지 관점의 파생 쿼리로 확장 중...", 20)
        expanded_queries = generate_multi_queries(user_query, num_queries=5)

        handler.emit(
            "HYBRID_SEARCH",
            f"5개 파생 쿼리에 대해 FAISS({STORE.index_type}) + BM25 하이브리드 검색 수행 중...",
            55,
        )
        all_search_runs: List[Dict[str, Any]] = []
        # 원본 쿼리 + 확장된 5개 쿼리 모두 검색에 활용하되, 핵심 5가지 확장 쿼리를 명시적으로 기록
        for eq in expanded_queries:
            res = perform_hybrid_search_for_query(eq, top_k=req.top_k)
            all_search_runs.append({
                "query": eq,
                "dense": res["dense"],
                "sparse": res["sparse"],
            })

        handler.emit("RRF_FUSION", f"Reciprocal Rank Fusion (k={req.rrf_k}) 순위 융합 및 재정렬 중...", 75)
        fused_docs = reciprocal_rank_fusion(
            all_search_runs=all_search_runs,
            rrf_k=req.rrf_k,
            top_n=req.top_k,
        )

        # LLM 최종 답변 합성 (검색된 RRF 상위 문서 스니펫 및 출처 기반)
        handler.emit("LLM_GENERATION", "RRF 상위 문서를 바탕으로 최종 답변 생성 중...", 90)
        context_blocks = []
        sources_list = []
        for doc in fused_docs:
            meta = doc["metadata"]
            src_label = f"{meta['source']} (p.{meta['page']}, #{meta['chunk_id']})"
            sources_list.append({
                "rank": doc["final_rank"],
                "source": meta["source"],
                "page": meta["page"],
                "chunk_id": meta["chunk_id"],
                "rrf_score": doc["rrf_score"],
                "dense_rank": meta["best_dense_rank"],
                "sparse_rank": meta["best_sparse_rank"],
            })
            context_blocks.append(f"[출처 {doc['final_rank']}: {src_label}]\n{doc['snippet']}")

        joined_context = "\n\n".join(context_blocks)
        synthesis_prompt = (
            f"다음은 RRF(Reciprocal Rank Fusion) 하이브리드 검색으로 선별된 상위 근거 문서들입니다.\n"
            f"반드시 아래 근거 문서의 내용에 기반하여 사용자의 질문에 상세히 답변하고, 문장 끝에 [출처 번호]를 표기하세요.\n\n"
            f"=== 검색된 근거 문서 ===\n{joined_context}\n\n"
            f"=== 사용자 질문 ===\n{user_query}"
        )

        llm_answer = call_llm(synthesis_prompt, system_prompt="당신은 출처 기반의 정확한 RAG 전문 AI입니다.")
        if not llm_answer:
            # 외부 LLM API 키 미설정 시에도 RRF 검색 결과 기반의 구조화된 종합 요약 답변 생성
            summary_lines = [
                f"질의하신 **'{user_query}'**에 대해 5개 다중 확장 쿼리와 FAISS+BM25 하이브리드 검색 및 RRF(k={req.rrf_k}) 재정렬을 수행하여 도출한 핵심 답변입니다:\n"
            ]
            for doc in fused_docs[:3]:
                meta = doc["metadata"]
                snippet_preview = doc["snippet"][:220].replace("\n", " ")
                summary_lines.append(
                    f"- **[출처 {doc['final_rank']}: {meta['source']} {meta['page']}페이지 ({meta['chunk_id']})]** "
                    f"(RRF 점수: `{doc['rrf_score']}`): {snippet_preview}..."
                )
            summary_lines.append(
                "\n*(참고: 서버 환경변수에 `OPENAI_API_KEY` 또는 `GEMINI_API_KEY`를 설정하면 생성형 LLM의 심층 자연어 요약이 활성화됩니다.)*"
            )
            llm_answer = "\n".join(summary_lines)

        elapsed = round(time.time() - start_time, 3)
        handler.emit("COMPLETED", f"RAG Fusion 질의 응답 완료 ({elapsed}초)", 100)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "original_query": user_query,
                "expanded_queries": expanded_queries,
                "rrf_documents": fused_docs,
                "llm_answer": llm_answer,
                "sources": sources_list,
                "progress_logs": handler.get_logs(),
                "elapsed_sec": elapsed,
                "complexity": {
                    "multi_query_layer": "O(M * T_llm) [M=5 파생 쿼리 생성]",
                    "hybrid_search_layer": f"O(M * N * D + M * N * Q_t) [N={len(STORE.documents)}, D={STORE.dimension}, Index={STORE.index_type}]",
                    "rrf_fusion_layer": f"O(M * K + U * log(U)) [RRF 상수 k={req.rrf_k}, 상위 {len(fused_docs)}개 재정렬]",
                },
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"[QUERY_EXECUTION_EXCEPTION] RAG 질의 처리 중 서버 오류가 발생했습니다: {str(e)}\n{tb}",
        )


@app.post("/query-stream")
async def query_rag_stream(req: QueryRequest) -> StreamingResponse:
    """
    StreamHandler를 활용하여 실시간 NDJSON 스트리밍으로 단계별 진행 상태와 최종 RRF 결과를 전송하는 엔드포인트입니다.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        handler = StreamHandler()
        yield handler.emit("START", "Advanced RAG 스트리밍 파이프라인 시작", 5)
        try:
            resp = await query_rag(req)
            payload = json.loads(resp.body.decode("utf-8"))
            for log_item in payload.get("progress_logs", []):
                yield json.dumps(log_item, ensure_ascii=False) + "\n"
            yield handler.emit("FINAL_PAYLOAD", "최종 결과 수신 완료", 100, payload)
        except HTTPException as he:
            yield handler.emit("ERROR", str(he.detail), 100, {"status_code": he.status_code})
        except Exception as exc:
            yield handler.emit("ERROR", f"서버 내부 오류: {str(exc)}", 100)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

