"""
서울시 도시계획 LangGraph Agentic RAG 서버 단위 및 파이프라인 통합 테스트 (test_main.py)
"""

from __future__ import annotations

import base64

from main import (
    PdfBase64UploadRequest,
    QueryRequest,
    get_rag_status,
    health_check,
    query_agentic_rag,
    reset_vector_db,
    upload_pdf_base64,
)
from rag_agent import GradeDocuments, RouteQuery, rag_service


def test_health_and_initial_empty_state():
    """초기 상태는 인덱싱된 문서가 0개인 빈 상태여야 합니다."""
    reset_vector_db()
    health = health_check()
    assert health["status"] == "ok"
    assert health["service"] == "seoul-agentic-rag"
    assert health["total_chunks"] == 0

    status = get_rag_status()
    assert status["total_chunks"] == 0
    assert status["sources"] == []
    assert "route_question" in status["graph_nodes"]
    assert "retrieve" in status["graph_nodes"]
    assert "grade_documents" in status["graph_nodes"]
    assert "generate" in status["graph_nodes"]
    assert "casual_talk" in status["graph_nodes"]


def test_pydantic_models_route_and_grade():
    """RouteQuery 및 GradeDocuments 구조화 모델 스키마 검증"""
    rq_vec = RouteQuery(datasource="vectorstore")
    rq_casual = RouteQuery(datasource="casual_talk")
    assert rq_vec.datasource == "vectorstore"
    assert rq_casual.datasource == "casual_talk"

    gd_yes = GradeDocuments(binary_score="yes")
    gd_no = GradeDocuments(binary_score="no")
    assert gd_yes.binary_score == "yes"
    assert gd_no.binary_score == "no"


def test_pdf_upload_splitting_and_rag_graph_execution():
    """
    1) PDF 문서 업로드 -> 텍스트 추출 -> 청크 분할 -> ChromaDB 인덱싱 검증
    2) RAG 질문(vectorstore) 라우팅 -> retrieve -> grade_documents -> generate 실행 검증
    3) 일반 질문(casual_talk) 라우팅 -> casual_talk 노드 실행 검증
    """
    reset_vector_db()

    sample_seoul_doc = (
        "2040 서울도시기본계획 및 서울시 도시계획 조례 제55조 개정안 안내.\n\n"
        "제1장 용도지역별 용적률 기준: 제2종 일반주거지역의 기본 용적률은 200% 이하로 한다. "
        "다만 역세권 청년주택 또는 공공임대주택 도입 시 지구단위계획 수립을 통해 준주거지역으로 상향할 수 있으며 "
        "이 경우 최대 허용 용적률은 500%까지 완화된다.\n\n"
        "제2장 공공기여(기부채납) 인센티브 기준: 용도지역 상향에 따른 용적률 증가분의 50%를 공공임대주택 및 "
        "지역 필요 기반시설로 기부채납해야 한다.\n\n"
        "제3장 역세권 활성화 사업: 역세권 반경 350m 이내 고밀복합개발 구역은 기존 35층 높이 규제를 폐지하고 "
        "창의적인 스카이라인 다변화를 유도한다."
    )
    b64_content = base64.b64encode(sample_seoul_doc.encode("utf-8")).decode("ascii")

    upload_res = upload_pdf_base64(
        PdfBase64UploadRequest(
            filename="2040_서울도시기본계획_조례55조.pdf",
            content_base64=b64_content,
            chunk_size=180,
            chunk_overlap=30,
        )
    )
    assert upload_res["status"] == "ok"
    assert upload_res["added_chunks"] >= 2
    assert upload_res["total_chunks"] >= 2

    # 1. 서울시 도시계획 RAG 질문 실행 (vectorstore 경로)
    rag_res = query_agentic_rag(
        QueryRequest(question="서울시 제2종 일반주거지역에서 역세권 청년주택 도입 시 최대 허용 용적률과 기부채납 기준은?")
    )
    assert rag_res["status"] == "ok"
    assert rag_res["route"] == "vectorstore"
    assert "retrieve" in rag_res["visited_nodes"]
    assert "grade_documents" in rag_res["visited_nodes"]
    assert "generate" in rag_res["visited_nodes"]
    assert rag_res["total_retrieved"] >= 1
    assert rag_res["filtered_count"] >= 1
    assert len(rag_res["answer"]) > 10

    # 2. 일반 질문 실행 (casual_talk 경로)
    casual_res = query_agentic_rag(QueryRequest(question="안녕하세요 반가워요! 오늘 기분이 어떠신가요?"))
    assert casual_res["status"] == "ok"
    assert casual_res["route"] == "casual_talk"
    assert "casual_talk" in casual_res["visited_nodes"]
    assert "retrieve" not in casual_res["visited_nodes"]
    assert len(casual_res["answer"]) > 10

    # 테스트 후 초기 빈 상태로 복원
    reset_vector_db()
