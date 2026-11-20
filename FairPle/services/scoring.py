"""
페어플 (FairPle) — 정량 산출물 환산 및 유효성 검증 엔진 (server/services/scoring.py)
참조: prd_02_core_features.md [F-01]
"""

import re
from typing import Dict, Optional

# 표준 정량 단위 카탈로그 (유형별 단위 라벨 및 단위당 기본 점수 u_j)
STANDARD_TASK_CATALOG: Dict[str, Dict[str, object]] = {
    "research": {"label": "사례/자료 조사", "unit_label": "건", "unit_points": 5},
    "survey": {"label": "설문 제작 및 응답 취합", "unit_label": "식", "unit_points": 15},
    "report_doc": {"label": "보고서 본문 작성", "unit_label": "페이지", "unit_points": 8},
    "ppt": {"label": "PPT 슬라이드 제작", "unit_label": "장", "unit_points": 4},
    "compile": {"label": "전체 자료 취합 및 최종 편집", "unit_label": "식", "unit_points": 20},
    "script_only": {"label": "발표 대본 작성", "unit_label": "분", "unit_points": 3},
    "speech": {"label": "현장/녹화 발표 수행", "unit_label": "분", "unit_points": 6},
    "script": {"label": "대본 작성 및 발표 수행", "unit_label": "분", "unit_points": 9},
    "analysis": {"label": "데이터 분석 및 차트 제작", "unit_label": "건", "unit_points": 6},
}

# 모호한 단일 역할명 블랙리스트 (정량 단위/수치 없이 단독 입력 시 차단)
VAGUE_ROLE_TITLES = {
    "자료조사",
    "조사",
    "리서치",
    "ppt",
    "피피티",
    "발표",
    "대본",
    "보고서",
    "정리",
    "취합",
    "기타",
    "보조",
    "서포트",
    "팀장",
}


def validate_quantitative_task(
    title: str,
    category: str,
    unit_label: str,
    quantity: int,
    custom_base_points: Optional[int] = None,
) -> None:
    """모호한 단일 역할명이나 수량/단위가 누락된 과업 등록을 차단"""
    clean_title = (title or "").strip()
    if len(clean_title) < 2:
        raise ValueError("과업명을 2자 이상 구체적으로 입력해 주세요.")

    normalized_title = re.sub(r"\s+", "", clean_title).lower()
    if normalized_title in VAGUE_ROLE_TITLES:
        raise ValueError(
            f"'{clean_title}' 같은 모호한 단일 역할명은 사용할 수 없습니다. "
            "구체적 정량 산출물(예: '사례 3건 조사', '슬라이드 8장 제작')로 입력해 주세요."
        )

    if quantity is None or int(quantity) <= 0:
        raise ValueError("정량 수량(quantity)은 1 이상이어야 합니다.")

    clean_unit = (unit_label or "").strip()
    cat_key = (category or "").strip().lower()
    if cat_key not in STANDARD_TASK_CATALOG and not clean_unit:
        raise ValueError("커스텀 업무는 정량 단위(예: 건, 장, 페이지, 명)를 반드시 입력해야 합니다.")

    if cat_key == "custom" or cat_key not in STANDARD_TASK_CATALOG:
        if custom_base_points is not None and not (5 <= int(custom_base_points) <= 50):
            raise ValueError("커스텀 업무의 기본 점수는 5~50pt 범위여야 합니다.")


def calculate_base_points(
    category: str,
    quantity: int,
    custom_base_points: Optional[int] = None,
    unit_label: Optional[str] = None,
    title: Optional[str] = None,
) -> int:
    """
    정량 산출물 기준표에 따른 기본 가중치(w_j = u_j * q_j) 자동 산출 순수 함수
    """
    cat_key = (category or "custom").strip().lower()
    qty = int(quantity) if quantity is not None else 0
    if qty <= 0:
        raise ValueError("수량(quantity)은 1 이상이어야 합니다.")

    resolved_unit = unit_label
    if not resolved_unit and cat_key in STANDARD_TASK_CATALOG:
        resolved_unit = str(STANDARD_TASK_CATALOG[cat_key]["unit_label"])

    if title is not None:
        validate_quantitative_task(
            title=title,
            category=cat_key,
            unit_label=resolved_unit or "",
            quantity=qty,
            custom_base_points=custom_base_points,
        )

    if cat_key in STANDARD_TASK_CATALOG:
        unit_pts = int(STANDARD_TASK_CATALOG[cat_key]["unit_points"])
        return unit_pts * qty

    if custom_base_points is not None:
        pts = int(custom_base_points)
        if not (5 <= pts <= 50):
            raise ValueError("커스텀 업무의 기본 점수는 5~50pt 범위여야 합니다.")
        return pts

    return 5 * qty
