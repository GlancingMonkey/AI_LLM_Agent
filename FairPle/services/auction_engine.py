"""
페어플 (FairPle) — 24시간 비동기 밀봉 역경매 최적 배정 엔진 (server/services/auction_engine.py)
참조: prd_02_core_features.md [F-02]
- 입찰 허용 범위: 0.8 * w_j <= b_{i,j} <= 2.0 * w_j
- 목적 함수: min sum b_{i,j} * x_{i,j}
- 균형 제약: 팀원별 낙찰 총점 S_i가 평균의 ±20% 범위 내 (M >= N일 때 전원 최소 1개 배정, 불가능 시 분산 최소화 차선 해)
- 동점 처리: 결정론적 시드(SHA-256) 기반 추첨
- 미입찰자 처리: 입찰가를 2.0 * w_j로 간주해 성실 입찰자 우선 배정 후, 미입찰자 배정 과업은 기본 가중치(1.0 * w_j)로 귀속
"""

import hashlib
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple


def get_bid_bounds(base_points: int) -> Tuple[int, int]:
    """과업 기본 가중치 w_j에 대한 허용 입찰 범위 [0.8*w_j, 2.0*w_j] 반환"""
    w = int(base_points)
    min_bid = max(1, int(math.floor(0.8 * w + 0.5)))
    max_bid = max(min_bid, int(math.ceil(2.0 * w)))
    return min_bid, max_bid


def validate_bid_points(base_points: int, bid_points: int) -> None:
    min_b, max_b = get_bid_bounds(base_points)
    b = int(bid_points)
    if b < min_b or b > max_b:
        raise ValueError(
            f"입찰 점수({b}pt)가 허용 범위({min_b}pt ~ {max_b}pt, 0.8w~2.0w)를 벗어났습니다."
        )


def _deterministic_tie_hash(seed_key: str, assignment: Tuple[int, ...]) -> str:
    raw = f"{seed_key}:{','.join(map(str, assignment))}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def settle_reverse_auction(
    tasks: List[Dict[str, Any]],
    members: List[Dict[str, Any]],
    bids: List[Dict[str, Any]],
    seed_key: str = "fairple-default-seed",
) -> Dict[str, Any]:
    """
    밀봉 역경매 최적 배정 수행
    Args:
        tasks: [{"id": str, "title": str, "base_points": int, ...}]
        members: [{"user_id": str, "display_name": str, "has_submitted_bid": bool}]
        bids: [{"task_id": str, "user_id": str, "bid_points": int}]
        seed_key: 동점 시 결정론적 추첨용 시드(방 ID 또는 초대코드)
    """
    if not tasks:
        raise ValueError("배정할 과업이 없습니다.")
    if not members:
        raise ValueError("참여 팀원이 없습니다.")

    n_members = len(members)
    m_tasks = len(tasks)
    member_ids = [str(m["user_id"]) for m in members]
    task_ids = [str(t["id"]) for t in tasks]
    base_weights = [int(t["base_points"]) for t in tasks]

    # 입찰 맵 구성: (user_id, task_id) -> bid_points
    bid_map: Dict[Tuple[str, str], int] = {}
    bidders_set = set()
    for b in bids:
        uid = str(b["user_id"])
        tid = str(b["task_id"])
        bid_map[(uid, tid)] = int(b["bid_points"])
        bidders_set.add(uid)

    non_bidders: List[str] = []
    for m in members:
        uid = str(m["user_id"])
        has_sub = bool(m.get("has_submitted_bid", False)) or (uid in bidders_set)
        if not has_sub:
            non_bidders.append(uid)

    # 유효 입찰 행렬 cost_matrix[i][j] (미입찰자는 2.0 * w_j 상한으로 간주)
    cost_matrix: List[List[int]] = []
    for i, uid in enumerate(member_ids):
        row: List[int] = []
        is_non_bidder = uid in non_bidders
        for j, tid in enumerate(task_ids):
            w_j = base_weights[j]
            _, max_b = get_bid_bounds(w_j)
            if is_non_bidder or (uid, tid) not in bid_map:
                row.append(max_b)
            else:
                row.append(bid_map[(uid, tid)])
        cost_matrix.append(row)

    require_min_one = m_tasks >= n_members

    best_balanced: Optional[Tuple[int, float, str, Tuple[int, ...]]] = None
    best_fallback: Optional[Tuple[float, int, str, Tuple[int, ...]]] = None

    # 팀플 규모(N=2~6, M=2~10)에 대해 완전 탐색 (M > 8일 경우 그리디+로컬서치 폴백)
    if (n_members ** m_tasks) <= 250000:
        candidate_iter = itertools.product(range(n_members), repeat=m_tasks)
    else:
        # 대규모 과업일 경우 그리디 후보군 생성
        greedy_assign = []
        loads = [0] * n_members
        for j in range(m_tasks):
            best_i = min(range(n_members), key=lambda i: (loads[i], cost_matrix[i][j]))
            greedy_assign.append(best_i)
            loads[best_i] += cost_matrix[best_i][j]
        candidate_iter = [tuple(greedy_assign)]

    for assign in candidate_iter:
        if require_min_one:
            counts = [0] * n_members
            for member_idx in assign:
                counts[member_idx] += 1
            if any(c == 0 for c in counts):
                continue

        # 실제 정산 점수 계산 (미입찰자에게 배정된 과업은 1.0 * w_j로 귀속)
        member_awarded_totals = [0] * n_members
        objective_cost = 0
        for j, member_idx in enumerate(assign):
            uid = member_ids[member_idx]
            objective_cost += cost_matrix[member_idx][j]
            if uid in non_bidders or (uid, task_ids[j]) not in bid_map:
                awarded = base_weights[j]
            else:
                awarded = cost_matrix[member_idx][j]
            member_awarded_totals[member_idx] += awarded

        mean_s = sum(member_awarded_totals) / n_members if n_members > 0 else 0.0
        variance = (
            sum((s - mean_s) ** 2 for s in member_awarded_totals) / n_members
            if n_members > 0
            else 0.0
        )

        # ±20% 균형 조건 확인
        within_20_pct = True
        if mean_s > 0:
            for s in member_awarded_totals:
                if abs(s - mean_s) / mean_s > 0.2001:
                    within_20_pct = False
                    break

        tie_hash = _deterministic_tie_hash(seed_key, assign)

        if within_20_pct:
            cand_key = (objective_cost, round(variance, 4), tie_hash, assign)
            if best_balanced is None or cand_key < best_balanced:
                best_balanced = cand_key

        fb_key = (round(variance, 4), objective_cost, tie_hash, assign)
        if best_fallback is None or fb_key < best_fallback:
            best_fallback = fb_key

    chosen_assign: Tuple[int, ...]
    is_balanced_20 = False
    if best_balanced is not None:
        chosen_assign = best_balanced[3]
        is_balanced_20 = True
    elif best_fallback is not None:
        chosen_assign = best_fallback[3]
    else:
        chosen_assign = tuple(j % n_members for j in range(m_tasks))

    # 결과 조립
    assignments: List[Dict[str, Any]] = []
    member_totals: Dict[str, int] = {uid: 0 for uid in member_ids}
    member_task_counts: Dict[str, int] = {uid: 0 for uid in member_ids}

    for j, member_idx in enumerate(chosen_assign):
        uid = member_ids[member_idx]
        tid = task_ids[j]
        w_j = base_weights[j]
        is_uncontested = (uid in non_bidders) or ((uid, tid) not in bid_map)
        awarded_pts = w_j if is_uncontested else bid_map[(uid, tid)]
        bonus_pts = awarded_pts - w_j

        member_totals[uid] += awarded_pts
        member_task_counts[uid] += 1

        assignments.append(
            {
                "task_id": tid,
                "title": tasks[j].get("title", ""),
                "assignee_id": uid,
                "assignee_name": members[member_idx].get("display_name", uid),
                "base_points": w_j,
                "awarded_points": awarded_pts,
                "bonus_points": bonus_pts,
                "is_non_bidder_fallback": is_uncontested,
            }
        )

    total_awarded = sum(member_totals.values())
    avg_points = round(total_awarded / n_members, 1) if n_members > 0 else 0.0
    std_dev = (
        round(
            math.sqrt(
                sum((pts - avg_points) ** 2 for pts in member_totals.values()) / n_members
            ),
            2,
        )
        if n_members > 0
        else 0.0
    )

    member_summaries = []
    for m in members:
        uid = str(m["user_id"])
        pts = member_totals[uid]
        ratio = round((pts / total_awarded) * 100, 1) if total_awarded > 0 else 0.0
        member_summaries.append(
            {
                "user_id": uid,
                "display_name": m.get("display_name", uid),
                "awarded_points": pts,
                "task_count": member_task_counts[uid],
                "share_percent": ratio,
                "participated_in_bid": uid not in non_bidders,
            }
        )

    return {
        "assignments": assignments,
        "member_summaries": member_summaries,
        "non_bidders": non_bidders,
        "total_awarded_points": total_awarded,
        "average_points": avg_points,
        "std_dev": std_dev,
        "is_within_20_percent_balance": is_balanced_20,
    }
