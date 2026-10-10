
"""
Top-1 Candidate Reranker

Scores surviving candidates using multiple features to select the best top-1,
replacing the naive candidates[0] strategy.

Features:
  1. Template OP-F1: operator overlap with the highest-scoring matched template
  2. Operator coverage: fraction of dynamically retrieved operators present
  3. Structural completeness: bracket balance, semicolons, consume ending
  4. Spatial operator bonus: presence of 'at' for query types that typically need it
  5. Complexity penalty: extremely long or extremely short candidates are penalized
"""

import re
from typing import List, Dict, Optional
from collections import Counter


_BRACKET_PAIRS = [('(', ')'), ('[', ']')]

_AT_BONUS_QUERY_TYPES = {
    'moving objects range query',
    'moving objects join query',
    'trajectory similarity query',
}


def _bracket_balance_score(candidate: str) -> float:
    """1.0 if all bracket types are balanced, decays with imbalance."""
    total_imbalance = 0
    for open_b, close_b in _BRACKET_PAIRS:
        total_imbalance += abs(candidate.count(open_b) - candidate.count(close_b))
    if total_imbalance == 0:
        return 1.0
    return max(0.0, 1.0 - total_imbalance * 0.15)


def _structural_score(candidate: str) -> float:
    """Score structural completeness of a SECONDO query."""
    score = 0.0
    c = candidate.strip()

    if c.endswith('consume;') or c.endswith('consume'):
        score += 0.3

    if c.startswith('query ') or c.startswith('let '):
        score += 0.2

    
    if 'let ' in c and 'query ' in c:
        let_pos = c.find('let ')
        query_pos = c.find('query ', let_pos + 1)
        if query_pos > let_pos:
            between = c[let_pos:query_pos]
            if ';' in between:
                score += 0.2
            else:
                score -= 0.3

    score += _bracket_balance_score(c) * 0.3
    return max(0.0, min(1.0, score))


def _operator_coverage_score(candidate: str, retrieved_operators: List[Dict]) -> float:
    """Fraction of retrieved operator names that appear in the candidate."""
    if not retrieved_operators:
        return 0.5
    c_lower = candidate.lower()
    hits = sum(1 for op in retrieved_operators
               if op.get('name', '').lower() in c_lower)
    return hits / len(retrieved_operators) if retrieved_operators else 0.0


def _template_similarity_score(candidate: str, templates: List[str]) -> float:
    """Best OP-F1 between candidate and any of the top templates."""
    if not templates:
        return 0.0

    from nalplan.eval_metrics import extract_operators_multiset
    cand_ops = extract_operators_multiset(candidate)
    if not cand_ops:
        return 0.0

    best = 0.0
    for tmpl in templates[:10]:
        tmpl_ops = extract_operators_multiset(tmpl)
        if not tmpl_ops:
            continue
        all_ops = set(cand_ops.keys()) | set(tmpl_ops.keys())
        tp = sum(min(cand_ops[op], tmpl_ops[op]) for op in all_ops)
        p = tp / sum(cand_ops.values())
        r = tp / sum(tmpl_ops.values())
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            best = max(best, f1)
    return best


def _spatial_at_bonus(candidate: str, query_type: str) -> float:
    """Bonus for using .Trip at / .UTrip at in query types that need it."""
    if query_type.lower() not in _AT_BONUS_QUERY_TYPES:
        return 0.0
    c_lower = candidate.lower()
    if '.trip at ' in c_lower or '.utrip at ' in c_lower:
        return 1.0
    return 0.0


def _length_penalty(candidate: str) -> float:
    """Penalize extremely short or long candidates."""
    length = len(candidate)
    if length < 30:
        return 0.3
    if length > 1500:
        return 0.7
    return 1.0


def score_candidate(candidate: str,
                    query_type: str = '',
                    templates: Optional[List[str]] = None,
                    retrieved_operators: Optional[List[Dict]] = None) -> float:
    """
    Compute a composite score for a single candidate.

    Weights (tuned for SECONDO query selection):
      template_sim:     0.30  (how well operators match known templates)
      structural:       0.20  (syntax correctness signals)
      operator_cov:     0.15  (uses retrieved operators)
      at_bonus:         0.25  (critical for range/join/similarity)
      length_penalty:   0.10  (avoid degenerate candidates)
    """
    w_template = 0.30
    w_structure = 0.20
    w_op_cov = 0.15
    w_at = 0.25
    w_length = 0.10

    s_template = _template_similarity_score(candidate, templates or [])
    s_structure = _structural_score(candidate)
    s_op_cov = _operator_coverage_score(candidate, retrieved_operators or [])
    s_at = _spatial_at_bonus(candidate, query_type)
    s_length = _length_penalty(candidate)

    return (w_template * s_template +
            w_structure * s_structure +
            w_op_cov * s_op_cov +
            w_at * s_at +
            w_length * s_length)


def rerank_candidates(candidates: List[str],
                      query_type: str = '',
                      templates: Optional[List[str]] = None,
                      retrieved_operators: Optional[List[Dict]] = None) -> List[str]:
    """
    Rerank a list of candidates by composite score (descending).
    Returns a new list with the best candidate first.
    """
    if not candidates:
        return candidates
    if len(candidates) == 1:
        return candidates

    scored = [(c, score_candidate(c, query_type, templates, retrieved_operators))
              for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def select_top1(candidates: List[str],
                query_type: str = '',
                templates: Optional[List[str]] = None,
                retrieved_operators: Optional[List[Dict]] = None) -> str:
    """Select the best candidate as top-1."""
    if not candidates:
        return ''
    reranked = rerank_candidates(candidates, query_type, templates, retrieved_operators)
    return reranked[0]
