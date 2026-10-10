
"""
          

       
  -     (Accuracy):      EM@1, EM@k, OP-F1@1, RM@1
  -       (Exec Corr):  EX@1, EX@k, VER@1, NER@1
  -    (Efficiency):       Latency, #LLM_Calls, Speedup, CacheHit, CR%, CCR, Prune-F1, FPR

  :
  - pandas, openpyxl (benchmark xlsx   )
  - secondo_connector (RM@1     )
"""

import re
import json
import hashlib
import logging
import subprocess
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

from nalplan.config import SECONDO_BIN
SECONDO_BIN = str(SECONDO_BIN)
QUERY_TIMEOUT = 10

QUERY_TYPE_MAPPING = {
    'Time Interval Query': 'Time interval query',
    'Time Point Query': 'Time interval query',
    'Range Query': 'Moving objects range query',
    'Range Query ': 'Moving objects range query',
    'Moving objects range query': 'Moving objects range query',
    'Nearest Neighbor Query': 'Moving objects nearest neighbor query',
    'Moving objects nearest neighbor query': 'Moving objects nearest neighbor query',
    'Join Query': 'Moving objects join query',
    'Join Query ': 'Moving objects join query',
    'Moving objects join query': 'Moving objects join query',
    'Similarity Query': 'Trajectory similarity query',
    'Trajectory similarity query': 'Trajectory similarity query',
    'Time interval query': 'Time interval query',
}

CANONICAL_TYPES = [
    'Time interval query',
    'Moving objects range query',
    'Moving objects nearest neighbor query',
    'Moving objects join query',
    'Trajectory similarity query',
]

_ALL_KNOWN_OPS = [
    'feed', 'filter', 'extend', 'project', 'consume', 'count',
    'head', 'sortby', 'groupby', 'extract', 'rename', 'let', 'query',
    'loopsel', 'loopjoin', 'symmjoin', 'hashjoin', 'spatialjoin',
    'tconsume', 'tcount', 'transformstream',
    'atperiods', 'atinstant', 'deftime', 'present', 'trajectory',
    'val', 'initial', 'final', 'inst', 'periods', 'samplempoint',
    'passes', 'inside', 'at', 'intersects', 'overlap', 'meet',
    'distance', 'line2region', 'box2d', 'bbox', 'atposition',
    'knearest', 'greeceknearest', 'chinaknearest', 'distancescan',
    'distancescan2', 'distancescan4', 'knearestfilter',
    'oldknearestfilter',
    'creatertree', 'bulkloadrtree', 'windowintersects',
    'createbtree', 'createbtree2', 'createhash',
    'createtbtree', 'creatextree', 'createmtree',
    'exactmatch', 'leftrange', 'rightrange',
    'rangesearch', 'nnsearch',
    'frechetdistance', 'lcss', 'rel_lcss', 'dist_euclidean',
    'dist_origin_and_destination', 'dist_hausdorff', 'dist_sdiagram',
    'dist_perpendicular', 'dist_angular', 'dist_lip',
    'to_pointseq', 'to_tpointseq',
    'everNearerThan', 'everNearerThan2',
    'the_mvalue', 'no_components', 'length', 'area', 'perimeter',
]


_OP_PATTERNS = []
for _op in sorted(_ALL_KNOWN_OPS, key=len, reverse=True):
    _OP_PATTERNS.append((re.compile(r'(?<!\.)(?<!\w)' + re.escape(_op) + r'(?!\w)', re.IGNORECASE), _op))






class ErrorType:
    SYNTAX = 'SYNTAX'
    ENTITY = 'ENTITY'
    OP_ARG = 'OP_ARG'
    EMPTY = 'EMPTY'
    TIMEOUT = 'TIMEOUT'
    SEMANTIC = 'SEMANTIC'
    NONE = 'NONE'






def normalize_query(q: str) -> str:
    """Normalize a SECONDO query for exact-match comparison."""
    if not q:
        return ''
    q = q.strip()
    q = q.rstrip(';').strip()
    q = re.sub(r'\s+', ' ', q)
    q = q.lower()
    return q






def compute_em(pred: str, gold: str) -> bool:
    return normalize_query(pred) == normalize_query(gold)


def compute_em_at_k(candidates: List[str], gold: str) -> bool:
    gold_norm = normalize_query(gold)
    return any(normalize_query(c) == gold_norm for c in candidates)






def extract_operators_multiset(query: str) -> Counter:
    """Extract all operator occurrences as a multiset (Counter)."""
    if not query:
        return Counter()
    q_lower = query.lower()
    counts = Counter()
    for pattern, op_name in _OP_PATTERNS:
        n = len(pattern.findall(q_lower))
        if n > 0:
            counts[op_name] += n
    return counts


def compute_op_f1(pred: str, gold: str) -> float:
    """Compute multiset-level operator F1 between pred and gold."""
    pred_ops = extract_operators_multiset(pred)
    gold_ops = extract_operators_multiset(gold)
    if not pred_ops and not gold_ops:
        return 1.0
    if not pred_ops or not gold_ops:
        return 0.0

    all_ops = set(pred_ops.keys()) | set(gold_ops.keys())
    tp = sum(min(pred_ops[op], gold_ops[op]) for op in all_ops)
    pred_total = sum(pred_ops.values())
    gold_total = sum(gold_ops.values())

    precision = tp / pred_total if pred_total > 0 else 0.0
    recall = tp / gold_total if gold_total > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)






@dataclass
class ExecutionResult:
    executable: bool = False
    has_result: bool = False
    result_hash: str = ''
    result_hash_relaxed: str = ''
    result_data: str = ''
    raw_output: str = ''
    error_msg: str = ''
    timed_out: bool = False
    exec_time_ms: float = 0.0


def execute_on_secondo(query: str, database: str = 'berlintest',
                       timeout: int = QUERY_TIMEOUT) -> ExecutionResult:
    """Execute a single query on SECONDO and return structured result."""
    result = ExecutionResult()
    clean_q = query.strip().rstrip(';')

    has_let = bool(re.search(r'^\s*let\s+', clean_q, re.IGNORECASE | re.MULTILINE))
    if has_let:
        parts = re.split(r';\s*(?=query\s)', clean_q, flags=re.IGNORECASE)
        cmd_lines = ';\n'.join(p.strip() for p in parts if p.strip())
        full_command = f"open database {database};\n{cmd_lines};\nquit;\n"
    else:
        full_command = f"open database {database};\n{clean_q};\nquit;\n"

    import time
    t0 = time.time()
    try:
        proc = subprocess.run(
            [SECONDO_BIN], input=full_command,
            capture_output=True, text=True, timeout=timeout
        )
        result.exec_time_ms = (time.time() - t0) * 1000
        stdout = proc.stdout or ''
        stderr = proc.stderr or ''
        result.raw_output = stdout

        error_phrases = [
            'error', 'Error', 'undefined', 'not defined',
            'does not exist', 'type mismatch', 'syntax error',
            'Cannot', 'invalid', 'expected', 'Operator .* expects'
        ]
        has_error = any(re.search(p, stdout) for p in error_phrases)

        if has_error:
            result.executable = False
            result.error_msg = _extract_error_snippet(stdout)
        else:
            result.executable = True
            data = _extract_result_data(stdout)
            result.result_data = data
            result.has_result = len(data.strip()) > 0 if data else False
            if result.has_result:
                result.result_hash = hashlib.md5(data.strip().encode()).hexdigest()
                normalized = _normalize_result_for_comparison(data)
                result.result_hash_relaxed = hashlib.md5(
                    normalized.encode()).hexdigest() if normalized else ''

    except subprocess.TimeoutExpired:
        result.exec_time_ms = timeout * 1000
        result.timed_out = True
        result.executable = True
        result.has_result = False
    except Exception as e:
        result.error_msg = str(e)

    return result


def _extract_error_snippet(stdout: str) -> str:
    for line in stdout.split('\n'):
        lower = line.lower()
        if any(w in lower for w in ['error', 'undefined', 'does not exist',
                                      'type mismatch', 'cannot', 'invalid']):
            return line.strip()[:200]
    return ''


def _extract_result_data(stdout: str) -> str:
    """Extract pure result data from SECONDO stdout, filtering ALL system info."""
    lines = stdout.split('\n')
    data_lines = []
    in_result = False
    _skip_prefixes = (
        'Times (elapsed', 'Total runtime', 'command', 'started at:',
        'noMemoryOperators', 'perOperator', '=> []', '=> [',
        'use zone', 'Detected invalid order',
        'case ', 'feddisch', 'remaining:',
    )
    _skip_contains = (
        '---------|', 'remaining:', 'feddisch (done)',
        'Thank you for using SECONDO',
    )
    for line in lines:
        if 'Secondo =>' in line or '--->' in line:
            in_result = True
            continue
        if in_result:
            stripped = line.strip()
            if not stripped or line.startswith('Secondo'):
                continue
            if stripped.startswith(_skip_prefixes):
                continue
            if any(s in stripped for s in _skip_contains):
                continue
            
            if stripped.startswith('<'):
                continue
            
            if stripped.startswith("'") and stripped.endswith("'") and len(stripped) < 500:
                continue
            
            if re.match(r'^\.+\s*$', stripped):
                continue
            data_lines.append(line)
    return '\n'.join(data_lines)


def _normalize_result_for_comparison(data: str) -> str:
    """Normalize extracted result data for relaxed execution equivalence comparison.
    
    Removes schema/type declarations, normalizes numbers/timestamps/whitespace,
    and sorts rows for order-independent comparison.
    """
    if not data or not data.strip():
        return ''

    text = data

    
    text = re.sub(
        r'\(\s*rel\s+\(\s*tuple\s+\((?:[^()]*|\([^()]*\))*\)\s*\)\s*\)',
        '', text, flags=re.DOTALL
    )

    
    text = re.sub(r'Generic display function used!', '', text)
    text = re.sub(r'^\s*Type\s*:\s*\w+\s*$', '', text, flags=re.MULTILINE)

    
    def _round_float(m):
        try:
            return f"{float(m.group(0)):.1f}"
        except (ValueError, OverflowError):
            return m.group(0)
    text = re.sub(r'\b\d+\.\d{2,}\b', _round_float, text)

    
    text = re.sub(
        r'(\d{4}-\d{2}-\d{2}-)(\d):',
        lambda m: m.group(1) + '0' + m.group(2) + ':', text
    )

    
    text = re.sub(r'\s+', ' ', text).strip()

    
    rows = _extract_balanced_groups(text)
    if len(rows) > 1:
        normalized_rows = sorted(rows)
        return '\n'.join(normalized_rows)

    return text


def _extract_balanced_groups(text: str) -> list:
    """Extract top-level balanced parenthesized groups from text."""
    groups = []
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '(':
            if depth == 0:
                start = i
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(text[start:i + 1])
                start = -1
    
    if len(groups) == 1 and len(groups[0]) > 4:
        inner = groups[0][1:-1].strip()
        children = _extract_balanced_groups(inner)
        if len(children) > 1:
            return children
    return groups






def compute_rm(pred: str, gold: str, database: str = 'berlintest',
               timeout: int = QUERY_TIMEOUT) -> Optional[bool]:
    """Execute both pred and gold, compare result hashes. Returns None if either fails."""
    pred_result = execute_on_secondo(pred, database, timeout)
    gold_result = execute_on_secondo(gold, database, timeout)

    if not pred_result.has_result or not gold_result.has_result:
        return None
    return pred_result.result_hash == gold_result.result_hash






def classify_error(exec_result: ExecutionResult,
                   rm_match: Optional[bool] = None) -> str:
    """Classify an execution failure into error type code."""
    if exec_result.timed_out:
        return ErrorType.TIMEOUT

    if not exec_result.executable:
        err = exec_result.error_msg.lower()
        if any(w in err for w in ['does not exist', 'undefined', 'not defined']):
            return ErrorType.ENTITY
        if any(w in err for w in ['type mismatch', 'expects', 'expected']):
            return ErrorType.OP_ARG
        return ErrorType.SYNTAX

    if not exec_result.has_result:
        return ErrorType.EMPTY

    if rm_match is False:
        return ErrorType.SEMANTIC

    return ErrorType.NONE






def paired_bootstrap_test(scores_a: List[float], scores_b: List[float],
                          n_bootstrap: int = 1000, seed: int = 42) -> float:
    """
    Test whether system A is significantly better than system B.
    Returns p-value: probability that B >= A under bootstrap resampling.
    """
    assert len(scores_a) == len(scores_b), "Score lists must be same length"
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    a = np.array(scores_a)
    b = np.array(scores_b)
    delta_obs = a.mean() - b.mean()

    count_ge = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        delta_boot = a[idx].mean() - b[idx].mean()
        if delta_boot <= 0:
            count_ge += 1
    return count_ge / n_bootstrap






@dataclass
class PruningMetrics:
    prune_count: int = 0
    prune_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0


def compute_pruning_metrics(pruned_labels: List[bool],
                            all_labels: List[bool]) -> PruningMetrics:
    """
    Args:
        pruned_labels: list of ground-truth invalid flags for *pruned* candidates
                       (True=invalid, False=valid)
        all_labels:    list of ground-truth invalid flags for *all* candidates
    """
    m = PruningMetrics()
    m.prune_count = len(pruned_labels)
    total = len(all_labels)
    if total == 0:
        return m
    m.prune_rate = m.prune_count / total

    total_invalid = sum(1 for x in all_labels if x)
    total_valid = total - total_invalid

    tp = sum(1 for x in pruned_labels if x)       
    fp = sum(1 for x in pruned_labels if not x)    

    m.precision = tp / m.prune_count if m.prune_count > 0 else 0.0
    m.recall = tp / total_invalid if total_invalid > 0 else 0.0
    m.f1 = (2 * m.precision * m.recall / (m.precision + m.recall)
            if (m.precision + m.recall) > 0 else 0.0)
    m.fpr = fp / total_valid if total_valid > 0 else 0.0

    return m






def load_benchmark(path: str = 'test_cases/benchmark-2026.3.18-fixed.xlsx') -> List[Dict]:
    """Load benchmark xlsx, return list of {nlq, gold, query_type, dataset}."""
    import pandas as pd
    df = pd.read_excel(path)
    records = []
    for _, row in df.iterrows():
        nlq = str(row.get('review', '')).strip()
        gold = str(row.get('executable language', '')).strip()
        cat = str(row.get('cat', '')).strip()
        query_type = QUERY_TYPE_MAPPING.get(cat, cat)

        dataset = _infer_dataset(gold, nlq)
        records.append({
            'nlq': nlq,
            'gold': gold,
            'query_type': query_type,
            'cat_original': cat,
            'dataset': dataset,
        })
    return records


def _infer_dataset(gold: str, nlq: str) -> str:
    combined = (gold + ' ' + nlq).lower()
    has_train = any(w in combined for w in ['unittrains', 'trains feed', 'train '])
    has_taxi = any(w in combined for w in ['unittaxis', 'taxis feed', 'taxi '])
    if has_train and not has_taxi:
        return 'berlintest'
    if has_taxi and not has_train:
        return 'nanjingtest'
    known_objs = ['buses', 'people', 'pedestrian', 'cars', 'vehicles', 'unitbuses']
    if any(w in combined for w in known_objs):
        return 'unknown'
    if 'berlintest' in combined or 'mehringdamm' in combined or 'tiergarten' in combined:
        return 'berlintest'
    if 'nanjingtest' in combined or 'nanjing' in combined:
        return 'nanjingtest'
    return 'unknown'


def filter_evaluable(records: List[Dict]) -> List[Dict]:
    """Return only berlintest + nanjingtest records (442 evaluable)."""
    return [r for r in records if r['dataset'] in ('berlintest', 'nanjingtest')]


def filter_berlin(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r['dataset'] == 'berlintest']


def filter_nanjing(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r['dataset'] == 'nanjingtest']


def filter_unknown(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r['dataset'] == 'unknown']


def stratified_sample(records: List[Dict], n_per_type: int = 15,
                      seed: int = 42, dataset: str = 'berlintest') -> List[Dict]:
    """Stratified random sample: n_per_type from each of 5 query types."""
    import random
    rng = random.Random(seed)
    subset = [r for r in records if r['dataset'] == dataset]
    by_type = {}
    for r in subset:
        by_type.setdefault(r['query_type'], []).append(r)

    sampled = []
    for qt in CANONICAL_TYPES:
        pool = by_type.get(qt, [])
        k = min(n_per_type, len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def stratified_split(records: List[Dict], train_ratio: float = 0.6,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified random split by query_type + dataset."""
    import random
    rng = random.Random(seed)
    groups = {}
    for r in records:
        key = (r['query_type'], r['dataset'])
        groups.setdefault(key, []).append(r)

    train, test = [], []
    for key, items in groups.items():
        rng.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        train.extend(items[:split_idx])
        test.extend(items[split_idx:])
    return train, test






@dataclass
class MethodResult:
    """Result for one NLQ from one method."""
    nlq: str = ''
    query_type: str = ''
    dataset: str = ''
    gold: str = ''
    candidates: List[str] = field(default_factory=list)
    top1: str = ''
    latency_ms: float = 0.0
    llm_calls: int = 0


@dataclass
class EvalScores:
    """Aggregated scores for a method."""
    n: int = 0
    em_at_1: float = 0.0
    em_at_k: float = 0.0
    op_f1_at_1: float = 0.0
    rm_at_1: float = 0.0
    rm_at_1_n: int = 0
    ex_at_1: float = 0.0
    ex_at_k: float = 0.0
    ver_at_1: float = 0.0
    ner_at_1: float = 0.0
    exec_n: int = 0
    avg_latency_ms: float = 0.0
    avg_llm_calls: float = 0.0
    error_dist: Dict[str, int] = field(default_factory=dict)


def evaluate_method(results: List[MethodResult],
                    compute_execution: bool = True,
                    compute_rm: bool = False) -> EvalScores:
    """
    Evaluate a list of MethodResult against gold.
    execution metrics computed only on berlintest subset.
    """
    scores = EvalScores()
    scores.n = len(results)
    if scores.n == 0:
        return scores

    em1_sum = 0
    emk_sum = 0
    opf1_sum = 0.0
    latency_sum = 0.0
    llm_calls_sum = 0

    berlin_results = [r for r in results if r.dataset == 'berlintest']
    scores.exec_n = len(berlin_results)
    ex1_sum = 0
    exk_sum = 0
    ver1_sum = 0
    ner1_sum = 0
    rm1_sum = 0
    rm1_count = 0
    error_counts = Counter()

    for r in results:
        top1 = r.top1 or (r.candidates[0] if r.candidates else '')
        em1_sum += int(compute_em(top1, r.gold))
        emk_sum += int(compute_em_at_k(r.candidates or [top1], r.gold))
        opf1_sum += compute_op_f1(top1, r.gold)
        latency_sum += r.latency_ms
        llm_calls_sum += r.llm_calls

    scores.em_at_1 = em1_sum / scores.n
    scores.em_at_k = emk_sum / scores.n
    scores.op_f1_at_1 = opf1_sum / scores.n
    scores.avg_latency_ms = latency_sum / scores.n
    scores.avg_llm_calls = llm_calls_sum / scores.n

    if compute_execution and berlin_results:
        for r in berlin_results:
            top1 = r.top1 or (r.candidates[0] if r.candidates else '')
            exec_r = execute_on_secondo(top1, 'berlintest', QUERY_TIMEOUT)

            is_ex = exec_r.executable and not exec_r.timed_out
            is_ner = exec_r.has_result
            is_ver = is_ex and is_ner
            ex1_sum += int(is_ex)
            ner1_sum += int(is_ner)
            ver1_sum += int(is_ver)

            cands = r.candidates or [top1]
            any_ex = False
            for c in cands:
                cr = execute_on_secondo(c, 'berlintest', QUERY_TIMEOUT)
                if cr.executable and not cr.timed_out:
                    any_ex = True
                    break
            exk_sum += int(any_ex)

            if not is_ver:
                err_type = classify_error(exec_r)
                error_counts[err_type] += 1

            if compute_rm and is_ver:
                gold_exec = execute_on_secondo(r.gold, 'berlintest', QUERY_TIMEOUT)
                if gold_exec.has_result:
                    rm_match = exec_r.result_hash == gold_exec.result_hash
                    rm1_sum += int(rm_match)
                    rm1_count += 1
                    if not rm_match:
                        error_counts[ErrorType.SEMANTIC] += 1

        n_b = len(berlin_results)
        scores.ex_at_1 = ex1_sum / n_b
        scores.ex_at_k = exk_sum / n_b
        scores.ver_at_1 = ver1_sum / n_b
        scores.ner_at_1 = ner1_sum / n_b
        scores.rm_at_1 = rm1_sum / rm1_count if rm1_count > 0 else 0.0
        scores.rm_at_1_n = rm1_count
        scores.error_dist = dict(error_counts)

    return scores


def format_scores_table(method_scores: Dict[str, EvalScores]) -> str:
    """Format scores as a printable comparison table."""
    header = (f"{'Method':<20} | {'EM@1':>6} {'OP-F1':>6} {'RM@1':>6} | "
              f"{'EX@1':>6} {'VER@1':>6} {'EX@k':>6} | "
              f"{'Lat(ms)':>8} {'#LLM':>5}")
    sep = '-' * len(header)
    lines = [header, sep]
    for name, s in method_scores.items():
        rm_str = f"{s.rm_at_1:.3f}" if s.rm_at_1_n > 0 else "  N/A"
        line = (f"{name:<20} | {s.em_at_1:>6.3f} {s.op_f1_at_1:>6.3f} {rm_str:>6} | "
                f"{s.ex_at_1:>6.3f} {s.ver_at_1:>6.3f} {s.ex_at_k:>6.3f} | "
                f"{s.avg_latency_ms:>8.0f} {s.avg_llm_calls:>5.1f}")
        lines.append(line)
    return '\n'.join(lines)


def format_type_breakdown(method_results: Dict[str, List[MethodResult]]) -> str:
    """Format per-query-type EM@1/VER@1 breakdown."""
    lines = []
    header_types = ['Time-I', 'Range', 'NN', 'Sim', 'Join']
    header = f"{'Method':<20} | " + ' | '.join(f"{t:>14}" for t in header_types)
    lines.append(header)
    lines.append('-' * len(header))

    for method, results in method_results.items():
        parts = []
        for qt in CANONICAL_TYPES:
            qt_results = [r for r in results if r.query_type == qt]
            if not qt_results:
                parts.append(f"{'---':>14}")
                continue
            em1 = sum(int(compute_em(r.top1 or (r.candidates[0] if r.candidates else ''), r.gold))
                      for r in qt_results) / len(qt_results)
            parts.append(f"{em1:>6.3f}       ")
        lines.append(f"{method:<20} | " + ' | '.join(parts))
    return '\n'.join(lines)






if __name__ == '__main__':
    print("=== eval_metrics.py self-test ===")

    q1 = "query Trains feed filter[.Id = 1] consume"
    q2 = "query  Trains  feed  filter[.Id = 1]  consume ;"
    print(f"EM(q1, q2) = {compute_em(q1, q2)}")

    q3 = "query Trains feed filter[.Id = 1] project[Trip] consume"
    print(f"OP-F1(q1, q3) = {compute_op_f1(q1, q3):.3f}")
    print(f"Operators q1: {dict(extract_operators_multiset(q1))}")
    print(f"Operators q3: {dict(extract_operators_multiset(q3))}")

    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    b = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    p = paired_bootstrap_test(a, b)
    print(f"Bootstrap p-value (a vs b): {p:.3f}")

    print("\nLoading benchmark...")
    try:
        records = load_benchmark()
        evaluable = filter_evaluable(records)
        berlin = filter_berlin(records)
        nanjing = filter_nanjing(records)
        unknown = filter_unknown(records)
        print(f"Total: {len(records)}, Evaluable: {len(evaluable)}, "
              f"Berlin: {len(berlin)}, Nanjing: {len(nanjing)}, Unknown: {len(unknown)}")

        by_type = Counter(r['query_type'] for r in evaluable)
        for qt, cnt in sorted(by_type.items()):
            print(f"  {qt}: {cnt}")
    except Exception as e:
        print(f"  (skipped: {e})")

    print("\nDone.")
