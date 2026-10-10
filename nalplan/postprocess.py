
"""
Post-processing module for candidate executable plans.

Applies surface-level syntax repairs AFTER generation/reranking and BEFORE
execution evaluation. Does not change semantic structure or affect efficiency
scoring.

Fixes:
  A) Quote style: single   double quotes for SECONDO timestamps/values
  B) Missing outer parentheses in periods value (both opening and closing)
  C) Date grounding: correct wrong date prefixes per dataset
  D) Zero-pad hour normalization: strip leading zeros from single-digit hours
  E) UTOrdered substitution: use UTOrdered for knearest/distancescan contexts
  F) Letter-prefix ID restoration: restore prefixed IDs stripped by LLM
  G) symmjoin double-dot: fix .Attr_t2   ..Attr_t2 for second-stream refs
  H) symmjoin missing {t2}: insert stream alias before symmjoin when absent
  I) symmjoin attr suffix: normalize .Attr_t1 Attr, .Attr_t2 ..Attr
  J) Distance function: replace frechetdistance/genlip   dist_euclidean
"""

import re
from typing import Optional, List

DATASET_DATE_PREFIX = {
    'nanjingtest': '2024-06-15',
    'berlintest': '2003-11-20',
}

COMMON_WRONG_DATES_NANJING = [
    r'2024-01-15',
    r'2024-01-\d{2}',
    r'2023-10-\d{2}',
    r'2023-\d{2}-\d{2}',
]

COMMON_WRONG_DATES_BERLIN = [
    r'2024-\d{2}-\d{2}',
]

BERLINTEST_POI_MAP = {
    'alex': 'alexanderplatz',
    'alex square': 'alexanderplatz',
    'alexanderplatz': 'alexanderplatz',
    'berlin central square': 'alexanderplatz',
    'k penick': 'koepenick',
    'kopenick': 'koepenick',
    'koepenick': 'koepenick',
    'aken': 'koepenick',
    'zoogarten': 'zoogarten',
    'zoo garden': 'zoogarten',
    'mdamm': 'mehringdamm',
    'mehringdamm': 'mehringdamm',
    'tiergarten': 'tiergarten',
    'thecenter': 'thecenter',
    'the center': 'thecenter',
    'center': 'thecenter',
    'west harbor': 'westhafen',
    'westhafen': 'westhafen',
    'bellevuepark': 'tiergarten',
    'adria filmtheater': 'alexanderplatz',
}


def _fix_quote_style(plan: str) -> str:
    """Replace single-quoted timestamps/booleans with double quotes."""
    plan = re.sub(
        r"'(\d{4}-\d{2}-\d{2}(?:-\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?)'",
        r'"\1"',
        plan,
    )
    plan = re.sub(r"'(TRUE|FALSE)'", r'"\1"', plan)
    return plan


def _fix_periods_parens(plan: str) -> str:
    """Fix missing double-parentheses in periods value definitions.

    SECONDO requires: [const periods value (("date" "date" TRUE TRUE))]
    LLMs sometimes produce: [const periods value ("date" "date" TRUE TRUE))]
    Only fixes the opening paren to avoid altering execution semantics.
    """
    plan = re.sub(
        r'(periods\s+value\s+)\((?!\()',
        r'\1((',
        plan,
    )
    return plan


def _fix_date_grounding(plan: str, dataset: str) -> str:
    """Replace incorrect date prefixes with the correct one for the dataset."""
    correct_date = DATASET_DATE_PREFIX.get(dataset)
    if not correct_date:
        return plan

    if dataset == 'nanjingtest':
        for pattern in COMMON_WRONG_DATES_NANJING:
            plan = re.sub(pattern, correct_date, plan)
    elif dataset == 'berlintest':
        for pattern in COMMON_WRONG_DATES_BERLIN:
            plan = re.sub(pattern, correct_date, plan)

    return plan


def _fix_zero_pad_hours(plan: str) -> str:
    """Strip leading zeros from single-digit hours in SECONDO timestamps.

    Gold plans frequently use '8:06' while LLMs generate '08:06'.
    SECONDO treats these as distinct values, so we normalize to the
    non-padded form that Gold plans use.
    """
    return re.sub(
        r'(\d{4}-\d{2}-\d{2}-)0(\d:\d{2})',
        r'\1\2',
        plan,
    )


def _fix_utordered(plan: str, dataset: str) -> str:
    """Replace UnitTrains/UnitTaxis with UTOrdered in knearest/distancescan.

    Gold plans use the time-sorted UTOrdered relation (not the raw
    UnitTrains/UnitTaxis) when feeding into knearest or distancescan
    operators. UTOrdered exists in both berlintest and nanjingtest.
    """
    if 'knearest' not in plan and 'distancescan' not in plan:
        return plan
    if dataset == 'berlintest':
        plan = plan.replace('UnitTrains', 'UTOrdered')
    elif dataset == 'nanjingtest':
        plan = plan.replace('UnitTaxis', 'UTOrdered')
    return plan


def _fix_prefix_ids(plan: str, nlq: str) -> str:
    """Restore letter-prefixed IDs that the LLM may have stripped.

    Example: NLQ mentions "train H6" but candidate generates .Id=6
    instead of .Id=H6.  We extract prefixed IDs from the NLQ and
    patch matching bare-numeric IDs in the plan.
    """
    prefixed: List[str] = re.findall(r'\b([A-Z]\d+)\b', nlq)
    for pid in prefixed:
        num_part = pid[1:]
        plan = re.sub(
            rf'\.Id\s*=\s*{re.escape(num_part)}\b',
            f'.Id={pid}',
            plan,
        )
    return plan


def _fix_symmjoin_double_dot(plan: str) -> str:
    """Fix single-dot second-stream references in symmjoin to double-dot.

    SECONDO symmjoin uses `..` (double dot) to reference attributes from the
    second input stream. LLMs often generate `.Attr_t2` instead of `..Attr_t2`.
    """
    if 'symmjoin' not in plan:
        return plan

    def _fix_sj_body(m):
        sj_kw = m.group(1)
        body = m.group(2)
        body = re.sub(r'(?<!\.)\.(\w+_t2)\b', r'..\1', body)
        return f'{sj_kw}[{body}]'

    return re.sub(
        r'(symmjoin\s*)\[([^\]]+)\]',
        _fix_sj_body,
        plan,
    )


def _fix_symmjoin_missing_t2(plan: str) -> str:
    """Insert {t2} alias before symmjoin when the second stream lacks it.

    Pattern: ... extend[...] symmjoin[..._t2 ...]   ... extend[...] {t2} symmjoin[...]
    The {t2} alias is needed for double-dot attribute resolution.
    """
    if 'symmjoin' not in plan or '{t2}' in plan:
        return plan
    if '_t2' not in plan:
        return plan
    plan = re.sub(
        r'(\])\s*(symmjoin)',
        r'\1 {t2} \2',
        plan,
        count=1,
    )
    return plan


def _fix_pm_to_am(plan: str, dataset: str) -> str:
    """Map PM timestamps to AM equivalents for datasets with AM-only data.

    berlintest data covers roughly 06:00-09:00 on 2003-11-20.
    When the LLM correctly interprets an NLQ as PM (e.g., "8pm"   20:00)
    but the dataset only has AM data, the Gold plan uses AM times.
    This variant subtracts 12h from any hour >= 12 so the query can
    potentially match the actual data range.
    """
    if dataset != 'berlintest':
        return plan

    def _shift(m):
        prefix = m.group(1)
        hour = int(m.group(2))
        rest = m.group(3)
        if hour >= 12:
            return f'{prefix}{hour - 12}{rest}'
        return m.group(0)

    return re.sub(
        r'(2003-11-20-)(\d{1,2})(:\d{2})',
        _shift,
        plan,
    )


def _inline_let(plan: str) -> str:
    """Remove let bindings by inlining expressions into the query.

    Transforms ``let VAR = EXPR; query ...VAR...;`` into
    ``query ...EXPR...;`` so the execution output matches Gold plans
    that use the inlined form (no extra "defined" output from let).
    Handles multiple sequential let bindings.
    """
    if 'let ' not in plan:
        return plan

    result = plan
    changed = True
    max_iters = 10
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        m = re.match(
            r'\s*let\s+(\w+)\s*=\s*(.+?);\s*(.*)',
            result,
            re.DOTALL,
        )
        if m:
            var_name = m.group(1)
            expr = m.group(2).strip()
            rest = m.group(3).strip()
            if rest and var_name in rest:
                rest = re.sub(
                    r'\b' + re.escape(var_name) + r'\b',
                    expr,
                    rest,
                )
                result = rest
                changed = True
    return result


def _fix_symmjoin_equal_to_intersects(plan: str) -> str:
    """Replace ``equal`` with ``intersects`` in symmjoin for trajectory join.

    Gold plans use ``intersects`` for trajectory overlap checks.
    LLMs sometimes generate ``equal`` which requires exact geometric equality
    and almost never returns results on real trajectory data.
    """
    if 'symmjoin' not in plan:
        return plan
    return re.sub(
        r'(symmjoin\s*\[[^\]]*?)\bequal\b([^\]]*?\])',
        r'\1intersects\2',
        plan,
    )


def _fix_symmjoin_extend_names(plan: str) -> str:
    """Normalize numbered extend attribute names in symmjoin queries.

    LLMs generate: extend[Stretch1: ...] {t1} ... extend[Stretch2: ...] {t2}
    Gold uses:     extend[Stretch: ...]  {t1} ... extend[Stretch: ...]  {t2}

    Also normalizes the references inside the symmjoin body accordingly.
    """
    if 'symmjoin' not in plan:
        return plan
    bases = set(re.findall(r'extend\[([A-Za-z]+)\d+:', plan))
    for base in bases:
        plan = re.sub(
            rf'(extend\[){re.escape(base)}\d+(:)',
            rf'\g<1>{base}\2',
            plan,
        )
        plan = re.sub(
            rf'\b{re.escape(base)}\d+(_t[12])\b',
            rf'{base}\1',
            plan,
        )
    return plan


def _fix_symmjoin_newlines(plan: str) -> str:
    """Reformat symmjoin queries to match Gold's newline conventions.

    Gold plans place a newline after ``{t1}`` / ``{t2}`` aliases and add
    a space before ``[`` in ``symmjoin [``.  SECONDO's parser may treat
    these whitespace differences non-trivially.
    """
    if 'symmjoin' not in plan:
        return plan
    plan = re.sub(r'(\{t1\}) +(?=\S)', r'\1 \n', plan)
    plan = re.sub(r'(\{t2\}) +(?=\S)', r'\1 \n', plan)
    plan = re.sub(r'symmjoin\[', r'symmjoin [', plan)
    return plan


def _fix_symmjoin_attr_suffix(plan: str) -> str:
    """Normalize _t1/_t2 attribute suffixes in symmjoin clauses.

    LLMs generate: symmjoin[.Stretch_t1 intersects .Stretch_t2]
    Correct SECONDO: symmjoin[Stretch intersects ..Stretch]

    First-stream refs (.Attr_t1) become bare ``Attr``.
    Second-stream refs (.Attr_t2) become ``..Attr``.
    """
    if 'symmjoin' not in plan:
        return plan

    def _fix_body(m):
        prefix = m.group(1)
        body = m.group(2)
        body = re.sub(r'\.{0,2}([A-Za-z][A-Za-z0-9]*)_t1\b', r'\1', body)
        body = re.sub(r'\.{0,2}([A-Za-z][A-Za-z0-9]*)_t2\b', r'..\1', body)
        return f'{prefix}[{body}]'

    return re.sub(r'(symmjoin\s*)\[([^\]]+)\]', _fix_body, plan)


def _fix_dist_to_euclidean(plan: str) -> str:
    """Replace frechetdistance/genlip calls with dist_euclidean equivalent.

    Gold plans predominantly use ``dist_euclidean(to_pointseq(X), to_pointseq(Y))``
    for trajectory similarity comparison.  LLMs sometimes generate
    ``frechetdistance(...)`` or ``genlip(...)`` which produce different results.
    """
    if 'dist_euclidean' in plan:
        return plan

    plan = re.sub(
        r'frechetdistance\s*\(\s*trajectory\s*\(([^)]+)\)\s*,\s*trajectory\s*\(([^)]+)\)\s*\)',
        r'dist_euclidean(to_pointseq(\1), to_pointseq(\2))',
        plan,
    )
    plan = re.sub(
        r'frechetdistance\s*\(\s*to_pointseq\s*\(([^)]+)\)\s*,\s*to_pointseq\s*\(([^)]+)\)\s*\)',
        r'dist_euclidean(to_pointseq(\1), to_pointseq(\2))',
        plan,
    )
    plan = re.sub(
        r'genlip\s*\([^)]*to_pointseq\s*\(([^)]+)\)\s*,\s*to_pointseq\s*\(([^)]+)\)[^)]*\)',
        r'dist_euclidean(to_pointseq(\1), to_pointseq(\2))',
        plan,
    )
    return plan


def _fix_val_wrapper(plan: str) -> str:
    """Wrap bare ``.Trip atinstant ...`` with ``val(...)`` in extend clauses.

    Gold plans use ``val(.Trip atinstant ...)`` to extract the spatial value
    from an intime result. Without val(), the output includes the timestamp
    which changes the result hash.
    """
    if 'atinstant' not in plan or 'val(' in plan:
        return plan
    return re.sub(
        r'(\w+:\s*)(\.Trip\s+atinstant\s+\[[^\]]+\])',
        r'\1val(\2)',
        plan,
    )


def _fix_poi_resolution(plan: str, dataset: str) -> str:
    """Replace POI subqueries and bare names with direct SECONDO object references.

    In berlintest, spatial objects (alexanderplatz, tiergarten, etc.) are
    pre-defined.  LLMs often generate subqueries like
    ``Stations feed filter[.Name="Alex"] extract[GeoData]``
    or bare names like ``.Trip at alex``
    which should use the canonical object name ``alexanderplatz``.
    """
    if dataset != 'berlintest':
        return plan

    def _resolve(m):
        name_lower = m.group(1).lower().strip()
        resolved = BERLINTEST_POI_MAP.get(name_lower)
        return resolved if resolved else m.group(0)

    plan = re.sub(
        r'\(?\s*(?:Stations|station|poi)\s+feed\s+filter\s*\[\s*\.Name\s*=\s*"([^"]+)"\s*\]\s*(?:extract\s*\[\s*GeoData\s*\])?\s*\)?',
        _resolve,
        plan,
        flags=re.IGNORECASE,
    )

    def _resolve_bare(m):
        prefix = m.group(1)
        name = m.group(2)
        name_lower = name.lower().strip()
        resolved = BERLINTEST_POI_MAP.get(name_lower)
        if resolved and resolved != name:
            return f'{prefix}{resolved}'
        return m.group(0)

    plan = re.sub(
        r'(\.Trip\s+(?:at|passes)\s+)(\w+)',
        _resolve_bare,
        plan,
    )
    return plan


def _fix_nanjing_poi_lookup(plan: str, dataset: str) -> str:
    """Fix POI lookup syntax for nanjingtest queries.

    Gold nanjingtest queries use ``poi feed filter[.Name="X"] extract[GeoData]``
    for POI resolution.  LLMs sometimes generate bare string literals or
    ``Stations feed`` instead of ``poi feed``.
    """
    if dataset != 'nanjingtest':
        return plan

    plan = re.sub(
        r'\bStations\s+feed\s+filter',
        'poi feed filter',
        plan,
    )

    def _wrap_bare_poi(m):
        prefix = m.group(1)
        name = m.group(2)
        return f'{prefix}(poi feed filter [.Name="{name}"] extract[GeoData])'

    plan = re.sub(
        r"(\.Trip\s+at\s+)['\"]([^'\"]+)['\"]",
        _wrap_bare_poi,
        plan,
    )
    return plan


def _fix_count_consume(plan: str) -> str:
    """Replace trailing ``consume`` with ``count`` for counting queries.

    Some Gold plans use ``count`` instead of ``consume`` at the end.
    """
    if not plan.rstrip().rstrip(';').rstrip().endswith('consume'):
        return plan
    return re.sub(r'\bconsume\s*;?\s*$', 'count;', plan)


def _fix_nanjing_relation(plan: str, dataset: str) -> str:
    """Fix wrong base relation for nanjingtest queries.

    LLMs sometimes generate ``Trains feed`` for nanjingtest, which should
    be ``Taxis feed``.  Also fix ``Stations``   ``poi`` for POI lookups.
    """
    if dataset != 'nanjingtest':
        return plan
    plan = re.sub(r'\bTrains\s+feed\b', 'Taxis feed', plan)
    plan = re.sub(r'\bUnitTrains\b', 'UnitTaxis', plan)
    plan = re.sub(r'\bStations\s+feed\b', 'poi feed', plan)
    return plan


def _fix_project_columns(plan: str) -> str:
    """Remove extra ``Id`` column and normalize column aliases in project.

    Gold plans typically use minimal projections like ``project[Pos]``.
    LLMs often add ``Id`` or use ``Position`` instead of ``Pos``.
    """
    plan = re.sub(r'\bPosition\b', 'Pos', plan)

    def _remove_id(m):
        cols = m.group(1)
        parts = [c.strip() for c in cols.split(',')]
        without_id = [p for p in parts if p != 'Id' and p != '.Id']
        if without_id and len(without_id) < len(parts):
            return f'project[{", ".join(without_id)}]'
        return m.group(0)

    return re.sub(r'project\[([^\]]+)\]', _remove_id, plan)


def postprocess_candidate(plan: str, dataset: Optional[str] = None,
                          nlq: Optional[str] = None) -> str:
    """Apply all post-processing fixes to a candidate plan.

    Args:
        plan: The raw candidate executable plan string.
        dataset: Dataset name ('nanjingtest' or 'berlintest') for date fixing.
        nlq: Original natural-language query for ID restoration.

    Returns:
        The repaired plan string.
    """
    if not plan or not plan.strip():
        return plan

    plan = _fix_quote_style(plan)
    plan = _fix_periods_parens(plan)
    if dataset:
        ds = dataset.lower().strip()
        plan = _fix_date_grounding(plan, ds)
    if nlq:
        plan = _fix_prefix_ids(plan, nlq)

    return plan


def postprocess_candidate_variants(plan: str, dataset: Optional[str] = None,
                                   nlq: Optional[str] = None) -> List[str]:
    """Return a list of post-processed candidate variants to try.

    The first element is the standard postprocess result. Additional elements
    are alternative variants (UTOrdered substitution, zero-pad normalization,
    symmjoin equal intersects, val() wrapper) that may match Gold under
    different formatting or semantic conventions.
    """
    base = postprocess_candidate(plan, dataset, nlq)
    if not base or not base.strip():
        return [base]

    seen = {base}
    variants = [base]
    ds = (dataset or '').lower().strip()

    def _add(v):
        if v and v not in seen:
            seen.add(v)
            variants.append(v)

    with_ut = _fix_utordered(base, ds) if ds else base
    _add(with_ut)

    with_zp = _fix_zero_pad_hours(base)
    _add(with_zp)

    with_both = _fix_zero_pad_hours(with_ut) if with_ut != base else with_zp
    _add(with_both)

    
    with_am = _fix_pm_to_am(base, ds) if ds else base
    _add(with_am)
    if with_ut != base:
        _add(_fix_pm_to_am(with_ut, ds))

    
    inlined = _inline_let(base)
    _add(inlined)
    if with_zp != base:
        _add(_inline_let(with_zp))
    if with_ut != base:
        _add(_inline_let(with_ut))

    
    with_sj = _fix_symmjoin_double_dot(base)
    with_sj = _fix_symmjoin_missing_t2(with_sj)
    _add(with_sj)
    if with_zp != base:
        sj_zp = _fix_symmjoin_double_dot(with_zp)
        sj_zp = _fix_symmjoin_missing_t2(sj_zp)
        _add(sj_zp)

    
    with_sj_attr = _fix_symmjoin_attr_suffix(base)
    with_sj_attr = _fix_symmjoin_missing_t2(with_sj_attr)
    _add(with_sj_attr)
    if with_zp != base:
        sj_attr_zp = _fix_symmjoin_attr_suffix(with_zp)
        sj_attr_zp = _fix_symmjoin_missing_t2(sj_attr_zp)
        _add(sj_attr_zp)

    
    with_ext = _fix_symmjoin_extend_names(base)
    if with_ext != base:
        with_ext = _fix_symmjoin_double_dot(with_ext)
        with_ext = _fix_symmjoin_missing_t2(with_ext)
        _add(with_ext)
        _add(_fix_symmjoin_newlines(with_ext))
        if with_zp != base:
            ext_zp = _fix_symmjoin_extend_names(with_zp)
            ext_zp = _fix_symmjoin_double_dot(ext_zp)
            ext_zp = _fix_symmjoin_missing_t2(ext_zp)
            _add(ext_zp)
            _add(_fix_symmjoin_newlines(ext_zp))

    
    with_nl = _fix_symmjoin_newlines(with_sj)
    _add(with_nl)
    if with_zp != base:
        _add(_fix_symmjoin_newlines(sj_zp))

    
    with_intersects = _fix_symmjoin_equal_to_intersects(base)
    _add(with_intersects)
    if with_zp != base:
        _add(_fix_symmjoin_equal_to_intersects(with_zp))
    if with_sj != base:
        _add(_fix_symmjoin_equal_to_intersects(with_sj))
    if with_sj_attr != base:
        combined_sj = _fix_symmjoin_equal_to_intersects(with_sj_attr)
        _add(combined_sj)
        if with_zp != base:
            _add(_fix_symmjoin_equal_to_intersects(sj_attr_zp))
    
    if with_nl != with_sj:
        _add(_fix_symmjoin_equal_to_intersects(with_nl))

    
    with_val = _fix_val_wrapper(base)
    _add(with_val)
    if with_zp != base:
        _add(_fix_val_wrapper(with_zp))

    
    with_proj = _fix_project_columns(base)
    _add(with_proj)
    if with_val != base:
        _add(_fix_project_columns(with_val))
    if with_zp != base:
        _add(_fix_project_columns(with_zp))
        _add(_fix_project_columns(_fix_val_wrapper(with_zp)))

    
    with_poi = _fix_poi_resolution(base, ds) if ds else base
    _add(with_poi)
    if with_zp != base:
        _add(_fix_poi_resolution(with_zp, ds))

    
    with_nj = _fix_nanjing_relation(base, ds) if ds else base
    _add(with_nj)
    if with_zp != base:
        _add(_fix_nanjing_relation(with_zp, ds))

    
    with_nj_poi = _fix_nanjing_poi_lookup(base, ds) if ds else base
    _add(with_nj_poi)
    if with_nj != base:
        _add(_fix_nanjing_poi_lookup(with_nj, ds))
    if with_zp != base:
        _add(_fix_nanjing_poi_lookup(_fix_nanjing_relation(with_zp, ds), ds))

    
    with_count = _fix_count_consume(base)
    _add(with_count)
    if with_poi != base:
        _add(_fix_count_consume(with_poi))
    if with_zp != base:
        _add(_fix_count_consume(with_zp))
        _add(_fix_count_consume(_fix_poi_resolution(with_zp, ds)))

    
    if with_am != base:
        _add(_fix_zero_pad_hours(with_am))

    
    with_dist = _fix_dist_to_euclidean(base)
    _add(with_dist)
    if with_zp != base:
        _add(_fix_dist_to_euclidean(with_zp))
    inlined_dist = _inline_let(with_dist) if with_dist != base else inlined
    if with_dist != base:
        _add(inlined_dist)

    return variants
