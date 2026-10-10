from pathlib import Path

"""
Baseline     

            Text-to-SQL     SECONDO      

1. DIN-SQL  (Decomposed In-Context Learning with Self-Correction)
   -      + Chain-of-Thought +    
   -   : Pourreza & Rafiei, NeurIPS 2023

2. DAIL-SQL (Text-to-SQL Empowered by Large Language Models)
   -         +        few-shot     
   -   : Gao et al., VLDB 2024

3. OpenSearch-SQL (Multi-Agent SQL Generation)
   -        Schema Linker / Planner / Generator / Validator
   -   : OpenSearch multi-agent architecture

         SECONDO            SQL  
      API Schema                   
"""

import os
import re
import json
import time
import random
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)





from nalplan.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
API_BASE_URL = LLM_BASE_URL
API_KEY = LLM_API_KEY
DEFAULT_MODEL = "gpt-4o"

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
EXAMPLE_PATH = os.path.join(PROJECT_ROOT, 'example.xlsx')

QUERY_TYPE_MAPPING = {
    'Time Interval Query': 'Time interval query',
    'Time Point Query': 'Time interval query',
    'Range Query': 'Moving objects range query',
    'Range Query ': 'Moving objects range query',
    'Moving objects range query': 'Moving objects range query',
    'Nearest Neighbor Query': 'Moving objects nearest neighbor query',
    'Join Query': 'Moving objects join query',
    'Join Query ': 'Moving objects join query',
    'Similarity Query': 'Trajectory similarity query',
}

SECONDO_SCHEMA = """
SECONDO Database Schema (berlintest):
- Trains: relation(Id: int, Trip: mpoint, Line: string, Up: bool)
- UnitTrains: relation(Id: int, UTrip: upoint, Line: string, Up: bool)
- Stations: relation(Name: string, GeoData: point, Pos: real)
- Schedules: relation(Id: int, Line: string, Up: bool, Departure: instant)

Data Types:
- mpoint: moving point (trajectory over time)
- upoint: unit of moving point (single temporal segment)
- instant: a point in time
- periods: time interval, syntax: [const periods value (("YYYY-MM-DD-HH:MM" "YYYY-MM-DD-HH:MM" TRUE TRUE))]
- point/region/line: spatial types

Valid location objects: mehringdamm, alexanderplatz, tiergarten, westhafen,
  zoogarten, koepenick, thecenter, train7sections, BGrenzenLine, PotsdamLine, mrainline
Date prefix for all times: 2003-11-20

Key Operators:
- Stream: feed, filter, extend, project, consume, extract, head, sortby, count
- Temporal: atperiods, atinstant, deftime, present, trajectory, val
- Spatial: passes, inside, at, intersects, distance
- NN: knearest, greeceknearest, chinaknearest, distancescan
- Similarity: frechetdistance, dist_euclidean, samplempoint, lcss
- Join: symmjoin, loopjoin, hashjoin, everNearerThan

Syntax Rules:
- Periods use double parentheses: [const periods value (("..." "..." TRUE TRUE))]
- Time strings use double quotes: "2003-11-20-06:00"
- No spaces around = in filter: filter[.Id=1]
- let statements end with semicolon before query: let t1 = ...; query ...
- Comparison: = <> < > <= >= (NOT == or !=)
- Boolean: and / or (NOT && or ||)
"""






@dataclass
class BaselineResult:
    """Baseline           """
    method: str
    nlq: str
    query_type: str
    candidates: List[str]
    reasoning: str = ''
    metadata: Dict = field(default_factory=dict)
    latency_ms: float = 0.0






class LLMClient:
    """   OpenAI Chat Completions   """

    def __init__(self, model: str = DEFAULT_MODEL,
                 api_key: str = API_KEY,
                 base_url: str = API_BASE_URL):
        self.model = model
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=120,
            )
        return self._client

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 4000) -> str:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ''
        except Exception as e:
            logger.error(f"[LLMClient] API call failed: {e}")
            return ''


class ExamplePool:
    """  example.xlsx       few-shot   """

    def __init__(self, path: str = EXAMPLE_PATH):
        self.examples: List[Dict] = []
        self._type_index: Dict[str, List[Dict]] = {}
        self._skeleton_groups: Dict[str, List[str]] = {}
        self._nlq_to_skeleton: Dict[str, str] = {}
        self._load(path)

    @staticmethod
    def _extract_skeleton(executable: str) -> str:
        """Extract structural skeleton   aggressively normalize tables/entities."""
        if not executable or not isinstance(executable, str):
            return ''
        e = executable.lower()
        e = re.sub(r'"[^"]*"', '"<V>"', e)
        e = re.sub(r'\b\d+\.?\d*\b', '<N>', e)
        
        for tbl in ['unittrains', 'unittaxis', 'unitbuses',
                     'trains', 'taxis', 'buses',
                     'utordered', 'stations', 'schedules',
                     'park', 'poi', 'geodata', 'strassen']:
            e = re.sub(r'\b' + tbl + r'\b', '<TBL>', e)
        
        for idx in ['unittrainsrtree', 'unittaxisrtree', 'strassen_geodata_rtree',
                     'unittrains_line_btree']:
            e = e.replace(idx, '<IDX>')
        
        for loc in ['mehringdamm', 'alexanderplatz', 'tiergarten', 'westhafen',
                     'zoogarten', 'koepenick', 'thecenter', 'flaechen',
                     'potsdamline', 'bgrenzenline', 'mrainline',
                     'kinos', 'westend', 'nordkreuz']:
            e = e.replace(loc, '<LOC>')
        
        e = re.sub(r'\s+', ' ', e)
        return e.strip()

    def _load(self, path: str):
        try:
            import pandas as pd
            df = pd.read_excel(path)
            for _, row in df.iterrows():
                cat = str(row.get('cat', '')).strip()
                qtype = QUERY_TYPE_MAPPING.get(cat, cat)
                nlq = str(row.get('review', ''))
                exe = str(row.get('executable language', ''))
                entry = {
                    'nlq': nlq,
                    'executable': exe,
                    'query_type': qtype,
                }
                self.examples.append(entry)
                self._type_index.setdefault(qtype, []).append(entry)

                skel = self._extract_skeleton(exe)
                nlq_key = nlq.strip().lower()
                self._nlq_to_skeleton[nlq_key] = skel
                self._skeleton_groups.setdefault(skel, []).append(nlq_key)

            logger.info(f"[ExamplePool] Loaded {len(self.examples)} examples, "
                        f"{len(self._skeleton_groups)} skeleton groups")
        except Exception as e:
            logger.warning(f"[ExamplePool] Failed to load: {e}")

    def get_by_type(self, query_type: str, k: int = 5,
                    exclude_nlq: str = '') -> List[Dict]:
        pool = self._type_index.get(query_type, [])
        if exclude_nlq:
            excl = exclude_nlq.strip().lower()
            pool = [e for e in pool if e['nlq'].strip().lower() != excl]
        if len(pool) <= k:
            return list(pool)
        return random.sample(pool, k)

    def get_all(self) -> List[Dict]:
        return list(self.examples)

    def is_empty(self) -> bool:
        return not self.examples

    def get_skeleton_exclusion_set(self, nlq: str) -> Set[str]:
        """Get all NLQs sharing the same executable skeleton as the given NLQ."""
        nlq_key = nlq.strip().lower()
        skel = self._nlq_to_skeleton.get(nlq_key, '')
        if not skel:
            return {nlq_key}
        return set(self._skeleton_groups.get(skel, [nlq_key]))

    def get_most_similar(self, nlq: str, query_type: str = '',
                         k: int = 5, mask_entities: bool = False,
                         exclude_nlq: str = '') -> List[Dict]:
        """     Jaccard           few-shot   """
        pool = self._type_index.get(query_type, self.examples) if query_type else self.examples
        if not pool:
            return []
        if exclude_nlq:
            excl = exclude_nlq.strip().lower()
            pool = [e for e in pool if e['nlq'].strip().lower() != excl]

        def _tokenize(text: str) -> Set[str]:
            t = text.lower()
            if mask_entities:
                t = re.sub(r'\b\d+\b', '<NUM>', t)
                t = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)?', '<TIME>', t)
                for loc in ['mehringdamm', 'alexanderplatz', 'tiergarten',
                            'westhafen', 'zoogarten', 'koepenick',
                            'thecenter', 'flaechen']:
                    t = t.replace(loc, '<LOC>')
            return set(t.split())

        nlq_tokens = _tokenize(nlq)
        scored = []
        for ex in pool:
            ex_tokens = _tokenize(ex['nlq'])
            if not nlq_tokens or not ex_tokens:
                scored.append((ex, 0.0))
                continue
            jaccard = len(nlq_tokens & ex_tokens) / len(nlq_tokens | ex_tokens)
            scored.append((ex, jaccard))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:k]]


class EmptyExamplePool(ExamplePool):
    """
                      

            nalspatial       NL->EXE               
                  DAIL-SQL                     
                         
    """

    def __init__(self):
        self.examples = []
        self._type_index = {}
        self._skeleton_groups = {}
        self._nlq_to_skeleton = {}

    def get_by_type(self, query_type: str, k: int = 5,
                    exclude_nlq: str = '') -> List[Dict]:
        return []

    def get_most_similar(self, nlq: str, query_type: str = '',
                         k: int = 5, mask_entities: bool = False,
                         exclude_nlq: str = '') -> List[Dict]:
        return []

    def get_skeleton_exclusion_set(self, nlq: str) -> Set[str]:
        return set()


def _extract_queries_from_text(text: str) -> List[str]:
    """
      LLM       SECONDO      

                consume / count                  
            nalspatial            
      - sum[A] / avg[A] / min[A] / max[A]      
      - extract[GeoData]   
      -    /       ... < 0.5 
    """
    queries = []
    seen = set()

    _END = r'(?:consume|count|sum\s*\[[^\]]*\]|avg\s*\[[^\]]*\]|'\
           r'min\s*\[[^\]]*\]|max\s*\[[^\]]*\]|extract\s*\[[^\]]*\]|'\
           r'aggregateB\s*\[[^\]]*\]|[<>]=?\s*[\d.]+)'

    
    
    _AT_END = r'(?=\s*(?:;|```|\n\s*\n|\Z))'

    
    
    
    _AT_START = r'^[ \t]*(?:\d+[.)]\s*|[-*]\s+)?'

    patterns = [
        _AT_START + r'(?:let\s+[^;]+;\s*)*query\s+.+?' + _END + _AT_END,
        _AT_START + r'query\s+\w+\s+feed\s+.+?' + _END + _AT_END,
    ]

    for pat in patterns:
        for m in re.finditer(pat, text,
                             re.IGNORECASE | re.DOTALL | re.MULTILINE):
            q = m.group(0).strip().rstrip(';').strip()
            q = re.sub(r'^(?:\d+[.)]\s*|[-*]\s+)', '', q).strip()
            if len(q) > 20 and q not in seen:
                queries.append(q)
                seen.add(q)

    if not queries:
        for line in text.split('\n'):
            line = line.strip().lstrip('0123456789.-) ')
            
            
            if re.match(r'(?:let\s|query\s)', line, re.IGNORECASE) and len(line) > 18:
                q = line.rstrip(';').strip()
                if q not in seen:
                    queries.append(q)
                    seen.add(q)

    return queries


def _try_parse_json_queries(text: str) -> List[str]:
    """    JSON     LLM          """
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end <= start:
            start = text.find('[')
            end = text.rfind(']') + 1
        if start != -1 and end > start:
            raw = text[start:end]
            raw = re.sub(r',\s*}', '}', raw)
            raw = re.sub(r',\s*]', ']', raw)
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(q) for q in data if isinstance(q, str) and len(q) > 20]
            if isinstance(data, dict):
                for key in ['queries', 'candidates', 'plans', 'results',
                            'corrected_queries', 'final_queries']:
                    if key in data and isinstance(data[key], list):
                        return [str(q) for q in data[key]
                                if isinstance(q, str) and len(q) > 20]
    except (json.JSONDecodeError, ValueError):
        pass
    return []









class DINSQLBaseline:
    """
    DIN-SQL     

       
                                                                             
      Schema         Classification &       CoT Query          Self-         
      Linking        Decomposition          Generation         Correction    
                                                                             

            
    1. Schema Linking:    NLQ      tables/columns/values
    2. Classification & Decomposition:                   
    3. CoT Generation:    Chain-of-Thought       
    4. Self-Correction: LLM                /    
    """

    METHOD_NAME = 'DIN-SQL'

    def __init__(self, model: str = DEFAULT_MODEL, num_candidates: int = 5,
                 schema: str = SECONDO_SCHEMA,
                 example_pool: Optional[ExamplePool] = None,
                 zero_shot: bool = False):
        self.llm = LLMClient(model=model)
        self.schema = schema
        self.example_pool = (EmptyExamplePool() if zero_shot
                             else (example_pool or ExamplePool()))
        self.num_candidates = num_candidates
        print(f"[DIN-SQL] Initialized with model={model}, "
              f"num_candidates={num_candidates}, zero_shot={zero_shot}")

    
    
    

    def _schema_linking(self, nlq: str) -> Dict:
        prompt = f"""Given the following SECONDO database schema and a natural language query,
identify the relevant schema elements.

{self.schema}

Natural Language Query: "{nlq}"

Identify:
1. Relevant relations (tables) and their attributes
2. Entity values mentioned (object IDs, locations, times, numbers)
3. Relevant operators that might be needed
4. Any spatial/temporal constraints

Output JSON:
{{
  "relations": [{{"name": "...", "attributes": ["..."]}}],
  "entities": {{
    "object_ids": [],
    "locations": [],
    "times": [],
    "numbers": []
  }},
  "candidate_operators": [],
  "constraints": []
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000,
        )

        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1 and end > start:
                raw = resp[start:end]
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                return json.loads(raw)
        except json.JSONDecodeError:
            pass

        return {
            "relations": [],
            "entities": {"object_ids": [], "locations": [], "times": [], "numbers": []},
            "candidate_operators": [],
            "constraints": [],
            "raw_response": resp[:500],
        }

    
    
    

    def _classify_and_decompose(self, nlq: str, schema_info: Dict) -> Dict:
        entities_summary = json.dumps(schema_info.get('entities', {}), ensure_ascii=False)
        operators_summary = ', '.join(schema_info.get('candidate_operators', []))

        prompt = f"""You are a SECONDO database expert. Given a natural language query and its
schema linking results, perform two tasks:

1. CLASSIFY the query difficulty:
   - EASY: single relation, single filter (e.g., "What is the trajectory of train X between T1 and T2?")
   - MEDIUM: requires temporal/spatial joins or knearest (e.g., "Find 5 nearest trains to train X")
   - HARD: multi-step with let-bindings, complex joins (e.g., "Compare trajectories of train X and Y passing location Z")

2. DECOMPOSE complex queries into sequential sub-problems.
   For EASY queries, keep a single step.

Natural Language Query: "{nlq}"
Identified Entities: {entities_summary}
Candidate Operators: {operators_summary}

Output JSON:
{{
  "difficulty": "EASY|MEDIUM|HARD",
  "query_type": "Time interval query | Moving objects range query | Moving objects nearest neighbor query | Moving objects join query | Trajectory similarity query",
  "sub_problems": [
    {{
      "step": 1,
      "description": "...",
      "involves": ["relation/operator names"]
    }}
  ],
  "reasoning": "Brief explanation of the decomposition"
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1500,
        )

        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1 and end > start:
                raw = resp[start:end]
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                return json.loads(raw)
        except json.JSONDecodeError:
            pass

        return {
            "difficulty": "MEDIUM",
            "query_type": "",
            "sub_problems": [{"step": 1, "description": nlq, "involves": []}],
            "reasoning": resp[:500],
        }

    
    
    

    def _cot_generate(self, nlq: str, schema_info: Dict,
                      decomposition: Dict, exclude_nlq: str = '') -> List[str]:
        query_type = decomposition.get('query_type', '')
        fewshot = self.example_pool.get_by_type(
            query_type, k=3, exclude_nlq=exclude_nlq or nlq
        ) if query_type else []

        fewshot_block = ""
        if fewshot:
            fewshot_block = "\n=== Reference Examples ===\n"
            for i, ex in enumerate(fewshot, 1):
                fewshot_block += (f"Example {i}:\n"
                                  f"  NLQ: {ex['nlq'][:150]}\n"
                                  f"  Query: {ex['executable'][:300]}\n\n")

        sub_problems = decomposition.get('sub_problems', [])
        
        
        steps_block = "\n".join(
            f"  Step {sp.get('step', i)}: "
            f"{sp.get('description', sp) if isinstance(sp, dict) else sp}"
            for i, sp in enumerate(sub_problems, 1)
        )

        entities = json.dumps(schema_info.get('entities', {}), ensure_ascii=False)
        difficulty = decomposition.get('difficulty', 'MEDIUM')

        prompt = f"""You are a SECONDO spatio-temporal database expert.

{self.schema}
{fewshot_block}
=== TASK ===
Generate SECONDO executable queries for the following natural language query.

NLQ: "{nlq}"
Difficulty: {difficulty}
Identified Entities: {entities}

Decomposed Steps:
{steps_block}

Think step by step (Chain-of-Thought):
1. For each sub-problem, determine the SECONDO operators needed
2. Construct the query fragment for each step
3. Combine fragments into complete executable queries
4. Verify bracket balance and syntax correctness

Generate {self.num_candidates} candidate queries using different operator choices
or structural variations where possible.

CRITICAL SYNTAX RULES:
- Periods: [const periods value (("2003-11-20-06:00" "2003-11-20-10:00" TRUE TRUE))]
- Time strings: double quotes "2003-11-20-HH:MM"
- filter: NO spaces around =   filter[.Id=1]
- let must end with ; before query
- Use = not == for equality

Output JSON:
{{
  "chain_of_thought": "step-by-step reasoning...",
  "candidates": [
    "query ...",
    "query ..."
  ]
}}"""

        resp = self.llm.chat(
            [{"role": "system", "content": "You are an expert in SECONDO database query generation."},
             {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=4000,
        )

        queries = _try_parse_json_queries(resp)
        if not queries:
            queries = _extract_queries_from_text(resp)
        return queries

    
    
    

    def _self_correct(self, nlq: str, candidates: List[str],
                      schema_info: Dict) -> List[str]:
        if not candidates:
            return []

        candidates_block = "\n".join(
            f"  Candidate {i}: {q}" for i, q in enumerate(candidates, 1)
        )

        prompt = f"""You are a SECONDO query validator and corrector.

{self.schema}

Original NLQ: "{nlq}"
Schema Linking: {json.dumps(schema_info.get('entities', {}), ensure_ascii=False)}

Generated Candidates:
{candidates_block}

For EACH candidate, check:
1. SYNTAX: bracket balance (count [ ] and ( ) pairs), semicolons between let/query
2. PERIODS FORMAT: Must use double parentheses (("..." "..." TRUE TRUE))
3. TIME FORMAT: Must use double quotes "2003-11-20-HH:MM"
4. FILTER: No spaces around = in filter[.Id=1]
5. OPERATOR USAGE: operator parameters match expected types
6. ENTITY CORRECTNESS: referenced IDs/locations/times match the NLQ
7. COMPLETENESS: query ends with consume or count

Fix any errors found. If a candidate is correct, keep it unchanged.
If a candidate is unfixable, discard it.

Output JSON:
{{
  "analysis": [
    {{
      "candidate_id": 1,
      "issues_found": ["issue1", ...],
      "is_fixable": true,
      "corrected": "fixed query ..."
    }}
  ],
  "corrected_queries": [
    "query1...",
    "query2..."
  ]
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=4000,
        )

        corrected = _try_parse_json_queries(resp)
        if not corrected:
            corrected = _extract_queries_from_text(resp)
        if not corrected:
            corrected = candidates

        return corrected

    
    
    

    def generate(self, nlq: str, exclude_nlq: str = '') -> BaselineResult:
        """
        DIN-SQL      

        Args:
            nlq:       
            exclude_nlq: LOO - exclude this NLQ from few-shot pool

        Returns:
            BaselineResult
        """
        t0 = time.time()

        
        schema_info = self._schema_linking(nlq)

        
        decomposition = self._classify_and_decompose(nlq, schema_info)
        query_type = decomposition.get('query_type', '')

        
        candidates = self._cot_generate(nlq, schema_info, decomposition,
                                         exclude_nlq=exclude_nlq or nlq)

        
        corrected = self._self_correct(nlq, candidates, schema_info)

        latency = (time.time() - t0) * 1000

        return BaselineResult(
            method=self.METHOD_NAME,
            nlq=nlq,
            query_type=query_type,
            candidates=corrected,
            reasoning=decomposition.get('reasoning', ''),
            metadata={
                'schema_info': schema_info,
                'decomposition': decomposition,
                'raw_candidates': candidates,
                'difficulty': decomposition.get('difficulty', ''),
            },
            latency_ms=latency,
        )









class DAILSQLBaseline:
    """
    DAIL-SQL     

              few-shot         prompt     

       
    1. Masked Question Similarity (MQS):
           ID/  /         NLQ    
    2. Query Similarity Selection (QSS):
                             
    3. DAIL Selection = MQS + QSS      

          "Question + SECONDO Query"     few-shot 
          Code Representation prompt    
    """

    METHOD_NAME = 'DAIL-SQL'

    def __init__(self, model: str = DEFAULT_MODEL, num_candidates: int = 5,
                 num_shots: int = 8, sim_exclude_threshold: float = 0.0,
                 structural_loo: bool = False,
                 exclusion_sets_path: str = '',
                 schema: str = SECONDO_SCHEMA,
                 example_pool: Optional[ExamplePool] = None,
                 zero_shot: bool = False):
        self.llm = LLMClient(model=model)
        self.schema = schema
        self.example_pool = (EmptyExamplePool() if zero_shot
                             else (example_pool or ExamplePool()))
        self.num_candidates = num_candidates
        self.num_shots = num_shots
        self.sim_exclude_threshold = sim_exclude_threshold
        self.structural_loo = structural_loo
        self._structure_cache: Dict[str, str] = {}
        self._exclusion_sets: Dict[str, Set[str]] = {}

        if exclusion_sets_path and os.path.exists(exclusion_sets_path):
            import json as _json
            with open(exclusion_sets_path, 'r', encoding='utf-8') as f:
                raw = _json.load(f)
            self._exclusion_sets = {k: set(v) for k, v in raw.items()}
            mode = f"precomputed-LOO ({len(self._exclusion_sets)} sets)"
        elif structural_loo:
            mode = "structural-LOO"
        elif sim_exclude_threshold > 0:
            mode = f"sim_exclude {sim_exclude_threshold}"
        else:
            mode = "basic-LOO"
        print(f"[DAIL-SQL] Initialized: model={model}, shots={num_shots}, "
              f"candidates={num_candidates}, mode={mode}")

    
    
    

    @staticmethod
    def _mask_entities(text: str) -> str:
        """                          """
        t = text.lower()
        t = re.sub(r'\b\d+\b', '<NUM>', t)
        t = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm)?', '<TIME>', t, flags=re.IGNORECASE)
        for loc in ['mehringdamm', 'alexanderplatz', 'tiergarten',
                     'westhafen', 'zoogarten', 'koepenick', 'thecenter',
                     'flaechen', 'potsdamline', 'bgrenzenline', 'mrainline']:
            t = t.replace(loc, '<LOC>')
        return t

    @staticmethod
    def _extract_query_skeleton(executable: str) -> str:
        """                       """
        if not executable or not isinstance(executable, str):
            return ''
        e = executable.lower()
        e = re.sub(r'"[^"]*"', '"<VAL>"', e)
        e = re.sub(r'\b\d+\.?\d*\b', '<N>', e)
        for loc in ['mehringdamm', 'alexanderplatz', 'tiergarten',
                     'westhafen', 'zoogarten', 'koepenick', 'thecenter']:
            e = e.replace(loc, '<LOC>')
        return e

    def _jaccard(self, s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _select_examples(self, nlq: str, query_type: str = '',
                         exclude_nlq: str = '') -> List[Dict]:
        """
        DAIL Selection: MQS + QSS with fair LOO.
        Priority: precomputed exclusion > structural LOO > sim threshold > basic LOO.
        """
        pool = (self.example_pool._type_index.get(query_type, [])
                if query_type else self.example_pool.examples)
        if not pool:
            return []

        excl_nlq = (exclude_nlq or nlq).strip().lower()

        if self._exclusion_sets:
            excl_set = self._exclusion_sets.get(excl_nlq, {excl_nlq})
            pool = [e for e in pool if e['nlq'].strip().lower() not in excl_set]
        elif self.structural_loo:
            excl_set = self.example_pool.get_skeleton_exclusion_set(excl_nlq)
            pool = [e for e in pool if e['nlq'].strip().lower() not in excl_set]
        elif self.sim_exclude_threshold > 0:
            masked_nlq_t = self._mask_entities(nlq)
            nlq_tokens_t = set(masked_nlq_t.split())
            pool = [e for e in pool
                    if e['nlq'].strip().lower() != excl_nlq and
                    self._jaccard(nlq_tokens_t, set(self._mask_entities(e['nlq']).split()))
                    < self.sim_exclude_threshold]
        else:
            pool = [e for e in pool if e['nlq'].strip().lower() != excl_nlq]

        if not pool:
            return []

        masked_nlq = self._mask_entities(nlq)
        nlq_tokens = set(masked_nlq.split())

        scored = []
        for ex in pool:
            masked_ex = self._mask_entities(ex['nlq'])
            ex_tokens = set(masked_ex.split())
            mqs = self._jaccard(nlq_tokens, ex_tokens)

            skeleton = self._extract_query_skeleton(ex['executable'])
            skeleton_tokens = set(skeleton.split())
            qss_scores = []
            for other in random.sample(pool, min(20, len(pool))):
                if other is ex:
                    continue
                other_skel = set(self._extract_query_skeleton(other['executable']).split())
                qss_scores.append(self._jaccard(skeleton_tokens, other_skel))
            qss = sum(qss_scores) / len(qss_scores) if qss_scores else 0.0

            combined = 0.7 * mqs + 0.3 * qss
            scored.append((ex, combined, mqs, qss))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = []
        seen_skeletons: Set[str] = set()
        for ex, score, mqs, qss in scored:
            skel = self._extract_query_skeleton(ex['executable'])
            if skel in seen_skeletons and len(selected) >= 3:
                continue
            seen_skeletons.add(skel)
            selected.append(ex)
            if len(selected) >= self.num_shots:
                break

        return selected

    
    
    

    def _build_prompt(self, nlq: str, examples: List[Dict]) -> str:
        """
           DAIL-SQL     Code Representation prompt

           
        /* Given SECONDO schema ... */
        -- Example 1: NLQ
        QUERY ...
        -- Example 2: NLQ
        QUERY ...
        -- Target: NLQ
        """
        prompt_parts = [f"/* {self.schema.strip()} */\n"]

        for i, ex in enumerate(examples, 1):
            nlq_ex = ex['nlq'][:200]
            exec_ex = ex['executable'][:400]
            prompt_parts.append(f"-- Example {i}: {nlq_ex}")
            prompt_parts.append(f"{exec_ex}\n")

        prompt_parts.append(f"-- Target: {nlq}")
        prompt_parts.append("-- Generate the SECONDO executable query:")

        return "\n".join(prompt_parts)

    def generate(self, nlq: str, query_type: str = '',
                 exclude_nlq: str = '') -> BaselineResult:
        """
        DAIL-SQL      

        Args:
            nlq:       
            query_type:              few-shot   
            exclude_nlq: LOO - exclude this NLQ from few-shot pool

        Returns:
            BaselineResult
        """
        t0 = time.time()

        
        examples = self._select_examples(nlq, query_type,
                                          exclude_nlq=exclude_nlq or nlq)

        
        code_prompt = self._build_prompt(nlq, examples)

        system_msg = (
            "You are a SECONDO spatio-temporal database expert. "
            "Generate valid SECONDO executable queries based on the examples provided. "
            "Output ONLY the executable queries, one per line. "
            f"Generate {self.num_candidates} different valid candidate queries.\n\n"
            "CRITICAL SYNTAX RULES:\n"
            "- Periods: [const periods value ((\"2003-11-20-06:00\" \"2003-11-20-10:00\" TRUE TRUE))]\n"
            "- Time: double quotes \"2003-11-20-HH:MM\"\n"
            "- filter: NO spaces around =   filter[.Id=1]\n"
            "- let ends with ; before query\n"
            "- Use = not ==, use <> not !=\n"
        )

        resp = self.llm.chat(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": code_prompt}],
            temperature=0.7, max_tokens=4000,
        )

        candidates = _extract_queries_from_text(resp)
        if not candidates:
            candidates = _try_parse_json_queries(resp)

        
        if len(candidates) < 2:
            resp2 = self.llm.chat(
                [{"role": "system", "content": system_msg},
                 {"role": "user", "content": code_prompt}],
                temperature=0.9, max_tokens=4000,
            )
            extra = _extract_queries_from_text(resp2)
            seen = set(candidates)
            for q in extra:
                if q not in seen:
                    candidates.append(q)
                    seen.add(q)

        latency = (time.time() - t0) * 1000

        detected_type = query_type
        if not detected_type and examples:
            type_counts: Dict[str, int] = {}
            for ex in examples[:3]:
                t = ex.get('query_type', '')
                type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts:
                detected_type = max(type_counts, key=type_counts.get)

        return BaselineResult(
            method=self.METHOD_NAME,
            nlq=nlq,
            query_type=detected_type,
            candidates=candidates,
            reasoning=f"Selected {len(examples)} DAIL examples "
                      f"(MQS+QSS), generated {len(candidates)} candidates",
            metadata={
                'num_examples_selected': len(examples),
                'example_nlqs': [ex['nlq'][:80] for ex in examples[:5]],
            },
            latency_ms=latency,
        )








class OpenSearchSQLBaseline:
    """
    OpenSearch-SQL      (Multi-Agent Architecture)

             

                      
       Coordinator             
                      
             
                      
                      
                      
                              
       Schema        Query    
       Linker       Planner   
                              
                        
                        
                 
                     
            Query              Agent      
          Generators 
          ( 3 agents) 
                     
               
               
                     
           Validator        +     
           & Voter   
                     

       Agent      LLM       system prompt /    
    - Generator A:           
    - Generator B:              
    - Generator C:           

        Validator Agent          
    """

    METHOD_NAME = 'OpenSearch-SQL'

    def __init__(self, model: str = DEFAULT_MODEL, num_candidates: int = 5,
                 schema: str = SECONDO_SCHEMA,
                 example_pool: Optional[ExamplePool] = None,
                 zero_shot: bool = False):
        self.llm = LLMClient(model=model)
        self.schema = schema
        self.example_pool = (EmptyExamplePool() if zero_shot
                             else (example_pool or ExamplePool()))
        self.num_candidates = num_candidates
        print(f"[OpenSearch-SQL] Initialized with model={model}, "
              f"num_candidates={num_candidates}")

    
    
    

    def _agent_schema_linker(self, nlq: str) -> Dict:
        prompt = f"""[ROLE] You are a Schema Linking Agent for the SECONDO spatio-temporal database.

{self.schema}

[TASK] Analyze the natural language query and link it to relevant schema elements.

NLQ: "{nlq}"

Provide:
1. Which relations are needed (Trains, UnitTrains, Stations, Schedules)
2. Which attributes will be used (.Id, .Trip, .UTrip, .Line, .Name, .GeoData, .Departure)
3. Entity values: object IDs (integers), locations (from valid list), times
4. Data type mappings (which entities map to mpoint/upoint/instant/periods/point/region)

Output JSON:
{{
  "relations": ["Trains", ...],
  "attributes": [".Id", ".Trip", ...],
  "entities": {{
    "object_ids": [30],
    "locations": ["mehringdamm"],
    "time_start": "HH:MM",
    "time_end": "HH:MM",
    "k_value": null,
    "distance": null
  }},
  "type_mappings": {{
    "entity_name": "secondo_type"
  }}
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000,
        )

        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1 and end > start:
                raw = resp[start:end]
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                return json.loads(raw)
        except json.JSONDecodeError:
            pass

        return {"relations": ["Trains"], "attributes": [".Id", ".Trip"],
                "entities": {}, "type_mappings": {}, "raw": resp[:500]}

    
    
    

    def _agent_query_planner(self, nlq: str, schema_info: Dict) -> Dict:
        prompt = f"""[ROLE] You are a Query Planning Agent for the SECONDO database.

{self.schema}

[CONTEXT]
NLQ: "{nlq}"
Schema Linking Result: {json.dumps(schema_info, ensure_ascii=False, default=str)[:1500]}

[TASK] Create a query execution plan:
1. Classify the query type (Time interval / Range / Nearest neighbor / Join / Trajectory similarity)
2. Determine the operator pipeline (sequence of SECONDO operators)
3. Identify if let-bindings are needed (for multi-step queries)
4. Plan the query structure: simple (single query) vs compound (let + query)

Output JSON:
{{
  "query_type": "...",
  "structure": "simple|compound",
  "needs_let_binding": false,
  "operator_pipeline": ["feed", "filter", "...", "consume"],
  "plan_description": "Step-by-step description...",
  "key_operators": ["primary operator for this query type"]
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1500,
        )

        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1 and end > start:
                raw = resp[start:end]
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                return json.loads(raw)
        except json.JSONDecodeError:
            pass

        return {"query_type": "", "structure": "simple",
                "operator_pipeline": [], "plan_description": resp[:500]}

    
    
    

    def _agent_generator_conservative(self, nlq: str, schema_info: Dict,
                                       plan: Dict, exclude_nlq: str = '') -> List[str]:
        """Generator A:               """
        query_type = plan.get('query_type', '')
        examples = self.example_pool.get_by_type(
            query_type, k=3, exclude_nlq=exclude_nlq or nlq
        ) if query_type else []

        examples_block = ""
        if examples:
            examples_block = "\n[REFERENCE EXAMPLES]\n"
            for i, ex in enumerate(examples, 1):
                examples_block += (f"NLQ: {ex['nlq'][:150]}\n"
                                   f"Query: {ex['executable'][:300]}\n\n")

        prompt = f"""[ROLE] You are a CONSERVATIVE Query Generator for SECONDO.
Your strategy: strictly follow established patterns and reference examples.
Minimize risk. Use proven operator combinations.

{self.schema}
{examples_block}
[CONTEXT]
NLQ: "{nlq}"
Query Plan: {json.dumps(plan, ensure_ascii=False, default=str)[:1000]}
Schema Info: {json.dumps(schema_info.get('entities', {}), ensure_ascii=False, default=str)}

[TASK] Generate 2 safe, template-following SECONDO queries.
Follow the reference examples closely. Only change entity values.

SYNTAX RULES: periods use (("...")), time uses double quotes, filter[.Id=1] no spaces, let ends with ;

Output the queries directly, one per line, starting with "query" or "let":"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=2000,
        )
        return _extract_queries_from_text(resp)

    def _agent_generator_explorative(self, nlq: str, schema_info: Dict,
                                      plan: Dict) -> List[str]:
        """Generator B:                """
        prompt = f"""[ROLE] You are an EXPLORATIVE Query Generator for SECONDO.
Your strategy: try different operator combinations and creative approaches.
Explore alternative operators that achieve the same goal.

{self.schema}

[CONTEXT]
NLQ: "{nlq}"
Query Plan: {json.dumps(plan, ensure_ascii=False, default=str)[:1000]}
Schema Info: {json.dumps(schema_info.get('entities', {}), ensure_ascii=False, default=str)}

[TASK] Generate 2 CREATIVE SECONDO queries using different operators/structures than typical.
For example:
- If knearest is typical, try distancescan or chinaknearest
- If simple query, try extend+project variation
- If atperiods, consider deftime+intersects alternative

SYNTAX RULES: periods use (("...")), time uses double quotes, filter[.Id=1] no spaces, let ends with ;

Output the queries directly, one per line, starting with "query" or "let":"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=2000,
        )
        return _extract_queries_from_text(resp)

    def _agent_generator_optimizer(self, nlq: str, schema_info: Dict,
                                    plan: Dict) -> List[str]:
        """Generator C:             """
        prompt = f"""[ROLE] You are an OPTIMIZATION-FOCUSED Query Generator for SECONDO.
Your strategy: generate queries optimized for execution efficiency.
Prefer operators with index support, minimize data scans, use UnitTrains where beneficial.

{self.schema}

Performance tips:
- UnitTrains with deftime filter is faster than Trains with atperiods for time-bounded queries
- distancescan uses spatial index, potentially faster than brute-force knearest
- filter early (push predicates close to feed) to reduce intermediate tuples
- Use head[N] to limit results when only top-K is needed
- Avoid unnecessary extend/project

[CONTEXT]
NLQ: "{nlq}"
Query Plan: {json.dumps(plan, ensure_ascii=False, default=str)[:1000]}
Schema Info: {json.dumps(schema_info.get('entities', {}), ensure_ascii=False, default=str)}

[TASK] Generate 2 EFFICIENCY-OPTIMIZED SECONDO queries.

SYNTAX RULES: periods use (("...")), time uses double quotes, filter[.Id=1] no spaces, let ends with ;

Output the queries directly, one per line, starting with "query" or "let":"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=2000,
        )
        return _extract_queries_from_text(resp)

    
    
    

    def _agent_validator_voter(self, nlq: str, candidates: List[str],
                                plan: Dict) -> List[str]:
        """           """
        if not candidates:
            return []

        candidates_block = "\n".join(
            f"  [{i}] {q}" for i, q in enumerate(candidates, 1)
        )

        prompt = f"""[ROLE] You are a Validation and Voting Agent for SECONDO queries.

{self.schema}

[CONTEXT]
NLQ: "{nlq}"
Expected Query Type: {plan.get('query_type', 'unknown')}
Expected Operators: {plan.get('key_operators', [])}

[CANDIDATES]
{candidates_block}

[TASK] For each candidate:
1. CHECK syntax: bracket balance, semicolons, quote style, periods format
2. CHECK semantics: does the query answer the NLQ? correct entity values?
3. CHECK operators: are they used with correct parameter types?
4. SCORE each candidate 0-10

Then RANK all valid candidates by score and output the top ones.
If any candidate has syntax errors that you can fix, fix them.
Discard candidates that are fundamentally wrong.

Output JSON:
{{
  "evaluations": [
    {{"id": 1, "score": 8, "issues": [], "fixed": null}},
    ...
  ],
  "ranked_queries": [
    "best query...",
    "second best...",
    ...
  ]
}}"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=4000,
        )

        ranked = _try_parse_json_queries(resp)
        if not ranked:
            ranked = _extract_queries_from_text(resp)
        if not ranked:
            ranked = candidates

        return ranked

    
    
    

    def generate(self, nlq: str, exclude_nlq: str = '') -> BaselineResult:
        """
        OpenSearch-SQL      

        Args:
            nlq:       
            exclude_nlq: LOO - exclude this NLQ from few-shot pool

        Returns:
            BaselineResult
        """
        t0 = time.time()

        
        schema_info = self._agent_schema_linker(nlq)

        
        plan = self._agent_query_planner(nlq, schema_info)
        query_type = plan.get('query_type', '')

        
        candidates_a = self._agent_generator_conservative(
            nlq, schema_info, plan, exclude_nlq=exclude_nlq or nlq)
        candidates_b = self._agent_generator_explorative(nlq, schema_info, plan)
        candidates_c = self._agent_generator_optimizer(nlq, schema_info, plan)

        all_candidates = []
        seen: Set[str] = set()
        for q in candidates_a + candidates_b + candidates_c:
            if q not in seen:
                all_candidates.append(q)
                seen.add(q)

        
        ranked = self._agent_validator_voter(nlq, all_candidates, plan)

        latency = (time.time() - t0) * 1000

        return BaselineResult(
            method=self.METHOD_NAME,
            nlq=nlq,
            query_type=query_type,
            candidates=ranked[:self.num_candidates],
            reasoning=(f"Multi-agent: SchemaLinker Planner "
                       f"3 Generators({len(candidates_a)}+"
                       f"{len(candidates_b)}+{len(candidates_c)}) "
                       f"Validator({len(all_candidates)} {len(ranked)})"),
            metadata={
                'schema_info': schema_info,
                'plan': plan,
                'generator_counts': {
                    'conservative': len(candidates_a),
                    'explorative': len(candidates_b),
                    'optimizer': len(candidates_c),
                    'total_unique': len(all_candidates),
                },
                'candidates_before_vote': all_candidates,
            },
            latency_ms=latency,
        )








class BaselineRunner:
    """
        Baseline    

              baseline         
    """

    BASELINES = {
        'DIN-SQL': DINSQLBaseline,
        'DAIL-SQL': DAILSQLBaseline,
        'OpenSearch-SQL': OpenSearchSQLBaseline,
    }

    def __init__(self, model: str = DEFAULT_MODEL,
                 num_candidates: int = 5,
                 methods: Optional[List[str]] = None,
                 schema: str = SECONDO_SCHEMA,
                 example_pool: Optional[ExamplePool] = None,
                 zero_shot: bool = False):
        """
        Args:
            model: LLM model name
            num_candidates:    baseline       
            methods:                
            schema:      schema    berlintest      schema       
            example_pool:     few-shot    
            zero_shot: True     baseline        NL->EXE     
        """
        self.instances: Dict[str, Any] = {}
        methods = methods or list(self.BASELINES.keys())

        for name in methods:
            if name not in self.BASELINES:
                logger.warning(f"Unknown baseline: {name}, skipping")
                continue
            cls = self.BASELINES[name]
            self.instances[name] = cls(model=model,
                                       num_candidates=num_candidates,
                                       schema=schema,
                                       example_pool=example_pool,
                                       zero_shot=zero_shot)

        print(f"\n[BaselineRunner] Initialized {len(self.instances)} baselines: "
              f"{list(self.instances.keys())}  (zero_shot={zero_shot})")

    def run_single(self, nlq: str, method: str,
                   query_type: str = '',
                   exclude_nlq: str = '') -> Optional[BaselineResult]:
        """     baseline (with LOO exclusion)"""
        if method not in self.instances:
            logger.error(f"Method {method} not initialized")
            return None

        instance = self.instances[method]
        if method == 'DAIL-SQL':
            return instance.generate(nlq, query_type=query_type,
                                      exclude_nlq=exclude_nlq or nlq)
        else:
            return instance.generate(nlq, exclude_nlq=exclude_nlq or nlq)

    def run_all(self, nlq: str,
                query_type: str = '') -> Dict[str, BaselineResult]:
        """     baseline"""
        results = {}
        for name in self.instances:
            print(f"\n{'='*60}")
            print(f"Running {name} ...")
            print('='*60)
            result = self.run_single(nlq, name, query_type)
            if result:
                results[name] = result
                print(f"[{name}] Generated {len(result.candidates)} candidates "
                      f"in {result.latency_ms:.0f}ms")
                for i, q in enumerate(result.candidates[:3], 1):
                    display = q[:120] + '...' if len(q) > 120 else q
                    print(f"  {i}. {display}")
        return results

    def run_batch(self, test_cases: List[Dict],
                  methods: Optional[List[str]] = None) -> List[Dict]:
        """
              

        Args:
            test_cases: [{"nlq": "...", "query_type": "..."}, ...]
            methods:             

        Returns:
            [{"nlq": ..., "query_type": ..., "results": {method: BaselineResult}}]
        """
        methods = methods or list(self.instances.keys())
        batch_results = []

        for idx, tc in enumerate(test_cases):
            nlq = tc.get('nlq', '')
            qtype = tc.get('query_type', tc.get('cat', ''))

            print(f"\n{'#'*60}")
            print(f"Test Case {idx+1}/{len(test_cases)}: {nlq[:80]}")
            print('#'*60)

            case_results = {}
            for method in methods:
                if method not in self.instances:
                    continue
                result = self.run_single(nlq, method, qtype)
                if result:
                    case_results[method] = result

            batch_results.append({
                'nlq': nlq,
                'query_type': qtype,
                'results': case_results,
            })

        return batch_results

    @staticmethod
    def save_results(batch_results: List[Dict], output_path: str):
        """         JSON"""
        serializable = []
        for item in batch_results:
            entry = {
                'nlq': item['nlq'],
                'query_type': item['query_type'],
                'results': {},
            }
            for method, result in item.get('results', {}).items():
                entry['results'][method] = {
                    'method': result.method,
                    'query_type': result.query_type,
                    'candidates': result.candidates,
                    'reasoning': result.reasoning,
                    'latency_ms': result.latency_ms,
                    'num_candidates': len(result.candidates),
                }
            serializable.append(entry)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\n[BaselineRunner] Results saved to {output_path}")

    @staticmethod
    def print_comparison(batch_results: List[Dict]):
        """      """
        method_stats: Dict[str, Dict] = {}

        for item in batch_results:
            for method, result in item.get('results', {}).items():
                if method not in method_stats:
                    method_stats[method] = {
                        'total': 0, 'total_candidates': 0,
                        'total_latency': 0.0, 'non_empty': 0,
                    }
                stats = method_stats[method]
                stats['total'] += 1
                stats['total_candidates'] += len(result.candidates)
                stats['total_latency'] += result.latency_ms
                if result.candidates:
                    stats['non_empty'] += 1

        print(f"\n{'='*70}")
        print("Baseline Comparison Summary")
        print('='*70)
        print(f"{'Method':<20} {'Cases':>6} {'NonEmpty':>8} "
              f"{'AvgCand':>8} {'AvgLatency':>12}")
        print('-'*60)

        for method, stats in method_stats.items():
            n = stats['total']
            if n == 0:
                continue
            avg_cand = stats['total_candidates'] / n
            avg_lat = stats['total_latency'] / n
            print(f"{method:<20} {n:>6} {stats['non_empty']:>8} "
                  f"{avg_cand:>8.1f} {avg_lat:>10.0f}ms")






def load_test_cases_csv(path: str) -> List[Dict]:
    """  testcase CSV       """
    import pandas as pd
    df = pd.read_csv(path)
    cases = []
    for _, row in df.iterrows():
        cases.append({
            'nlq': str(row.get('nlq', '')),
            'query_type': str(row.get('cat', '')),
        })
    return cases


if __name__ == '__main__':
    print("=" * 70)
    print("Baseline Methods for Comparison Experiments")
    print("=" * 70)

    test_nlq = "Find the 5 nearest trains to train 30 between 8am and 10am."

    runner = BaselineRunner(model=DEFAULT_MODEL, num_candidates=5)
    results = runner.run_all(test_nlq)

    for method, result in results.items():
        print(f"\n{'='*60}")
        print(f"{method}: {len(result.candidates)} candidates, "
              f"{result.latency_ms:.0f}ms")
        print(f"Query Type: {result.query_type}")
        print(f"Reasoning: {result.reasoning[:200]}")
        for i, q in enumerate(result.candidates, 1):
            print(f"  {i}. {q}")
