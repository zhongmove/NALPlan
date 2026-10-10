



"""
      LLM-based Cost Model     

                      
1. **      (IndexKB)**:   SECONDO      Algebra             
   - 12      : B-tree, Hash, BTree2, R-tree, R-tree3, TB-tree,
     X-tree, M-tree, SETI, UGrid, MMRTree join, spatialjoin/cdacspatialjoin
   -         (UnitTrains_Line_btree, UnitTrainsrtree, strassen_GeoData_rtree  )
   -                 
   -        (     ->         )
   - SECONDO Cost      (from improvedcosts.pl / operators.pl / costs2014.pl)

2. **      (IndexEnumerator)**:               (11      )
   P1:       +    filter   R-tree windowintersects
   P2:       + ID/equality filter   B-tree exactmatch
   P3:  sort + head(k)   distancescan (NN via R-tree)
   P4:  symmjoin   loopjoin + R-tree index
   P5:  inline creatertree            
   P6:  string equality filter   Hash exactmatch (O(1))
   P7:  symmjoin   spatialjoin (plane-sweep O(n+m+k))
   P8:  temporal/spatial filter on upoint   TB-tree windowintersects
   P9:  distancescan variant selection (distancescan/2/4)
   P10: knearestfilter   distancescan via R-tree
   P11: symmjoin   realJoinMMRTree (main-memory R-tree join)

3. **DBA       (DBAExperienceKB)**:    DBA              
   - 5    : index_selection, join_strategy, filter_ordering, cost_pattern, general
   -      (   DBA   ) + LLM               
   -      JSON           
   -       LLM prompt                
   -       :      /            

4. **LLM-based Cost Model**:    SECONDO         + DBA             
   -   :            (  /    )      SECONDO cost    DBA   
   -    GPT-4o              
   - LLM              insights            
   -      +      fallback      
   -      +     

Author: System
Date: 2026-03-12
"""
import os
import re
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import signal
from nalplan.config import ALGEBRAS_PATH
ALGEBRAS_DIR = str(ALGEBRAS_PATH)
OPTIMIZER_DIR = str(ALGEBRAS_PATH.parent / 'Optimizer')
INDEX_KB_CACHE = os.path.join(os.path.dirname(__file__), 'index_kb_cache.json')
DBA_EXPERIENCE_KB_CACHE = os.path.join(os.path.dirname(__file__), 'dba_experience_kb.json')
from nalplan.config import LLM_BASE_URL, LLM_API_KEY
OPENAI_API_BASE = LLM_BASE_URL
OPENAI_API_KEY = LLM_API_KEY
OPENAI_MODEL = 'gpt-4o'
logger = logging.getLogger(__name__)

@dataclass
class IndexInfo:
    """    """
    index_name: str
    relation: str
    attribute: str
    index_type: str
    data_type: str
    is_persistent: bool
    stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationMeta:
    """     """
    name: str
    attributes: List[Tuple[str, str]]
    cardinality: int = 0
    tuple_size: float = 0.0

@dataclass
class IndexRewriteCandidate:
    """      """
    original_query: str
    rewritten_query: str
    rewrite_type: str
    index_used: str
    description: str
    expected_speedup: str = ''

@dataclass
class CostEstimate:
    """      """
    query: str
    estimated_cost_ms: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ''
    has_index: bool = False
    index_details: str = ''
    rank: int = 0

@dataclass
class DBAExperienceEntry:
    """DBA       
    
       DBA                 
       LLM                     
    """
    rule_id: str
    category: str
    description: str
    evidence_queries: List[str] = field(default_factory=list)
    confidence: float = 0.5
    frequency: int = 1
    source: str = 'seed_manual'
    timestamp: str = ''

class IndexKB:
    """
    SECONDO      
    
          :
    1.             
    2.               
    3.      ->            
    """
    TYPE_INDEX_MAP = {'int': ['btree', 'hash'], 'real': ['btree', 'hash'], 'string': ['btree', 'hash'], 'bool': ['btree'], 'point': ['rtree', 'xtree', 'mtree'], 'line': ['rtree', 'xtree', 'mtree'], 'region': ['rtree', 'xtree'], 'points': ['rtree'], 'rect': ['rtree', 'xtree'], 'upoint': ['rtree3', 'tbtree', 'seti', 'ugrid'], 'mpoint': ['rtree3', 'seti', 'ugrid'], 'instant': ['btree'], 'periods': []}
    INDEX_CREATE_SYNTAX = {'btree': {'create': '{rel} createbtree[{attr}]', 'create_let': 'let {idx_name} = {rel} createbtree[{attr}]', 'lookup_exact': '{idx_name} {rel} exactmatch[{val}]', 'lookup_range': '{idx_name} {rel} range[{lo}, {hi}]', 'lookup_exactS': '{idx_name} exactmatchS[{val}] {rel} gettuples'}, 'hash': {'create': '{rel} createhash[{attr}]', 'create_let': 'let {idx_name} = {rel} createhash[{attr}]', 'lookup_exact': '{idx_name} {rel} exactmatch[{val}]', 'lookup_exactS': '{idx_name} exactmatchS[{val}] {rel} gettuples', 'note': 'Hash index: O(1) exact match, no range queries. Best for high-cardinality equality lookups.'}, 'btree2': {'create': '{rel} createbtree2[{attr}]', 'create_let': 'let {idx_name} = {rel} feed addid createbtree2[{attr}]', 'lookup_exact': '{idx_name} {rel} exactmatch[{val}]', 'lookup_exactS': '{idx_name} exactmatchS[{val}] {rel} gettuples', 'lookup_range': '{idx_name} {rel} range[{lo}, {hi}]', 'lookup_rangeS': '{idx_name} rangeS[{lo}, {hi}] {rel} gettuples', 'lookup_leftrange': '{idx_name} {rel} leftrange[{val}]', 'lookup_rightrange': '{idx_name} {rel} rightrange[{val}]', 'note': 'BTree2: enhanced B-tree with configurable fill degree and caching.'}, 'rtree': {'create': '{rel} creatertree[{attr}]', 'create_let': 'let {idx_name} = {rel} creatertree[{attr}]', 'windowintersects': '{idx_name} {rel} windowintersects[bbox({query_obj})]', 'windowintersectsS': '{idx_name} windowintersectsS[bbox({query_obj})] {rel} gettuples', 'distancescan': '{idx_name} {rel} distancescan[{query_obj}, {k}]', 'distancescan2': '{idx_name} {rel} distancescan2[{query_obj}, {k}]'}, 'rtree3': {'create': '{rel} feed extend[Tid: tupleid(.)] sortby[{attr}] bulkloadrtree[{attr}]', 'create_let': 'let {idx_name} = {rel} feed extend[Tid: tupleid(.)] sortby[{attr}] bulkloadrtree[{attr}]', 'distancescan4': '{idx_name} {rel} distancescan4[{query_obj}, {instant}, {k}]', 'windowintersects': '{idx_name} {rel} windowintersects[{bbox}]'}, 'tbtree': {'create': '{rel} createtbtree[{id_attr}, {attr}]', 'create_let': 'let {idx_name} = {rel} createtbtree[{id_attr}, {attr}]', 'bulkload': '{rel} feed addid bulkloadtbtree[{id_attr}, {attr}, TID]', 'windowintersects': '{idx_name} {rel} windowintersects[{bbox}]', 'windowintersectsS': '{idx_name} windowintersectsS[{bbox}] {rel} gettuples', 'note': 'TB-tree: trajectory-bundle tree for upoint indexed by (trajectory_id, upoint). Optimizes temporal+spatial window queries on moving objects.'}, 'xtree': {'create': '{rel} creatextree[{attr}]', 'create_let': 'let {idx_name} = {rel} creatextree[{attr}]', 'rangesearch': '{idx_name} {rel} rangesearch[{hpoint}, {radius}]', 'nnsearch': '{idx_name} {rel} nnsearch[{hpoint}, {k}]', 'nnscan': '{idx_name} {rel} nnscan[{hpoint}]', 'windowintersects': '{idx_name} {rel} windowintersects[{hrect}]', 'note': 'X-tree: extended R-tree for high-dimensional data. Uses hpoint/hrect types. Supports NN search and range search.'}, 'mtree': {'create': '{rel} createmtree[{attr}]', 'create_let': 'let {idx_name} = {rel} createmtree[{attr}]', 'nnsearch': '{idx_name} {rel} nnsearch[{query_obj}, {k}]', 'rangesearch': '{idx_name} {rel} rangesearch[{query_obj}, {radius}]', 'note': 'M-tree: metric tree for similarity search using arbitrary distance functions. Supports NN and range queries on any metric space (string, point, etc.).'}, 'seti': {'create': 'createSETI({bounding_rect}, {grid_cells})', 'create_let': 'let {idx_name} = createSETI({bounding_rect}, {grid_cells})', 'intersectsWindow': 'intersectsWindow({idx_name}, {rect}, {t_start}, {t_end})', 'insideWindow': 'insideWindow({idx_name}, {rect}, {t_start}, {t_end})', 'getTrajectory': 'getTrajectory({idx_name}, {obj_id})', 'insertStream': '{stream} insertStream[{idx_name}, {upload_attr}]', 'note': 'SETI: scalable and efficient trajectory index. Grid-based spatial partitioning with temporal ordering. For moving objects (upload units   upoint stream).'}, 'ugrid': {'create': 'createUGrid({bounding_rect}, {grid_cells})', 'create_let': 'let {idx_name} = createUGrid({bounding_rect}, {grid_cells})', 'intersectsWindow': 'intersectsWindow({idx_name}, {rect}, {t_start}, {t_end})', 'insideWindow': 'insideWindow({idx_name}, {rect}, {t_start}, {t_end})', 'getTrajectory': 'getTrajectory({idx_name}, {obj_id})', 'note': 'UGrid: uniform grid index for moving objects. Similar to SETI but with uniform spatial partitioning. Good for uniformly distributed data.'}, 'spatialjoin': {'operator': '{stream1} {stream2} spatialjoin[{attr1}, {attr2}]', 'note': 'spatialjoin: plane-sweep spatial join on two streams with spatial attributes. Replaces symmjoin with bbox test. Cost   O(n+m+k) vs symmjoin O(n*m). Requires spatial/rect attributes.'}, 'cdacspatialjoin': {'operator': '{stream1} {stream2} cdacspatialjoin[{attr1}, {attr2}]', 'note': 'CDACSpatialJoin: cache-aware divide-and-conquer spatial join. Optimized plane-sweep for large datasets. Works with tuple streams or tblock streams.'}, 'mmrtree_join': {'realJoinMMRTree': '{stream1} {stream2} realJoinMMRTree[{attr1}, {attr2}, {min_entries}, {max_entries}]', 'joinMMRTreeIt': '{stream1} {stream2} joinMMRTreeIt[{attr1}, {attr2}, {min_entries}, {max_entries}]', 'itSpatialJoin': '{stream1} {stream2} itSpatialJoin[{attr1}, {attr2}, {min_entries}, {max_entries}, {buf_size}]', 'note': 'MMRTree joins: main-memory R-tree based spatial joins. Build in-memory R-tree on one stream, probe with other. Efficient for medium-sized datasets that fit in memory.'}}
    _BUILTIN_INDEXES = [IndexInfo('UnitTrains_Line_btree', 'UnitTrains', 'Line', 'btree', 'int', True, {'noentries': 51544, 'nouniquekeys': 9}), IndexInfo('UnitTrainsrtree', 'UnitTrains', 'UTrip', 'rtree3', 'upoint', True, {'entries': 51544}), IndexInfo('strassen_GeoData_rtree', 'strassen', 'GeoData', 'rtree', 'line', True, {'entries': 3212}), IndexInfo('plz_Ort_btree', 'plz', 'Ort', 'btree', 'string', True, {'noentries': 41267, 'nouniquekeys': 16788}), IndexInfo('plz_PLZ_btree', 'plz', 'PLZ', 'btree', 'int', True, {'noentries': 41267, 'nouniquekeys': 28922}), IndexInfo('otestrelaux_Box_rtree', 'otestrelaux', 'Box', 'rtree', 'rect', True, {'entries': 74})]
    _RELATION_META = {'Trains': RelationMeta('Trains', [('Id', 'int'), ('Line', 'int'), ('Up', 'bool'), ('Trip', 'mpoint')], cardinality=562, tuple_size=5000.0), 'UnitTrains': RelationMeta('UnitTrains', [('Id', 'int'), ('Line', 'int'), ('Up', 'bool'), ('UTrip', 'upoint'), ('No', 'int')], cardinality=51544, tuple_size=200.0), 'Stations': RelationMeta('Stations', [('Name', 'string'), ('GeoData', 'point'), ('Pos', 'real')], cardinality=50, tuple_size=100.0), 'Kinos': RelationMeta('Kinos', [('Name', 'string'), ('Strasse', 'string'), ('GeoData', 'point')], cardinality=174, tuple_size=100.0), 'strassen': RelationMeta('strassen', [('Name', 'string'), ('Typ', 'string'), ('GeoData', 'line')], cardinality=3212, tuple_size=300.0), 'plz': RelationMeta('plz', [('PLZ', 'int'), ('Ort', 'string')], cardinality=41267, tuple_size=30.0), 'Kneipen': RelationMeta('Kneipen', [('Name', 'string'), ('Strasse', 'string'), ('GeoData', 'point')], cardinality=300, tuple_size=100.0), 'Restaurants': RelationMeta('Restaurants', [('Name', 'string'), ('Art', 'string'), ('Strasse', 'string'), ('GeoData', 'point')], cardinality=200, tuple_size=120.0), 'Flaechen': RelationMeta('Flaechen', [('Name', 'string'), ('GeoData', 'region')], cardinality=100, tuple_size=500.0), 'UTOrdered': RelationMeta('UTOrdered', [('Id', 'int'), ('Line', 'int'), ('Up', 'bool'), ('UTrip', 'upoint'), ('No', 'int')], cardinality=51544, tuple_size=200.0)}
    COST_CONSTANTS = {'feed': {'pertuple': 0.00194, 'perbyte': 1.96e-05}, 'feedproject': {'msPerTuple': 0.002, 'msPerByte': 3.6e-05, 'msPerAttr': 0.0018}, 'consume': {'msPerTuple': 0.024, 'msPerByteCore': 0.0003, 'msPerByteLOB': 0.001338}, 'filter': {'msPerTuple': 0.01}, 'extend': {'msPerTuple': 0.0012, 'msPerTupleAndAttribute': 0.00085}, 'project': {'msPerTuple': 0.00073, 'msPerTupleAndAttr': 0.0004}, 'rename': {'pertuple': 0.000367}, 'rdup': {'msPerTuple': 0.01, 'msPerComparison': 0.1, 'defaultSelectivity': 0.9}, 'product': {'msPerByteRightOnBufferOverflow': 0.0003, 'msPerByteTotalOutput': 4.2e-05}, 'sortby': {'msPerByteInputSorted': 0.000396, 'msPerByteOutput': 0.000194, 'msPerByteWrittenToDisk': 4e-05}, 'btreelookup': {'msPerSearch': 0.15, 'msPerResultTuple': 0.018}, 'hashlookup': {'msPerSearch': 0.05, 'msPerResultTuple': 0.012, 'note': 'Hash O(1) amortized, faster than B-tree for exact match'}, 'createbtree': {'msPerTuple': 0.12, 'note': 'from createbtreeTC(0.12) in operators.pl'}, 'windowintersects': {'msPerTuple': 0.00194, 'msPerByte': 1.06e-05}, 'distancescan': {'msPerResult_logCard': 0.25, 'note': 'from distancescanTC(0.25); Cost = C1 + 0.25 * ResCard * log(ResCard+1)'}, 'ksmallest': {'msPerTupleInput': 0.05, 'msPerTupleOutput': 0.23, 'note': 'from ksmallestTC(0.05, 0.23) in operators.pl'}, 'hashjoin': {'msPerProbeTuple': 0.023, 'msPerRightTuple': 0.0067, 'msPerResultTuple': 0.0025, 'byteBufferSizeRatioY': 0.75}, 'mergejoin': {'msPerTupleRead': 0.0008077, 'msPerResultTuple': 0.0012058, 'msPerResultAttribute': 0.0001072}, 'sortmergejoin': {'msPerByteReadMerge': 0.0001738, 'msPerByteReadSort': 0.00043}, 'symmjoin': {'msPerTuplePair': 0.2, 'note': '  QUADRATIC: Cost = Card1   Card2   0.2'}, 'loopjoin': {'note': 'Cost = CardX   CostY(inner per tuple). Very efficient with index on inner.'}, 'spatialjoin': {'msPerInputTuple': 20.0, 'msPerResultTuple': 0.7, 'note': 'from spatialjoinTC(20.0, 0.7); Cost = CostX + CostY + 20*(CardX+CardY) + 0.7*ResultCard. Plane-sweep based, much better than symmjoin for spatial.'}, 'general': {'maxMemBytePerOperator': 4194304}, 'gettuples': {'note': 'Cost = CardResult   (random_disk_seek + tuple_read). Adds disk I/O overhead per tuple.'}}

    def __init__(self):
        """        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _load_cache(self):
        """    """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _save_cache(self):
        """    """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _has_index(self, name: str) -> bool:
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_indexes_for_relation(self, relation: str) -> List[IndexInfo]:
        """          """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_indexes_for_attribute(self, relation: str, attribute: str) -> List[IndexInfo]:
        """           """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_relation_meta(self, relation: str) -> Optional[RelationMeta]:
        """       """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_applicable_index_types(self, data_type: str) -> List[str]:
        """              """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def suggest_index(self, relation: str, attribute: str, data_type: str) -> Optional[Dict]:
        """      """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_all_indexes(self) -> List[IndexInfo]:
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_cost_constants(self) -> Dict:
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def format_cost_formulas_for_llm(self) -> str:
        """    SECONDO cost     LLM     """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def format_db_meta_for_llm(self) -> str:
        """           LLM     """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def format_index_syntax_for_llm(self) -> str:
        """         LLM     """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

class IndexEnumerator:
    """
                                    
    
    11       (Pattern 1-11):
    P1:       +    filter   R-tree windowintersects + filter
    P2:       + ID/equality filter   B-tree exactmatch
    P3:  sort + head(k)   distancescan (NN via R-tree)
    P4:  symmjoin   loopjoin + R-tree index
    P5:  inline creatertree            
    P6:  equality filter on string   Hash exactmatch
    P7:  symmjoin with spatial predicate   spatialjoin operator (plane-sweep)
    P8:  temporal/spatial filter on upoint   TB-tree windowintersects
    P9:  distancescan variant selection (distancescan/2/4 based on index type)
    P10: knearestfilter/manual knearest   distancescan via R-tree
    P11: symmjoin   realJoinMMRTree (main-memory R-tree join)
    """

    def __init__(self, index_kb: IndexKB):
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def analyze_and_enumerate(self, query: str, query_type: str, nlq: str='') -> List[IndexRewriteCandidate]:
        """
                     
        
        Args:
            query:        
            query_type:     
            nlq:       
        
        Returns:
                            
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _reuse_existing_index(self, query: str) -> List[IndexRewriteCandidate]:
        """   inline creatertree/createbtree               """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_spatial_filter(self, query: str) -> List[IndexRewriteCandidate]:
        """   filter   R-tree windowintersects + filter"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_id_filter(self, query: str) -> List[IndexRewriteCandidate]:
        """ID filter   B-tree exactmatch"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_nn_pattern(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """NN: sort-based   distancescan"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_join_pattern(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """symmjoin   loopjoin + R-tree"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_hash_lookup(self, query: str) -> List[IndexRewriteCandidate]:
        """P6: Equality filter on string   Hash exactmatch (O(1) vs O(n))"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_spatialjoin(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """P7: Replace symmjoin with spatialjoin operator for spatial join predicates"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_tbtree_filter(self, query: str) -> List[IndexRewriteCandidate]:
        """P8: Replace full scan + temporal filter with TB-tree windowintersects"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_distancescan_variant(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """P9: Select optimal distancescan variant based on available index type"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_knearest_pattern(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """P10: Replace manual knearest pattern with distancescan"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rewrite_mmrtree_join(self, query: str, query_type: str) -> List[IndexRewriteCandidate]:
        """P11: Replace symmjoin with main-memory R-tree join (realJoinMMRTree)"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

class DBAExperienceKB:
    """
    DBA            DBA                 
    
          :
                                                                    
             (Database Knowledge)                             
       -    SECONDO    /     /                         
       -      IndexKB / COST_CONSTANTS                      
                                                                    
            (Experience Knowledge)                           
       -    LLM    DBA                             
       -           JSON                           
       -     :                                                 
         1. index_selection                                   
         2. join_strategy      Join                             
         3. filter_ordering    Filter                           
         4. cost_pattern                                      
         5. general                                           
                                                                    
    
       :
    1.        seed rules (    ) +             
    2.    LLM             prompt      
    3.    LLM              insights
    4.    LLM               
    5.           (  +1,      )
    """
    _instance = None
    _SEED_EXPERIENCES = [DBAExperienceEntry(rule_id='seed_idx_01', category='index_selection', description='For UnitTrains (51544 rows), always prefer the persistent UnitTrainsrtree (rtree3 on UTrip) over inline creatertree. Index build cost alone   6185ms (51544   0.12ms/tuple), which exceeds typical query execution time.', confidence=0.95, source='seed_manual'), DBAExperienceEntry(rule_id='seed_idx_02', category='index_selection', description='Hash index is ~30% faster than B-tree for exact equality lookups on high-cardinality columns (e.g., plz.Ort with 16788 unique keys out of 41267). But hash CANNOT support range queries   use B-tree if range queries are needed.', confidence=0.9, source='seed_manual'), DBAExperienceEntry(rule_id='seed_idx_03', category='index_selection', description='For small relations (<100 rows, e.g., Stations=50, otestrelaux=74), index access rarely outperforms full scan due to I/O overhead of index lookup + gettuples. A full scan with filter is usually faster.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_idx_04', category='index_selection', description='R-tree3 (3D) is preferred over standard R-tree for upoint/mpoint attributes because it indexes the temporal dimension. distancescan4 on rtree3 supports spatio-temporal nearest-neighbor with an instant parameter.', confidence=0.9, source='seed_manual'), DBAExperienceEntry(rule_id='seed_idx_05', category='index_selection', description='TB-tree groups unit records by trajectory ID, making it optimal for trajectory-level queries (e.g., "find all units of train X in area Y"). For pure spatial window queries without trajectory grouping, R-tree is better.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_join_01', category='join_strategy', description='symmjoin is O(n m)   NEVER use it when n m > 10000. For spatial predicates, prefer spatialjoin (plane-sweep, O(n+m+k)). For indexed inner relations, prefer loopjoin + R-tree windowintersects.', confidence=0.95, source='seed_manual'), DBAExperienceEntry(rule_id='seed_join_02', category='join_strategy', description='When both join inputs have bbox-able spatial attributes and total rows < 50000, realJoinMMRTree (in-memory R-tree join) can outperform spatialjoin by avoiding sort-based plane sweep. Use min=4, max=8 as default R-tree parameters.', confidence=0.75, source='seed_manual'), DBAExperienceEntry(rule_id='seed_join_03', category='join_strategy', description='For joining Trains/UnitTrains with Stations (spatial nearest), loopjoin with the smaller relation (Stations, 50 rows) as outer and R-tree index lookup on the larger relation as inner is the most efficient pattern.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_filter_01', category='filter_ordering', description='Always push selective filters (especially equality predicates like .Id = 5) before expensive operations (extend, project, join). A filter with 1% selectivity on 50000 rows reduces downstream processing to just 500 rows.', confidence=0.95, source='seed_manual'), DBAExperienceEntry(rule_id='seed_filter_02', category='filter_ordering', description='Temporal filters (deftime intersects period) are typically cheaper to evaluate than spatial filters (passes/inside region) for mpoint/upoint. When both are needed, apply temporal filter first to reduce candidate set before spatial test.', confidence=0.8, source='seed_manual'), DBAExperienceEntry(rule_id='seed_filter_03', category='filter_ordering', description='For trajectory queries: filter by Id first (B-tree exactmatch   ~5700 rows from 51544), then by temporal range, then by spatial predicate. Each stage should reduce cardinality by at least 50% to be worthwhile.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_cost_01', category='cost_pattern', description='For nearest-neighbor on strassen (3212 rows), distancescan via strassen_GeoData_rtree costs ~O(k log(3212))   30-50ms for k=5, vs full scan + sortby + head   O(3212 log(3212))   300ms. ~6x speedup.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_cost_02', category='cost_pattern', description='For UnitTrains temporal window queries, UnitTrainsrtree windowintersects reduces scan from 51544 rows to ~1000-5000 (depending on time window), saving 80-95% of tuples processed. Expected cost: 50-200ms vs 1000-2000ms.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_cost_03', category='cost_pattern', description='consume cost is dominated by LOB (large object) size for mpoint attributes. For Trains (mpoint Trip   5000B), consume   562   (0.024 + 5000 0.001338)   3775ms. Reducing cardinality via filter/index BEFORE consume is critical.', confidence=0.8, source='seed_manual'), DBAExperienceEntry(rule_id='seed_general_01', category='general', description='Inline index creation (creatertree/createbtree) within a query is expensive and should only be used when: (1) no persistent index exists AND (2) the index will be used in multiple subqueries within the same plan. Single-use indexes are rarely cost-effective.', confidence=0.9, source='seed_manual'), DBAExperienceEntry(rule_id='seed_general_02', category='general', description='For SECONDO "let" variable patterns (e.g., let var = Rel feed extract[Trip]), the result is a single value, not a stream. Queries referencing "var" later should not apply stream operators (feed, filter) on it directly.', confidence=0.85, source='seed_manual'), DBAExperienceEntry(rule_id='seed_general_03', category='general', description='When a query computes trajectory similarity (e.g., lcss, frechetdistance), the dominant cost is the similarity computation itself, not scan/filter. Index-based pre-filtering (spatial/temporal window) is essential to reduce the number of expensive similarity computations.', confidence=0.8, source='seed_manual')]
    CATEGORIES = ['index_selection', 'join_strategy', 'filter_ordering', 'cost_pattern', 'general']
    _CATEGORY_KEYWORDS = {'index_selection': ['index', 'rtree', 'btree', 'hash', 'tbtree', 'seti', 'ugrid', 'creatertree', 'createbtree', 'createhash', 'exactmatch', 'windowintersects', 'distancescan', 'persistent'], 'join_strategy': ['join', 'symmjoin', 'loopjoin', 'hashjoin', 'mergejoin', 'spatialjoin', 'cdacspatialjoin', 'mmrtree', 'product'], 'filter_ordering': ['filter', 'extend', 'project', 'sortby', 'head', 'rename', 'selectivity', 'predicate', 'push'], 'cost_pattern': ['cost', 'cardinality', 'tuplesize', 'performance', 'speedup', 'expensive', 'millisecond', 'scan'], 'general': []}

    def __init__(self):
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    @classmethod
    def get_instance(cls) -> 'DBAExperienceKB':
        """    """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _load(self):
        """                 seed rules"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def save(self):
        """          JSON   """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_all(self) -> List[DBAExperienceEntry]:
        """      """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_by_category(self, category: str) -> List[DBAExperienceEntry]:
        """       """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_relevant_experiences(self, query: str, query_type: str, limit: int=12) -> List[DBAExperienceEntry]:
        """
                      
        
                +       +    /    
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def format_for_llm(self, query: str, query_type: str) -> str:
        """
                   LLM        
        
            system prompt   user prompt  
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def add_experience(self, category: str, description: str, evidence_query: str='', confidence: float=0.5, source: str='llm_discovered') -> Optional[str]:
        """
                  
        
        Returns:
                 rule_id         None
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def reinforce(self, rule_id: str, evidence_query: str=''):
        """       (  +1,      )"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def learn_from_llm_response(self, llm_result: Dict, nlq: str, query_type: str) -> int:
        """
          LLM             DBA   
        
        LLM        'dba_insights'   :
        [
            {
                "category": "index_selection",
                "insight": "description of the optimization insight"
            },
            ...
        ]
        
        Returns:
                  
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def get_stats(self) -> Dict:
        """           """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

class LLMCostModel:
    """
    LLM-based Cost Model
    
      SECONDO   cost                   
      GPT-4o                    
    
        :
    -                     
    -          :            filter    join   
    - LLM    cost               
    """

    def __init__(self, index_kb: IndexKB, api_key: str=None, experience_kb: DBAExperienceKB=None):
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _get_client(self):
        """   OpenAI client"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def optimize_candidates(self, nlq: str, query_type: str, candidates: List[str], test_id: int=0) -> Dict:
        """
                      +      +   
        
        Args:
            nlq:       
            query_type:     
            candidates:          
            test_id:   ID
            
        Returns:
            {
                'ranked_plans': [{query, rank, cost_estimate, has_index, reasoning}, ...],
                'best_plan': str,
                'index_suggestions': [...],
                'analysis': str
            }
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _llm_cost_evaluate(self, nlq: str, query_type: str, variants: List[Dict], max_retries: int=3) -> List[Dict]:
        """
           GPT-4o                
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _rule_based_ranking(self, variants: List[Dict]) -> List[Dict]:
        """   fallback    (   P1-P11     )"""
        raise NotImplementedError('Implementation withheld until paper acceptance.')

class QueryOptimizationPipeline:
    """
             
    
      :             
      :    NLQ          +     
    
       DBA                    
    """

    def __init__(self, api_key: str=None):
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def optimize_all(self, survivors: List[Dict], output_path: str=None, resume: bool=True) -> Dict:
        """
                   
        
        Args:
            survivors: [{test_id, path_id, nlq, query_type, executables: [...]}, ...]
            output_path:       
            resume:           
            
        Returns:
                  
        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _save_partial(self, results: List[Dict], total: int, output_path: str):
        """        """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """      """
        raise NotImplementedError('Implementation withheld until paper acceptance.')

def main():
    """     """
    raise NotImplementedError('Implementation withheld until paper acceptance.')
if __name__ == '__main__':
    main()
