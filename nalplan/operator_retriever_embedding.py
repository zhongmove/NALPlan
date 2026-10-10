from pathlib import Path

"""
  OpenAI text-embedding-3-large       
 TF-IDF        

                      
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle
import time
import requests





from nalplan.config import ALGEBRAS_PATH
ALGEBRAS_PATH = str(ALGEBRAS_PATH)
from nalplan.config import CACHE_ROOT
CACHE_DIR = str(CACHE_ROOT)
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, 'operator_embeddings_openai.pkl')


from nalplan.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
API_BASE_URL = LLM_BASE_URL
API_KEY = LLM_API_KEY
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  


CORE_ALGEBRAS = [
    'Temporal', 'TemporalExt', 'TemporalLifted', 'Spatial', 'SpatialJoin',
    'NearestNeighbor', 'TrajectorySimilarity', 'Relation', 'ExtRelation',
    'Stream', 'Standard', 'MovingRegion', 'RTree', 'BTree',
]






class Operator:
    """    SECONDO   """
    
    def __init__(self, name: str, number: int, signature: str, 
                 example: str, result: str, algebra: str):
        self.name = name
        self.number = number
        self.signature = signature
        self.example = example
        self.result = result
        self.algebra = algebra
        self.description = self._generate_description()
    
    def _generate_description(self) -> str:
        """          """
        desc_parts = [
            f"Operator: {self.name}",
            f"Algebra: {self.algebra}",
            f"Signature: {self.signature}",
            f"Example: {self.example}",
        ]
        return ' | '.join(desc_parts)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'number': self.number,
            'signature': self.signature,
            'example': self.example,
            'result': self.result,
            'algebra': self.algebra,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Operator':
        op = cls(
            name=d['name'], number=d['number'], signature=d['signature'],
            example=d['example'], result=d['result'], algebra=d['algebra']
        )
        return op
    
    def __repr__(self):
        return f"Operator({self.name}, {self.algebra})"






class OpenAIEmbeddingVectorizer:
    """  OpenAI text-embedding-3-large        requests  """
    
    def __init__(self):
        self.api_url = f"{API_BASE_URL}/embeddings"
        self.api_key = API_KEY
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.operators = []
        self.operator_embeddings = None
        self.embedding_dim = EMBEDDING_DIM
    
    def get_embedding(self, text: str) -> np.ndarray:
        """       embedding"""
        try:
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text
            }
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return np.array(result['data'][0]['embedding'])
            else:
                print(f"[Embedding] API   {response.status_code}: {response.text[:200]}")
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            print(f"[Embedding]   embedding  : {e}")
            return np.zeros(self.embedding_dim)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[np.ndarray]:
        """    embeddings"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                payload = {
                    "model": EMBEDDING_MODEL,
                    "input": batch
                }
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    batch_embeddings = [np.array(d['embedding']) for d in result['data']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    print(f"[Embedding]   API   {response.status_code}: {response.text[:200]}")
                    all_embeddings.extend([np.zeros(self.embedding_dim) for _ in batch])
                
                if (i + batch_size) % 100 == 0 and i > 0:
                    print(f"      {min(i + batch_size, len(texts))}/{len(texts)}     ...")
                    time.sleep(0.3)  
                    
            except Exception as e:
                print(f"[Embedding]        (batch {i}): {e}")
                all_embeddings.extend([np.zeros(self.embedding_dim) for _ in batch])
        
        return all_embeddings
    
    def fit(self, operators: List[Operator]):
        """         """
        self.operators = operators
        
        print(f"[Embedding]       {len(operators)}     ...")
        print(f"[Embedding]     : {EMBEDDING_MODEL}")
        print(f"[Embedding] API  : {self.api_url}")
        
        
        descriptions = [op.description for op in operators]
        
        
        start_time = time.time()
        embeddings = self.get_embeddings_batch(descriptions)
        elapsed = time.time() - start_time
        
        self.operator_embeddings = np.array(embeddings)
        
        print(f"[Embedding]          {elapsed:.1f}  ")
        print(f"[Embedding]     : {self.operator_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Operator, float]]:
        """            """
        query_embedding = self.get_embedding(query)
        
        
        similarities = np.dot(self.operator_embeddings, query_embedding)
        norms = np.linalg.norm(self.operator_embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms[norms == 0] = 1  
        similarities = similarities / norms
        
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.operators[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: str):
        """       """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'operators': [op.to_dict() for op in self.operators],
            'embeddings': self.operator_embeddings.tolist(),
            'model': EMBEDDING_MODEL
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[Embedding]       : {path}")
    
    def load(self, path: str) -> bool:
        """       """
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.operators = [Operator.from_dict(d) for d in data['operators']]
            self.operator_embeddings = np.array(data['embeddings'])
            
            print(f"[Embedding]         {len(self.operators)}     ")
            return True
        except Exception as e:
            print(f"[Embedding]     : {e}")
            return False






class EmbeddingOperatorRetriever:
    """
      OpenAI Embedding       
                  
    """
    
    
    TYPE_KEYWORDS = {
        'Time interval query': 'trajectory atperiods atinstant present val deftime temporal time interval period',
        'Moving objects range query': 'passes inside intersects at deftime present range within area region',
        'Moving objects nearest neighbor query': 'knearest distancescan distance nearest neighbor proximity closest head sortby',
        'Moving objects join query': 'everynearerthan overlap join intersects distance meet touch',
        'Trajectory similarity query': 'frechetdistance lcss dist_euclidean dist_origin_and_destination similarity trajectory distance samplempoint compare'
    }
    
    def __init__(self, use_cache: bool = True):
        """      """
        self.vectorizer = OpenAIEmbeddingVectorizer()
        
        
        if use_cache and self.vectorizer.load(EMBEDDING_CACHE_PATH):
            print("[Retriever]      embedding  ")
        else:
            print("[Retriever]       embedding  ...")
            self._build_index()
    
    def _build_index(self):
        """       """
        
        from nalplan.operator_retriever import OperatorParser
        parser = OperatorParser(ALGEBRAS_PATH)
        operators = parser.parse_all(core_only=True)
        
        
        local_operators = []
        for op in operators:
            local_op = Operator(
                name=op.name, number=op.number, signature=op.signature,
                example=op.example, result=op.result, algebra=op.algebra
            )
            local_operators.append(local_op)
        
        self.vectorizer.fit(local_operators)
        self.vectorizer.save(EMBEDDING_CACHE_PATH)
    
    def _extract_operators_from_templates(self, templates: List[str]) -> List[str]:
        """                """
        import re
        operators = set()
        
        
        patterns = [
            r'\b([a-z_][a-z0-9_]*)\s*\(',  
            r'\b([a-z_][a-z0-9_]*)\s*\[',  
            r'\.\s*([a-z_][a-z0-9_]*)',     
        ]
        
        for template in templates:
            template_lower = template.lower()
            for pattern in patterns:
                matches = re.findall(pattern, template_lower)
                for match in matches:
                    
                    if match not in {'select', 'from', 'where', 'and', 'or', 'not', 
                                    'true', 'false', 'null', 'as', 'let', 'in', 'if', 
                                    'then', 'else', 'query', 'the', 'instant', 'periods'}:
                        operators.add(match)
        
        return list(operators)
    
    def _get_type_keywords(self, query_type: str) -> str:
        """            """
        if not query_type:
            return ""
        
        if query_type in self.TYPE_KEYWORDS:
            return self.TYPE_KEYWORDS[query_type]
        
        query_type_lower = query_type.lower()
        for key, value in self.TYPE_KEYWORDS.items():
            if key.lower() == query_type_lower:
                return value
        return ""
    
    def search_combined(self, nlq: str, exec_templates: List[str], 
                        query_type: str = None, top_k: int = 30,
                        method: str = "multi_query") -> List[Tuple[Operator, float]]:
        """
          NLQ               
        
        Args:
            nlq:       
            exec_templates:          
            query_type:     
            top_k:         
            method:      - "simple", "multi_query", "hybrid"
        """
        if method == "simple":
            return self._search_simple(nlq, exec_templates, query_type, top_k)
        elif method == "multi_query":
            return self._search_multi_query(nlq, exec_templates, query_type, top_k)
        elif method == "hybrid":
            return self._search_hybrid(nlq, exec_templates, query_type, top_k)
        else:
            return self._search_multi_query(nlq, exec_templates, query_type, top_k)
    
    def _search_simple(self, nlq: str, exec_templates: List[str], 
                       query_type: str, top_k: int) -> List[Tuple[Operator, float]]:
        """         """
        template_info = ""
        if exec_templates:
            template_sample = exec_templates[0][:500] if exec_templates else ""
            template_info = f" Related executable template: {template_sample}"
        
        type_info = ""
        if query_type:
            type_info = f" Query type: {query_type}"
        
        enhanced_query = f"Find SECONDO database operators for: {nlq}{type_info}{template_info}"
        results = self.vectorizer.search(enhanced_query, top_k=top_k * 2)
        
        return self._deduplicate(results, top_k)
    
    def _search_multi_query(self, nlq: str, exec_templates: List[str], 
                            query_type: str, top_k: int) -> List[Tuple[Operator, float]]:
        """
                
           NLQ             embedding         
        """
        
        WEIGHT_NLQ = 0.4
        WEIGHT_TYPE = 0.3
        WEIGHT_TEMPLATE = 0.3
        
        score_map = {}  
        
        
        nlq_query = f"SECONDO database operators for query: {nlq}"
        nlq_results = self.vectorizer.search(nlq_query, top_k=top_k * 2)
        for op, score in nlq_results:
            key = op.name.lower()
            if key not in score_map:
                score_map[key] = [op, 0.0]
            score_map[key][1] += score * WEIGHT_NLQ
        
        
        if query_type:
            type_keywords = self._get_type_keywords(query_type)
            type_query = f"SECONDO operators for {query_type}: {type_keywords}"
            type_results = self.vectorizer.search(type_query, top_k=top_k * 2)
            for op, score in type_results:
                key = op.name.lower()
                if key not in score_map:
                    score_map[key] = [op, 0.0]
                score_map[key][1] += score * WEIGHT_TYPE
        
        
        if exec_templates:
            template_ops = self._extract_operators_from_templates(exec_templates)
            if template_ops:
                template_query = f"SECONDO operators similar to: {' '.join(template_ops[:20])}"
                template_results = self.vectorizer.search(template_query, top_k=top_k * 2)
                for op, score in template_results:
                    key = op.name.lower()
                    if key not in score_map:
                        score_map[key] = [op, 0.0]
                    score_map[key][1] += score * WEIGHT_TEMPLATE
                
                
                for op_name in template_ops:
                    if op_name in score_map:
                        score_map[op_name][1] += 0.15  
                    else:
                        
                        op_info = self.get_operator_info(op_name)
                        if op_info:
                            score_map[op_name] = [op_info[0], 0.25]
        
        
        sorted_results = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
        return [(op, score) for op, score in sorted_results[:top_k]]
    
    def _search_hybrid(self, nlq: str, exec_templates: List[str], 
                       query_type: str, top_k: int) -> List[Tuple[Operator, float]]:
        """
             TF-IDF          Embedding         
        
           
        1. TF-IDF      (top_k*2)           
        2. Embedding         
        3.     RRF   TF-IDF                 
        """
        from nalplan.operator_retriever import OperatorRetriever as TFIDFRetriever
        
        
        tfidf_retriever = TFIDFRetriever(use_cache=True)
        tfidf_results = tfidf_retriever.search_combined(
            nlq=nlq,
            exec_templates=exec_templates,
            query_type=query_type,
            top_k=top_k * 2  
        )
        
        
        emb_results = self._search_multi_query(nlq, exec_templates, query_type, top_k * 2)
        
        
        score_map = {}
        K = 60  
        TFIDF_WEIGHT = 1.5  
        EMB_WEIGHT = 1.0
        
        
        for rank, (op, original_score) in enumerate(tfidf_results):
            key = op.name.lower()
            if key not in score_map:
                score_map[key] = [op, 0.0]
            score_map[key][1] += TFIDF_WEIGHT * (1.0 / (K + rank + 1))
            
            if original_score > 0.2:
                score_map[key][1] += 0.01
        
        
        for rank, (op, _) in enumerate(emb_results):
            key = op.name.lower()
            if key not in score_map:
                score_map[key] = [op, 0.0]
            score_map[key][1] += EMB_WEIGHT * (1.0 / (K + rank + 1))
        
        
        sorted_results = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
        return [(op, score) for op, score in sorted_results[:top_k]]
    
    def _deduplicate(self, results: List[Tuple[Operator, float]], 
                     top_k: int) -> List[Tuple[Operator, float]]:
        """  """
        seen_names = set()
        filtered_results = []
        
        for op, score in results:
            if op.name not in seen_names:
                seen_names.add(op.name)
                filtered_results.append((op, score))
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def get_operator_info(self, operator_name: str) -> List[Operator]:
        """            """
        return [op for op in self.vectorizer.operators 
                if op.name.lower() == operator_name.lower()]






def compare_retrievers():
    """  TF-IDF Embedding      """
    import pandas as pd
    from nalplan.operator_retriever import OperatorRetriever as TFIDFRetriever
    
    print("=" * 80)
    print("           ")
    print("  : TF-IDF / Embedding-Simple / Embedding-MultiQuery / Embedding-Hybrid")
    print("=" * 80)
    
    
    print("\n1.    TF-IDF   ...")
    tfidf_retriever = TFIDFRetriever(use_cache=True)
    
    print("\n2.    Embedding   ...")
    embedding_retriever = EmbeddingOperatorRetriever(use_cache=True)
    
    
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'benchmark' / 'nalplan_benchmark_examples.csv')
    
    
    test_cases = [
        {
            'nlq': 'Which taxi has a similar trajectory to taxi 30 between 8am and 12pm?',
            'query_type': 'Trajectory similarity query',
            'templates': df[df['type'].str.lower() == 'trajectory similarity query']['template'].tolist()[:5],
            'expected_ops': ['frechetdistance', 'dist_euclidean', 'dist_origin_and_destination', 
                           'knearest', 'intersects', 'deftime', 'passes']
        },
        {
            'nlq': 'Find the 5 nearest taxis to train 15 between 9am and 11am.',
            'query_type': 'Moving objects nearest neighbor query',
            'templates': df[df['type'].str.lower() == 'moving objects nearest neighbor query']['template'].tolist()[:5],
            'expected_ops': ['knearest', 'distancescan4', 'distance', 'head', 'sortby', 'greeceknearest', 'chinanearest', 'deftime', 'intersects', 'bulkloadrtree']
        },
        {
            'nlq': 'Which trains passed through Mehringdamm at 8:30am?',
            'query_type': 'Moving objects range query',
            'templates': df[df['type'].str.lower() == 'moving objects range query']['template'].tolist()[:5],
            'expected_ops': ['passes', 'inside', 'at', 'deftime', 'intersects']
        },
        {
            'nlq': 'What is the trajectory of train 7 between 6am and 10am?',
            'query_type': 'Time interval query',
            'templates': df[df['type'].str.lower() == 'time interval query']['template'].tolist()[:5],
            'expected_ops': ['trajectory', 'atperiods', 'atinstant', 'present', 'val']
        },
        {
            'nlq': 'Which taxis have the same trajectory between 9am and 10am?',
            'query_type': 'Moving objects join query',
            'templates': df[df['type'].str.lower() == 'moving objects join query']['template'].tolist()[:5],
            'expected_ops': ['trajectory', 'intersects', 'atperiods', 'symmjoin', 'loopjoin', 'windowintersects']
        },
    ]
    
    
    summary = {
        'TF-IDF': [],
        'Emb-Simple': [],
        'Emb-MultiQuery': [],
        'Emb-Hybrid': []
    }
    
    print("\n" + "=" * 80)
    print("      ")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"   {i}: {case['query_type']}")
        print(f"NLQ: {case['nlq']}")
        print("=" * 80)
        
        TOP_K = 30  
        
        
        print("\n 1. TF-IDF ")
        tfidf_results = tfidf_retriever.search_combined(
            nlq=case['nlq'],
            exec_templates=case['templates'],
            query_type=case['query_type'],
            top_k=TOP_K
        )
        tfidf_names = [op.name for op, _ in tfidf_results]
        tfidf_covered = set(case['expected_ops']) & set(op.lower() for op in tfidf_names)
        tfidf_rate = len(tfidf_covered)/len(case['expected_ops'])*100
        summary['TF-IDF'].append(tfidf_rate)
        print(f"Top 15: {tfidf_names[:15]}")
        print(f"   : {tfidf_rate:.1f}% ({len(tfidf_covered)}/{len(case['expected_ops'])})")
        
        
        print("\n 2. Embedding-Simple ")
        simple_results = embedding_retriever.search_combined(
            nlq=case['nlq'],
            exec_templates=case['templates'],
            query_type=case['query_type'],
            top_k=TOP_K,
            method="simple"
        )
        simple_names = [op.name for op, _ in simple_results]
        simple_covered = set(case['expected_ops']) & set(op.lower() for op in simple_names)
        simple_rate = len(simple_covered)/len(case['expected_ops'])*100
        summary['Emb-Simple'].append(simple_rate)
        print(f"Top 15: {simple_names[:15]}")
        print(f"   : {simple_rate:.1f}% ({len(simple_covered)}/{len(case['expected_ops'])})")
        
        
        print("\n 3. Embedding-MultiQuery        ")
        multi_results = embedding_retriever.search_combined(
            nlq=case['nlq'],
            exec_templates=case['templates'],
            query_type=case['query_type'],
            top_k=TOP_K,
            method="multi_query"
        )
        multi_names = [op.name for op, _ in multi_results]
        multi_covered = set(case['expected_ops']) & set(op.lower() for op in multi_names)
        multi_rate = len(multi_covered)/len(case['expected_ops'])*100
        summary['Emb-MultiQuery'].append(multi_rate)
        print(f"Top 15: {multi_names[:15]}")
        print(f"   : {multi_rate:.1f}% ({len(multi_covered)}/{len(case['expected_ops'])})")
        
        
        print("\n 4. Embedding-Hybrid TF-IDF + Embedding    ")
        hybrid_results = embedding_retriever.search_combined(
            nlq=case['nlq'],
            exec_templates=case['templates'],
            query_type=case['query_type'],
            top_k=TOP_K,
            method="hybrid"
        )
        hybrid_names = [op.name for op, _ in hybrid_results]
        hybrid_covered = set(case['expected_ops']) & set(op.lower() for op in hybrid_names)
        hybrid_rate = len(hybrid_covered)/len(case['expected_ops'])*100
        summary['Emb-Hybrid'].append(hybrid_rate)
        print(f"Top 15: {hybrid_names[:15]}")
        print(f"   : {hybrid_rate:.1f}% ({len(hybrid_covered)}/{len(case['expected_ops'])})")
    
    
    print("\n" + "=" * 80)
    print("           ")
    print("=" * 80)
    for method, rates in summary.items():
        avg_rate = sum(rates) / len(rates)
        print(f"{method:20s}: {avg_rate:.1f}%  (   : {[f'{r:.0f}%' for r in rates]})")


def simple_test():
    """       embedding API       """
    print("=" * 80)
    print("       Embedding API     ")
    print("=" * 80)
    
    
    print("\n   1     embedding...")
    vectorizer = OpenAIEmbeddingVectorizer()
    test_text = "Find trajectory similarity between two moving objects"
    embedding = vectorizer.get_embedding(test_text)
    
    if np.any(embedding != 0):
        print(f"      embedding   : {len(embedding)}")
        print(f"   5  : {embedding[:5].tolist()}")
    else:
        print("    embedding  ")
        return
    
    
    print("\n   2     embedding 3    ...")
    test_texts = [
        "Operator: trajectory | Algebra: Temporal",
        "Operator: distance | Algebra: Spatial", 
        "Operator: knearest | Algebra: NearestNeighbor"
    ]
    embeddings = vectorizer.get_embeddings_batch(test_texts, batch_size=10)
    
    success_count = sum(1 for e in embeddings if np.any(e != 0))
    print(f"       {success_count}/{len(test_texts)}  embedding")
    
    
    print("\n   3       ...")
    try:
        retriever = EmbeddingOperatorRetriever(use_cache=True)
        print(f"             {len(retriever.vectorizer.operators)}     ")
        
        
        print("\n   4       ...")
        results = retriever.search_combined(
            nlq="Find similar trajectories",
            exec_templates=[],
            query_type="Trajectory similarity query",
            top_k=5
        )
        
        print(f"      {len(results)}     :")
        for op, score in results:
            print(f"  - {op.name} ({op.algebra}): {score:.4f}")
            
        print("\n" + "=" * 80)
        print("                   ")
        print("=" * 80)
        
    except Exception as e:
        print(f"          : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        compare_retrievers()
    else:
        simple_test()
