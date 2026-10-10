
"""
SECONDO       
 Algebras .examples                   

     
1.   .examples          
2.           TF-IDF 
3.   NLQ               
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle





from nalplan.config import ALGEBRAS_PATH
ALGEBRAS_PATH = str(ALGEBRAS_PATH)
from nalplan.config import CACHE_ROOT
CACHE_DIR = str(CACHE_ROOT)


CORE_ALGEBRAS = [
    'Temporal',           
    'TemporalExt',        
    'TemporalLifted',     
    'Spatial',            
    'SpatialJoin',        
    'NearestNeighbor',    
    'TrajectorySimilarity', 
    'Relation',           
    'ExtRelation',        
    'Stream',             
    'Standard',           
    'MovingRegion',       
    'RTree',              
    'BTree',              
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
        
        
        self.input_types, self.output_type = self._parse_signature(signature)
        
        
        self.description = self._generate_description()
    
    def _parse_signature(self, signature: str) -> Tuple[List[str], str]:
        """       """
        try:
            if '->' in signature:
                parts = signature.split('->')
                input_part = parts[0].strip()
                output_part = parts[1].strip() if len(parts) > 1 else ''
                
                input_types = []
                input_part = input_part.strip('() ')
                if ' x ' in input_part:
                    input_types = [t.strip() for t in input_part.split(' x ')]
                else:
                    input_types = [t.strip() for t in input_part.split() if t.strip()]
                
                return input_types, output_part
            else:
                return [], signature
        except:
            return [], signature
    
    def _generate_description(self) -> str:
        """          """
        desc_parts = [
            f"Operator {self.name}",
            f"Signature {self.signature}",
            f"Example {self.example}",
        ]
        
        if self.input_types:
            desc_parts.append(f"Input types {' '.join(self.input_types)}")
        if self.output_type:
            desc_parts.append(f"Output type {self.output_type}")
        
        return ' '.join(desc_parts)
    
    def to_dict(self) -> Dict:
        """     """
        return {
            'name': self.name,
            'number': self.number,
            'signature': self.signature,
            'example': self.example,
            'result': self.result,
            'algebra': self.algebra,
            'input_types': self.input_types,
            'output_type': self.output_type,
            'description': self.description
        }
    
    def __repr__(self):
        return f"Operator({self.name}, {self.algebra})"






class OperatorParser:
    """  .examples          """
    
    def __init__(self, algebras_path: str = ALGEBRAS_PATH):
        self.algebras_path = algebras_path
        self.operators = []
        self.operators_by_name = defaultdict(list)
        self.operators_by_algebra = defaultdict(list)
    
    def parse_all(self, core_only: bool = True) -> List[Operator]:
        """    .examples  """
        self.operators = []
        
        
        examples_files = []
        for root, dirs, files in os.walk(self.algebras_path):
            for file in files:
                if file.endswith('.examples'):
                    algebra_name = file.replace('.examples', '')
                    if core_only and algebra_name not in CORE_ALGEBRAS:
                        continue
                    examples_files.append((os.path.join(root, file), algebra_name))
        
        print(f"[Parser]    {len(examples_files)}  .examples  ")
        
        
        for file_path, algebra_name in examples_files:
            operators = self._parse_file(file_path, algebra_name)
            self.operators.extend(operators)
            print(f"  - {algebra_name}: {len(operators)}     ")
        
        
        for op in self.operators:
            self.operators_by_name[op.name].append(op)
            self.operators_by_algebra[op.algebra].append(op)
        
        print(f"[Parser]     {len(self.operators)}     ")
        print(f"[Parser]        : {len(self.operators_by_name)}  ")
        
        return self.operators
    
    def _parse_file(self, file_path: str, algebra_name: str) -> List[Operator]:
        """    .examples  """
        operators = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            blocks = self._split_into_blocks(content)
            
            for block in blocks:
                op = self._parse_block(block, algebra_name)
                if op:
                    operators.append(op)
        
        except Exception as e:
            print(f"[Parser]        {file_path}: {e}")
        
        return operators
    
    def _split_into_blocks(self, content: str) -> List[str]:
        """            """
        pattern = r'(?=Operator\s*:\s*\w)'
        blocks = re.split(pattern, content)
        return [b.strip() for b in blocks if b.strip() and 'Operator' in b]
    
    def _parse_block(self, block: str, algebra_name: str) -> Optional[Operator]:
        """        """
        try:
            name_match = re.search(r'Operator\s*:\s*(\S+)', block)
            number_match = re.search(r'Number\s*:\s*(\d+)', block)
            sig_match = re.search(r'Signature\s*:\s*(.+?)(?=\n|Example|$)', block, re.DOTALL)
            example_match = re.search(r'Example\s*:\s*(.+?)(?=\n.*?:|Result|$)', block, re.DOTALL)
            result_match = re.search(r'Result\s*:\s*(.+?)(?=\n\n|\n.*?Operator|$)', block, re.DOTALL)
            
            if not name_match:
                return None
            
            name = name_match.group(1).strip()
            number = int(number_match.group(1)) if number_match else 1
            signature = sig_match.group(1).strip() if sig_match else ''
            example = example_match.group(1).strip() if example_match else ''
            result = result_match.group(1).strip() if result_match else ''
            
            return Operator(
                name=name, number=number, signature=signature,
                example=example, result=result, algebra=algebra_name
            )
        
        except:
            return None






class OperatorVectorizer:
    """        -   TF-IDF"""
    
    def __init__(self):
        self.operators = []
        self.operator_vectors = None
        self.vocabulary = {}
        self.idf = None
        self.vector_dim = 0
    
    def fit(self, operators: List[Operator]):
        """         """
        self.operators = operators
        self._build_vocabulary()
        self._compute_tfidf_vectors()
        
        print(f"[Vectorizer]      : {len(operators)}     ")
        print(f"[Vectorizer]      : {len(self.vocabulary)}")
    
    def _build_vocabulary(self):
        """     """
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        
        for op in self.operators:
            words = self._tokenize(op.description)
            unique_words = set(words)
            
            for word in words:
                word_counts[word] += 1
            for word in unique_words:
                doc_counts[word] += 1
        
        
        n_docs = len(self.operators)
        filtered_words = [
            word for word, count in word_counts.items()
            if count >= 2 and doc_counts[word] < n_docs * 0.8
        ]
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(filtered_words))}
        self.vector_dim = len(self.vocabulary)
        
        
        self.idf = np.zeros(self.vector_dim)
        for word, idx in self.vocabulary.items():
            self.idf[idx] = np.log(n_docs / (1 + doc_counts[word]))
    
    def _tokenize(self, text: str) -> List[str]:
        """  """
        text = text.lower()
        text = re.sub(r'[^a-z0-9_\s]', ' ', text)
        words = text.split()
        words = [w for w in words if len(w) > 1]
        return words
    
    def _compute_tfidf_vectors(self):
        """  TF-IDF  """
        n_ops = len(self.operators)
        self.operator_vectors = np.zeros((n_ops, self.vector_dim))
        
        for i, op in enumerate(self.operators):
            words = self._tokenize(op.description)
            
            tf = defaultdict(int)
            for word in words:
                if word in self.vocabulary:
                    tf[word] += 1
            
            for word, count in tf.items():
                idx = self.vocabulary[word]
                self.operator_vectors[i, idx] = count * self.idf[idx]
            
            norm = np.linalg.norm(self.operator_vectors[i])
            if norm > 0:
                self.operator_vectors[i] /= norm
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """      """
        words = self._tokenize(query)
        
        vector = np.zeros(self.vector_dim)
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] = self.idf[idx]
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Operator, float]]:
        """            """
        query_vector = self.vectorize_query(query)
        similarities = np.dot(self.operator_vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.operators[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: str):
        """       """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'operators': [op.to_dict() for op in self.operators],
            'vocabulary': self.vocabulary,
            'idf': self.idf.tolist(),
            'operator_vectors': self.operator_vectors.tolist(),
            'vector_dim': self.vector_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[Vectorizer]       : {path}")
    
    def load(self, path: str) -> bool:
        """       """
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.operators = []
            for op_dict in data['operators']:
                op = Operator(
                    name=op_dict['name'], number=op_dict['number'],
                    signature=op_dict['signature'], example=op_dict['example'],
                    result=op_dict['result'], algebra=op_dict['algebra']
                )
                self.operators.append(op)
            
            self.vocabulary = data['vocabulary']
            self.idf = np.array(data['idf'])
            self.operator_vectors = np.array(data['operator_vectors'])
            self.vector_dim = data['vector_dim']
            
            print(f"[Vectorizer]            {len(self.operators)}     ")
            return True
        
        except Exception as e:
            print(f"[Vectorizer]       : {e}")
            return False






class OperatorRetriever:
    """
          
        NLQ                
    """
    
    def __init__(self, algebras_path: str = ALGEBRAS_PATH, 
                 cache_path: str = None, use_cache: bool = True):
        """      """
        if cache_path is None:
            cache_path = os.path.join(CACHE_DIR, 'operator_vectors.pkl')
        
        self.algebras_path = algebras_path
        self.cache_path = cache_path
        self.parser = OperatorParser(algebras_path)
        self.vectorizer = OperatorVectorizer()
        
        
        if use_cache and self.vectorizer.load(cache_path):
            print("[Retriever]           ")
        else:
            print("[Retriever]          ...")
            self._build_index()
    
    def _build_index(self):
        """       """
        operators = self.parser.parse_all(core_only=True)
        self.vectorizer.fit(operators)
        self.vectorizer.save(self.cache_path)
    
    def search_combined(self, nlq: str, exec_templates: List[str], 
                        query_type: str = None, top_k: int = 30) -> List[Tuple[Operator, float]]:
        """
          NLQ               
        
        Args:
            nlq:       
            exec_templates:              
            query_type:             
            top_k:           
            
        Returns:
            (   ,    )    
        """
        
        template_keywords = self._extract_operators_from_templates(exec_templates)
        
        
        type_keywords = self._get_type_keywords(query_type)
        
        
        enhanced_query = self._build_enhanced_query(nlq, template_keywords, type_keywords)
        
        
        results = self.vectorizer.search(enhanced_query, top_k=top_k * 2)
        
        
        results = self._rerank_by_template_operators(results, template_keywords)
        
        
        must_have_ops = self._get_must_have_operators(query_type)
        results = self._merge_must_have_operators(
            results, 
            must_have_ops, 
            type_keywords=type_keywords,
            template_ops=template_keywords
        )
        
        
        seen_names = set()
        filtered_results = []
        
        for op, score in results:
            if op.name not in seen_names:
                seen_names.add(op.name)
                filtered_results.append((op, score))
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def _get_must_have_operators(self, query_type: str) -> List[str]:
        """                """
        must_have = {
            'Time interval query': [
                'atinstant', 'atperiods', 'deftime', 'trajectory', 'val', 'present'
            ],
            'Moving objects range query': [
                'passes', 'inside', 'at', 'deftime', 'intersects'
            ],
            'Moving objects nearest neighbor query': [
                'knearest', 'knearestfilter', 'distancescan4', 'distance', 'head', 'sortby',
                'greeceknearest', 'chinaknearest', 'bulkloadrtree', 'deftime', 'intersects'
            ],
            'Moving objects join query': [
                'symmjoin', 'loopjoin', 'windowintersects', 'intersects', 'overlaps',
                'trajectory', 'atperiods'
            ],
            
            'Spatio-temporal join query': [
                'symmjoin', 'loopjoin', 'windowintersects', 'intersects', 'overlaps',
                'trajectory', 'atperiods'
            ],
            'Trajectory similarity query': [
                'frechetdistance', 'dist_euclidean', 'dist_origin_and_destination',
                'dist_perpendicular', 'dist_angular', 'samplempoint', 'pointseq',
                'knearest', 'distance', 'passes', 'deftime', 'intersects'
            ]
        }
        return must_have.get(query_type, [])
    
    def _merge_must_have_operators(self, results: List[Tuple[Operator, float]], 
                                    must_have_names: List[str],
                                    type_keywords: List[str] = None,
                                    template_ops: List[str] = None) -> List[Tuple[Operator, float]]:
        """
                                 
                       
        
                               
        """
        
        must_have_lower = set(m.lower() for m in must_have_names)
        type_kw_lower = set(k.lower() for k in type_keywords) if type_keywords else set()
        template_lower = set(t.lower() for t in template_ops) if template_ops else set()
        
        
        boosted_results = []
        existing_names = set()
        
        for op, score in results:
            op_lower = op.name.lower()
            
            if op_lower in template_lower:
                score = max(score, 0.20) + 0.05  
            elif op_lower in must_have_lower:
                score = max(score, 0.18) + 0.03  
            elif op_lower in type_kw_lower:
                score = max(score, 0.15) + 0.02  
            
            boosted_results.append((op, score))
            existing_names.add(op_lower)
        
        
        all_ops_to_add = must_have_lower | type_kw_lower | template_lower
        
        for op_name in all_ops_to_add:
            if op_name not in existing_names:
                ops = self.get_operator_info(op_name)
                if ops:
                    
                    if op_name in template_lower:
                        base_score = 0.22
                    elif op_name in must_have_lower:
                        base_score = 0.19
                    else:
                        base_score = 0.16
                    
                    boosted_results.append((ops[0], base_score))
                    existing_names.add(op_name)
        
        
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results
    
    def _extract_operators_from_templates(self, templates: List[str]) -> List[str]:
        """                """
        operator_names = set()
        
        
        operator_patterns = [
            
            r'\b(feed|filter|extend|project|consume|count|rename)\b',
            
            r'\b(atinstant|atperiods|present|deftime|trajectory|val|initial|final)\b',
            
            r'\b(knearest|knearestfilter|distancescan\d*)\b',
            
            r'\b(symmjoin|loopjoin|windowintersects|hashjoin)\b',
            
            r'\b(intersects|passes|inside|at|overlaps|meets|touches)\b',
            
            r'\b(sortby|head|extract|bulkloadrtree|creatertree|sortmergejoin)\b',
            
            r'\b(distance|bbox|frechetdistance)\b',
            
            r'\b(dist_euclidean|dist_origin_and_destination|dist_perpendicular|dist_angular)\b',
            r'\b(dist_lip|dist_sdiagram|lcss|edr|erp)\b',
            r'\b(to_pointseq|samplempoint|sample|genlip|genlip2)\b',
            
            r'\b(pointseq|tpointseq|mpoint|units)\b',
        ]
        
        for template in templates:
            template_lower = template.lower()
            for pattern in operator_patterns:
                matches = re.findall(pattern, template_lower)
                operator_names.update(matches)
        
        return list(operator_names)
    
    def _get_type_keywords(self, query_type: str) -> List[str]:
        """             """
        type_keywords = {
            'Time interval query': [
                'atinstant', 'atperiods', 'deftime', 'trajectory', 'val',
                'inst', 'present', 'initial', 'final', 'units', 'periods',
                'mpoint', 'instant', 'periods'
            ],
            'Moving objects range query': [
                'passes', 'inside', 'intersection', 'at', 'atperiods',
                'present', 'distance', 'trajectory', 'deftime', 'filter',
                'mpoint', 'region', 'point', 'intersects'
            ],
            'Moving objects nearest neighbor query': [
                'distance', 'knearest', 'knearestfilter', 'distancescan',
                'distancescan4', 'closest', 'trajectory', 'bulkloadrtree', 'rtree',
                'sortby', 'head', 'mpoint', 'greeceknearest', 'chinaknearest',
                'knearestvector', 'deftime', 'intersects'
            ],
            'Moving objects join query': [
                'intersection', 'passes', 'overlaps', 'meets', 'inside',
                'trajectory', 'at', 'atinstant', 'deftime', 'symmjoin',
                'loopjoin', 'windowintersects', 'intersects', 'atperiods'
            ],
            
            'Spatio-temporal join query': [
                'intersection', 'passes', 'overlaps', 'meets', 'inside',
                'trajectory', 'at', 'atinstant', 'deftime', 'symmjoin',
                'loopjoin', 'windowintersects', 'intersects', 'atperiods'
            ],
            'Trajectory similarity query': [
                
                'frechetdistance', 'dist_euclidean', 'dist_origin_and_destination',
                'dist_perpendicular', 'dist_angular', 'dist_sdiagram',
                
                'pointseq', 'tpointseq', 'mpoint', 'samplempoint', 'sample',
                
                'distance', 'knearest', 'trajectory',
                
                'atperiods', 'deftime', 'intersects', 'passes'
            ]
        }
        
        return type_keywords.get(query_type, [])
    
    def _build_enhanced_query(self, nlq: str, template_keywords: List[str], 
                              type_keywords: List[str]) -> str:
        """      """
        parts = [nlq]
        
        
        if template_keywords:
            parts.append(' '.join(template_keywords))
        
        
        if type_keywords:
            parts.append(' '.join(type_keywords[:10]))  
        
        return ' '.join(parts)
    
    def _rerank_by_template_operators(self, results: List[Tuple[Operator, float]], 
                                       template_operators: List[str]) -> List[Tuple[Operator, float]]:
        """                  """
        template_ops_lower = set(op.lower() for op in template_operators)
        
        reranked = []
        for op, score in results:
            
            if op.name.lower() in template_ops_lower:
                score = min(score * 1.5, 1.0)
            reranked.append((op, score))
        
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def get_operator_info(self, operator_name: str) -> List[Operator]:
        """            """
        return [op for op in self.vectorizer.operators if op.name.lower() == operator_name.lower()]
    
    def display_results(self, results: List[Tuple[Operator, float]], title: str = "       "):
        """      """
        print(f"\n{'='*60}")
        print(title)
        print("=" * 60)
        
        for i, (op, score) in enumerate(results, 1):
            print(f"\n{i}. {op.name} (Score: {score:.3f})")
            print(f"   Algebra: {op.algebra}")
            print(f"   Signature: {op.signature}")
            example_display = op.example[:70] + "..." if len(op.example) > 70 else op.example
            print(f"   Example: {example_display}")






if __name__ == "__main__":
    print("=" * 60)
    print("         ")
    print("=" * 60)
    
    
    retriever = OperatorRetriever(use_cache=True)
    
    
    test_cases = [
        {
            'nlq': "What is the trajectory of train 7 between 6am and 10am?",
            'query_type': "Time interval query",
            'exec_templates': [
                'query <OBJECT_TABLE> feed filter[.Id=<OBJECT_ID>] filter[.Trip present [const periods value (("<DATE>-<TIME>" "<DATE>-<TIME>" TRUE TRUE))]] extend[Stretch: trajectory(.Trip atperiods [const periods value (("<DATE>-<TIME>" "<DATE>-<TIME>" TRUE TRUE))])] project[Stretch] consume;'
            ]
        },
        {
            'nlq': "Which trains passed through Mehringdamm at 8:30am?",
            'query_type': "Moving objects range query",
            'exec_templates': [
                'query <OBJECT_TABLE> feed filter [(deftime(.Trip at (<AUX_TABLE> feed filter [.Name="<LOCATION>"] extract[GeoData])) intersects [const periods value (("<DATE>-<TIME>" "<DATE>-<TIME>" TRUE TRUE))])] consume;'
            ]
        },
        {
            'nlq': "Find the 5 nearest taxis to train 15 between 9am and 11am.",
            'query_type': "Moving objects nearest neighbor query",
            'exec_templates': [
                'let t<OBJECT_ID> = <OBJECT_TABLE> feed filter[.Id = <OBJECT_ID>] extract[Trip]; query <UNIT_TABLE> feed filter [(deftime(.UTrip) intersects [const periods value (("<DATE>-<TIME>" "<DATE>-<TIME>" TRUE TRUE))])] knearest[UTrip, t<OBJECT_ID>, <K>] consume;'
            ]
        },
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"NLQ: {case['nlq']}")
        print(f"  : {case['query_type']}")
        print("=" * 60)
        
        results = retriever.search_combined(
            nlq=case['nlq'],
            exec_templates=case['exec_templates'],
            query_type=case['query_type'],
            top_k=5
        )
        
        retriever.display_results(results, "         ")
