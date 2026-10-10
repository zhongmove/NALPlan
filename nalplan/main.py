from pathlib import Path

"""
  SECONDO    Text-to-Executable-Language    

     
1. NLQ   SLM.csv             NLQ       
2.         executable_templates.csv               LLM  

   5          
1. Time interval query -       
2. Moving objects range query -         
3. Moving objects nearest neighbor query -          
4. Moving objects join query -         
5. Trajectory similarity query -        
"""

import sys
import os
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional





PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


from nalplan.slm_postprocessor import SLMPostProcessor


from nalplan.operator_retriever import OperatorRetriever


MOVING_OBJECT_QUERY_TYPES = [
    'Time interval query',
    'Moving objects range query',
    'Moving objects nearest neighbor query',
    'Moving objects join query',
    'Trajectory similarity query'
]






class ExecutableTemplateManager:
    """
              
           executable_templates.csv         
            LLM         
    """
    
    def __init__(self, template_path: str = None):
        """
                     
        
        Args:
            template_path: executable_templates.csv    
        """
        if template_path is None:
            template_path = os.path.join(PROJECT_ROOT, 'executable_templates.csv')
        
        self.template_path = template_path
        self.template_data = None
        self.query_types = []
        self.type_to_templates = {}  
        
        self._load_templates()
    
    def _load_templates(self):
        """  executable_templates.csv  """
        try:
            self.template_data = pd.read_csv(self.template_path, encoding='utf-8')
            print(f"[ExecTemplate]            : {self.template_path}")
            print(f"[ExecTemplate]     {len(self.template_data)}    ")
            
            
            self._parse_templates()
            
        except Exception as e:
            print(f"[ExecTemplate]       : {e}")
            raise
    
    def _parse_templates(self):
        """            """
        
        type_col = 'type'
        template_col = 'template'
        
        print(f"[ExecTemplate]          :")
        
        for qtype in MOVING_OBJECT_QUERY_TYPES:
            matching_rows = self.template_data[
                self.template_data[type_col] == qtype
            ]
            
            if len(matching_rows) > 0:
                templates = matching_rows[template_col].tolist()
                self.type_to_templates[qtype] = templates
                self.query_types.append(qtype)
                print(f"  - {qtype}: {len(templates)}    ")
            else:
                self.type_to_templates[qtype] = []
                print(f"  - {qtype}: 0    ")
        
        total = sum(len(v) for v in self.type_to_templates.values())
        print(f"[ExecTemplate]   {total}         ")
    
    def get_templates_by_type(self, query_type: str) -> List[str]:
        """                  """
        return self.type_to_templates.get(query_type, [])
    
    def get_all_templates(self) -> Dict[str, List[str]]:
        """             """
        return self.type_to_templates.copy()






class NLQTemplateManager:
    """
    NLQ     
        SLM.csv          
             
    """
    
    def __init__(self, slm_path: str = None):
        """
           NLQ     
        
        Args:
            slm_path: SLM.csv    
        """
        if slm_path is None:
            slm_path = os.path.join(PROJECT_ROOT, 'SLM.csv')
        
        self.slm_path = slm_path
        self.slm_data = None
        self.type_to_nlq_templates = {}
        
        self._load_slm()
    
    def _load_slm(self):
        """  SLM.csv  """
        try:
            self.slm_data = pd.read_csv(self.slm_path, encoding='utf-8')
            print(f"[NLQTemplate]     NLQ  : {self.slm_path}")
            
            
            columns = self.slm_data.columns.tolist()
            type_col = columns[0]
            template_col = columns[1]
            
            for qtype in MOVING_OBJECT_QUERY_TYPES:
                matching_rows = self.slm_data[
                    self.slm_data[type_col].str.lower() == qtype.lower()
                ]
                if len(matching_rows) > 0:
                    self.type_to_nlq_templates[qtype] = matching_rows[template_col].tolist()
            
            total = sum(len(v) for v in self.type_to_nlq_templates.values())
            print(f"[NLQTemplate]   {total}  NLQ       ")
            
        except Exception as e:
            print(f"[NLQTemplate]     : {e}")
    
    def instantiate_nlq(self, template: str) -> str:
        """
           NLQ             
        
        Args:
            template: NLQ     
            
        Returns:
                 NLQ
        """
        nlq = template
        
        
        replacements = {
            '<OBJECT>': random.choice(['train', 'taxi', 'bus', 'car']),
            '<OBJECT1>': random.choice(['train', 'taxi']),
            '<OBJECT2>': random.choice(['bus', 'car']),
            '<OBJECT_ID>': str(random.randint(1, 100)),
            '<OBJECT_ID1>': str(random.randint(1, 50)),
            '<OBJECT_ID2>': str(random.randint(51, 100)),
            '<SPECIFIC_LOCATION>': random.choice(['Mehringdamm', 'Alexanderplatz', 'Tiergarten', 'Westhafen']),
            '<START_TIME>': random.choice(['6:00am', '7:30am', '8:00am', '9:00am']),
            '<END_TIME>': random.choice(['10:00am', '11:30am', '12:00pm', '2:00pm']),
            '<SPECIFIC_TIME>': random.choice(['8:30am', '9:15am', '10:00am']),
            '<NUMBER>': str(random.randint(3, 10)),
        }
        
        for placeholder, value in replacements.items():
            nlq = nlq.replace(placeholder, value)
        
        return nlq
    
    def get_random_nlqs(self, query_type: str, count: int = 3) -> List[Tuple[str, str]]:
        """
                    NLQ
        
        Returns:
            List of (template, instantiated_nlq) tuples
        """
        templates = self.type_to_nlq_templates.get(query_type, [])
        if not templates:
            return []
        
        selected = random.sample(templates, min(count, len(templates)))
        return [(t, self.instantiate_nlq(t)) for t in selected]






class QueryClassifier:
    """
           
      NLU            5         
    """
    
    def __init__(self):
        """      """
        self.postprocessor = SLMPostProcessor()
        self._load_nlu()
    
    def _load_nlu(self):
        """  NLU  """
        try:
            from NLU import predict_type, get_semantic_information, labels
            self.predict_type_func = predict_type
            self.get_semantic_information = get_semantic_information
            self.labels = labels
            print("[Classifier]     NLU  ")
        except Exception as e:
            print(f"[Classifier]   NLU    : {e}")
            self.predict_type_func = None
            self.get_semantic_information = None
            self.labels = []
    
    def classify(self, nlq: str) -> Tuple[str, str]:
        """ NLQ               """
        if self.predict_type_func is not None:
            try:
                nlu_type = self.predict_type_func(nlq)
            except:
                nlu_type = "Normal Basic Query"
        else:
            nlu_type = self._fallback_classify(nlq)
        
        slm_type, reason = self.postprocessor.postprocess(nlq, nlu_type)
        return slm_type, reason
    
    def parse(self, nlq: str) -> Dict:
        """  NLQ       """
        result = {
            'query_type': '',
            'postprocess_reason': '',
            'nlu_type': '',
            'places': [],
            'time': [],
            'object_ids': [],
            'neighbor_num': '0',
        }
        
        if self.get_semantic_information is not None:
            try:
                (nlu_type, spatial_relations, places, neighbor_num, 
                 max_distance, time, obj_ids, sorted_attr,
                 is_less, is_more, is_decreasing, is_increasing) = self.get_semantic_information(nlq)
                
                slm_type, reason = self.postprocessor.postprocess(nlq, nlu_type)
                
                result = {
                    'query_type': slm_type,
                    'postprocess_reason': reason,
                    'nlu_type': nlu_type,
                    'places': places,
                    'time': time,
                    'object_ids': obj_ids,
                    'neighbor_num': neighbor_num,
                }
            except Exception as e:
                slm_type, reason = self.classify(nlq)
                result['query_type'] = slm_type
                result['postprocess_reason'] = reason
        else:
            slm_type, reason = self.classify(nlq)
            result['query_type'] = slm_type
            result['postprocess_reason'] = reason
        
        return result
    
    def _fallback_classify(self, nlq: str) -> str:
        """       -     MOVING_OBJECT_QUERY_TYPES        """
        nlq_lower = nlq.lower()
        
        if 'nearest' in nlq_lower or 'closest' in nlq_lower:
            return 'Moving objects nearest neighbor query'
        elif 'similar' in nlq_lower:
            return 'Trajectory similarity query'
        elif 'same' in nlq_lower and 'trajectory' in nlq_lower:
            return 'Moving objects join query'
        elif 'pass' in nlq_lower or 'through' in nlq_lower:
            return 'Moving objects range query'
        elif 'between' in nlq_lower:
            return 'Time interval query'
        else:
            return 'Time interval query'  






class ExecutableTemplateMatcher:
    """
              
      NLQ                  
              
    """
    
    def __init__(self, exec_template_manager: ExecutableTemplateManager, 
                 classifier: QueryClassifier,
                 operator_retriever: OperatorRetriever = None):
        self.exec_manager = exec_template_manager
        self.classifier = classifier
        self.operator_retriever = operator_retriever
    
    def match(self, nlq: str, retrieve_operators: bool = True, 
              operator_top_k: int = 30) -> Dict:
        """
         NLQ                          
        
        Args:
            nlq:       
            retrieve_operators:          
            operator_top_k:         
            
        Returns:
                                          
        """
        
        semantic_info = self.classifier.parse(nlq)
        query_type = semantic_info['query_type']
        
        
        exec_templates = self.exec_manager.get_templates_by_type(query_type)
        
        
        scored_templates = self._score_templates(nlq, exec_templates, semantic_info)
        
        
        all_templates = [t[0] for t in scored_templates]
        all_scores = [t[1] for t in scored_templates]
        
        
        retrieved_operators = []
        if retrieve_operators and self.operator_retriever is not None:
            operator_results = self.operator_retriever.search_combined(
                nlq=nlq,
                exec_templates=all_templates,  
                query_type=query_type,
                top_k=operator_top_k
            )
            retrieved_operators = [
                {
                    'name': op.name,
                    'algebra': op.algebra,
                    'signature': op.signature,
                    'example': op.example,
                    'score': score
                }
                for op, score in operator_results
            ]
        
        return {
            'nlq': nlq,
            'query_type': query_type,
            'semantic_info': semantic_info,
            'matched_templates': all_templates,  
            'scores': all_scores,  
            'total_candidates': len(exec_templates),
            'retrieved_operators': retrieved_operators
        }
    
    def _score_templates(self, nlq: str, templates: List[str], 
                         semantic_info: Dict) -> List[Tuple[str, float]]:
        """       """
        scored = []
        nlq_lower = nlq.lower()
        
        for template in templates:
            score = self._calculate_score(nlq_lower, template, semantic_info)
            scored.append((template, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _calculate_score(self, nlq_lower: str, template: str, 
                         semantic_info: Dict) -> float:
        """            """
        score = 0.0
        template_lower = template.lower()
        
        
        structure_score = 0.0
        
        
        if semantic_info.get('time'):
            if 'periods' in template_lower or 'instant' in template_lower:
                structure_score += 0.3
            if 'atperiods' in template_lower or 'atinstant' in template_lower:
                structure_score += 0.2
        
        
        if semantic_info.get('places'):
            if '<location>' in template_lower or '<aux_table>' in template_lower:
                structure_score += 0.3
            if 'at' in template_lower or 'passes' in template_lower:
                structure_score += 0.2
        
        
        if semantic_info.get('object_ids'):
            if '<object_id>' in template_lower:
                structure_score += 0.3
        
        
        if semantic_info.get('neighbor_num') and semantic_info['neighbor_num'] != '0':
            if '<k>' in template_lower or 'knearest' in template_lower:
                structure_score += 0.3
        
        score += 0.4 * min(structure_score, 1.0)
        
        
        operator_keywords = {
            'trajectory': 0.1,
            'filter': 0.05,
            'feed': 0.05,
            'consume': 0.05,
            'knearest': 0.15,
            'distancescan': 0.15,
            'symmjoin': 0.15,
            'intersects': 0.1,
            'deftime': 0.1,
        }
        
        operator_score = 0.0
        for op, weight in operator_keywords.items():
            if op in template_lower:
                operator_score += weight
        
        score += 0.3 * min(operator_score, 1.0)
        
        
        nlq_keywords = set(nlq_lower.split())
        keyword_matches = 0
        
        important_keywords = ['train', 'taxi', 'bus', 'car', 'nearest', 'similar', 
                            'between', 'pass', 'through', 'same', 'trajectory']
        for kw in important_keywords:
            if kw in nlq_lower:
                keyword_matches += 1
        
        score += 0.3 * min(keyword_matches / 5, 1.0)
        
        return min(score, 1.0)






class ExecutableLanguageGenerator:
    """
            
             NLQ                
    """
    
    def __init__(self, enable_operator_retrieval: bool = True):
        """      """
        print("=" * 70)
        print("    Text-to-Executable-Language     ")
        print("=" * 70)
        
        
        self.exec_manager = ExecutableTemplateManager()
        self.nlq_manager = NLQTemplateManager()
        self.classifier = QueryClassifier()
        
        
        self.operator_retriever = None
        if enable_operator_retrieval:
            print("\n[Generator]          ...")
            self.operator_retriever = OperatorRetriever(use_cache=True)
        
        
        self.matcher = ExecutableTemplateMatcher(
            self.exec_manager, 
            self.classifier,
            self.operator_retriever
        )
        
        print("\n[Generator]       ")
        print(f"[Generator]        : {MOVING_OBJECT_QUERY_TYPES}")
        if self.operator_retriever:
            print(f"[Generator]      :    ")
    
    def process(self, nlq: str, operator_top_k: int = 30) -> Dict:
        """
          NLQ                
        
        Args:
            nlq:       
            operator_top_k:         
            
        Returns:
                               
        """
        return self.matcher.match(nlq, retrieve_operators=True, operator_top_k=operator_top_k)
    
    def display_result(self, result: Dict, show_operators: bool = True, 
                       show_top_templates: int = 5):
        """
              
        
        Args:
            result:     
            show_operators:        
            show_top_templates:    N    0     
        """
        print("\n" + "=" * 70)
        print("    ")
        print("=" * 70)
        print(f"NLQ: {result['nlq']}")
        print(f"    : {result['query_type']}")
        print(f"        : {result['total_candidates']}")
        
        print(f"\n    :")
        for key, value in result['semantic_info'].items():
            if key not in {'query_type', 'postprocess_reason', 'nlu_type'} and value and value != '0' and value != []:
                print(f"  - {key}: {value}")
        
        
        templates_to_show = result['matched_templates']
        scores_to_show = result['scores']
        
        if show_top_templates > 0 and len(templates_to_show) > show_top_templates:
            print(f"\n        (    {show_top_templates} /   {len(templates_to_show)}  ):")
            templates_to_show = templates_to_show[:show_top_templates]
            scores_to_show = scores_to_show[:show_top_templates]
        else:
            print(f"\n          (  {len(templates_to_show)}  ):")
        
        for i, (template, score) in enumerate(zip(templates_to_show, scores_to_show), 1):
            display_template = template[:100] + "..." if len(template) > 100 else template
            print(f"  {i}. [Score: {score:.3f}]")
            print(f"     {display_template}")
        
        
        if show_operators and result.get('retrieved_operators'):
            print(f"\n          (  {len(result['retrieved_operators'])}  ):")
            for i, op in enumerate(result['retrieved_operators'][:10], 1):
                print(f"  {i}. {op['name']} ({op['algebra']}) [Score: {op['score']:.3f}]")
                print(f"     Signature: {op['signature']}")
                example_display = op['example'][:60] + "..." if len(op['example']) > 60 else op['example']
                print(f"     Example: {example_display}")
    
    def run_tests(self, samples_per_type: int = 2):
        """
              SLM.csv   NLQ                  
        """
        print("\n" + "=" * 70)
        print("    NLQ                 ")
        print("=" * 70)
        
        total_tests = 0
        successful_matches = 0
        
        for query_type in MOVING_OBJECT_QUERY_TYPES:
            print(f"\n{'='*70}")
            print(f"    : {query_type}")
            print("=" * 70)
            
            
            nlq_samples = self.nlq_manager.get_random_nlqs(query_type, samples_per_type)
            
            if not nlq_samples:
                print(f"  [!]      NLQ  ")
                continue
            
            for template, nlq in nlq_samples:
                total_tests += 1
                
                
                result = self.process(nlq)
                
                
                is_correct = result['query_type'] == query_type
                if is_correct and result['matched_templates']:
                    successful_matches += 1
                    status = " "
                else:
                    status = " "
                
                print(f"\n  {status} NLQ: {nlq[:80]}...")
                print(f"        : {query_type}")
                print(f"        : {result['query_type']}")
                print(f"         : {len(result['matched_templates'])}/{result['total_candidates']}")
                
                if result['matched_templates']:
                    best_template = result['matched_templates'][0][:80] + "..."
                    print(f"        : {best_template}")
        
        
        print("\n" + "=" * 70)
        print("    ")
        print("=" * 70)
        print(f"    : {total_tests}")
        print(f"    : {successful_matches}")
        print(f"   : {successful_matches/total_tests*100:.1f}%" if total_tests > 0 else "N/A")






if __name__ == "__main__":
    
    generator = ExecutableLanguageGenerator()
    
    
    generator.run_tests(samples_per_type=2)
    
    
    print("\n" + "=" * 70)
    print("        ")
    print("=" * 70)
    
    test_nlqs = [
        "What is the trajectory of train 7 between 6am and 10am?",
        "Which trains passed through Mehringdamm at 8:30am?",
        "Find the 5 nearest taxis to train 15 between 9am and 11am.",
        "Find the same trajectory between train 5 and taxi 8 from 6am to 9am.",
        "Which taxi has a similar trajectory to taxi 30 between 8am and 12pm?",
    ]
    
    for nlq in test_nlqs:
        result = generator.process(nlq)  
        generator.display_result(result, show_top_templates=5)  
        print("\n" + "-" * 70)
