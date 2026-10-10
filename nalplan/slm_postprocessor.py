
"""
SLM         
  5 SLM            
1. Time interval query
2. Moving objects range query  
3. Moving objects nearest neighbor query
4. Moving objects join query
5. Trajectory similarity query
"""

import re
from typing import Tuple, List, Optional

class SLMPostProcessor:
    """
    SLM        
              NLU        
    """
    
    
    SLM_TYPES = [
        'Time interval query',
        'Moving objects range query',
        'Moving objects nearest neighbor query', 
        'Moving objects join query',
        'Trajectory similarity query'
    ]
    
    
    
    NLU_TO_SLM_MAPPING = {
        
        'Time Interval Query': 'Time interval query',
        'Time Point Query': 'Time interval query',
        'Time Range Query': 'Time interval query',
        
        
        'Spatial Range Query': 'Moving objects range query',
        'Spatio-temporal Range Query': 'Moving objects range query',
        
        
        'Spatial Nearest Neighbor Query': 'Moving objects nearest neighbor query',
        'Moving Objects Nearest Neighbor Query': 'Moving objects nearest neighbor query',
        
        
        'Spatial Join Query': 'Moving objects join query',
        'Spatio-temporal Join Query': 'Moving objects join query',
        
        
        'Similarity Query': 'Trajectory similarity query',
        
        
        'Normal Basic Query': None,
        'Spatial Basic-distance Query': None,
        'Spatial Basic-direction Query': None,
        'Spatial Basic-length Query': None,
        'Spatial Basic-area Query': None,
        'Spatial Aggregation-count Query': None,
        'Spatial Aggregation-sum Query': None,
        'Spatial Aggregation-max Query': None,
        'Temporal Aggregation Query': None,
    }
    
    def __init__(self):
        """       """
        
        self.moving_objects = ['train', 'trains', 'taxi', 'taxis', 'bus', 'buses', 
                               'vehicle', 'vehicles', 'car', 'cars']
        
        
        self.location_verbs = ['pass', 'passed', 'passing', 'through', 'across', 'via',
                               'visit', 'visited', 'visiting', 'enter', 'entered', 
                               'cross', 'crossed', 'crossing', 'reach', 'reached',
                               'go', 'went', 'gone', 'travel', 'traveled', 'travelling',
                               'move', 'moved', 'moving', 'near', 'along', 'at', 'in']
        
        
        self.time_keywords = ['am', 'pm', 'between', 'from', 'to', 'during', 'at',
                              'interval', 'time', 'hour', 'minute', 'o\'clock']
        
        
        self.nearest_keywords = ['nearest', 'closest', 'neighbor', 'neighbours', 
                                 'approaches', 'close to', 'near to', 'next to']
        
        
        self.join_keywords = [
            'same trajectory', 'same path', 'same route',
            'matching trajectory', 'matching trajectories', 'matching path', 'matching paths',
            'common trajectory', 'common path', 'common route',
            'shared trajectory', 'shared path', 'shared route',
            'overlapping trajectory', 'overlapping trajectories', 'overlapping path', 'overlapping paths',
            'coinciding trajectory', 'coinciding trajectories', 'coinciding path', 'coinciding paths',
            'coincide', 'coincides',
            'identical trajectory', 'identical trajectories', 'identical path', 'identical paths',
            'identical movement', 'identical movements',
            'trajectory that match', 'trajectories that match', 'path that match', 'paths that match',
            'trajectory intersection', 'trajectory intersections',
            'compare trajectory', 'compare trajectories', 'compare path', 'compare paths',
            'compare movement', 'compare movements',
            'replicate the trajectory', 'replicate the path',
            'follow the same', 'following the same',
            'share a path', 'share a trajectory', 'shares a path', 'shares a trajectory',
            'share the trajectory', 'shares the trajectory',
            'sharing the same', 'traveled the same',
            'move together', 'moved together',
            'trajectory of .* and .*',  
        ]
        
        
        self.similarity_keywords = [
            'similar', 'similarity', 'similarities',
            'resemble', 'resembles', 'resembling', 'resemblance',
            'comparable', 'compare',
            'close to', 'closest to', 'closest',
            'match', 'matches', 'matching',
            'like', 'alike',
            'correlate', 'correlated', 'correlation',
            'overlap', 'overlapping', 'overlaps',
            'align', 'aligned', 'aligns',
            'mimic', 'mimics',
            'demonstrate similar', 'demonstrates similar',
            'movement pattern', 'motion pattern',
        ]
        
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """            """
        
        self.time_pattern = re.compile(
            r'\d+\s*(am|pm|:\d+|o\'clock)|between\s+\d+|from\s+\d+.*to\s+\d+|during|at\s+\d+',
            re.IGNORECASE
        )
        
        
        self.location_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
        
        self.object_id_pattern = re.compile(
            r'\b(train|taxi|bus|vehicle|car)s?\s+(\d+)\b',
            re.IGNORECASE
        )
        
        
        self.two_objects_pattern = re.compile(
            r'\b(train|taxi|bus|vehicle|car)s?\s+(\d+)\s+(and|,|with)\s+(train|taxi|bus|vehicle|car)?\s*(\d+)\b',
            re.IGNORECASE
        )
        
        
        self.join_pattern = re.compile(
            r'(same\s+(trajectory|path|route)|'
            r'matching\s+(trajectory|trajectories|path|paths)|'
            r'common\s+(trajectory|path|route)|'
            r'shared\s+(trajectory|path|route)|shared\s+a\s+(path|trajectory)|'
            r'share[sd]?\s+the\s+(trajectory|path)|'
            r'share[sd]?\s+a\s+(path|trajectory)|'
            r'coincid(e|es|ed|ing)\s*(trajectory|trajectories|path|paths)?|'
            r'identical\s+(trajectory|trajectories|path|paths|movement)|'
            r'trajectory\s+intersection|'
            r'overlapping\s+(trajectory|trajectories|path|paths)|'
            r'overlap(s|ped|ping)?\s+(trajectory|trajectories|path|paths)|'
            r'overlap(s|ped|ping)?\s+for\s+|'
            r'overlap(s|ped|ping)?\s+between\s+|'
            r'trajectories\s+that\s+overlap|trajectory\s+that\s+overlap|'
            r'paths\s+that\s+overlap|path\s+that\s+overlap|'
            r'replicate\s+the\s+(trajectory|path)|'
            r'follow(s|ed|ing)?\s+the\s+same|'
            r'move(d|s)?\s+together|'
            r'traveling\s+the\s+same|traveled\s+the\s+same|'
            r'trajectories\s+that\s+match|trajectory\s+that\s+match|'
            r'paths\s+that\s+match|path\s+that\s+match)',
            re.IGNORECASE
        )
        
        
        self.similarity_pattern = re.compile(
            r'(similar(ity|ities)?|resemble?s?|resembl(ing|ance)|'
            r'comparable|correlat(e|ed|ion)|'
            r'mimic(s)?|demonstrate[sd]?\s+similar|'
            r'movement\s+pattern|motion\s+pattern|'
            r'trajectory\s+pattern|path\s+pattern|'
            r'close\s+in\s+shape|close\s+to|'
            r'most\s+like|looks?\s+like|'
            r'trajectory\s+distance|'
            r'align(s|ed)?|'
            r'movement\s+trend)',
            re.IGNORECASE
        )
    
    def postprocess(self, nlq: str, nlu_predicted_type: str) -> Tuple[str, str]:
        """
         NLU         
        
        Args:
            nlq:       
            nlu_predicted_type: NLU         
            
        Returns:
            Tuple[str, str]: (    SLM  ,     )
        """
        nlq_lower = nlq.lower()
        
        
        has_moving_object = self._has_moving_object(nlq_lower)
        has_time = self._has_time(nlq_lower)
        has_location_action = self._has_location_action(nlq_lower, nlq)  
        has_location_name = self._has_location_name(nlq)
        has_nearest = self._has_nearest_keyword(nlq_lower)
        has_join_pattern = self._has_join_pattern(nlq_lower)
        has_similarity_pattern = self._has_similarity_pattern(nlq_lower)
        has_two_objects = self._has_two_objects(nlq_lower)
        has_strong_similarity = self._has_strong_similarity_pattern(nlq_lower)
        
        
        
        
        
        if has_strong_similarity:
            return 'Trajectory similarity query', 'Rule0: Contains strong similarity keywords (similarity score/similar trajectory)'
        
        
        if has_nearest and has_moving_object and not self._is_trajectory_similarity_nearest(nlq_lower):
            return 'Moving objects nearest neighbor query', 'Rule1: Contains nearest/closest keywords with moving objects'
        
        
        
        if self._has_strong_join_pattern(nlq_lower):
            return 'Moving objects join query', 'Rule2: Contains strong join pattern keywords (same/shared/common/identical)'
        
        
        if has_similarity_pattern:
            return 'Trajectory similarity query', 'Rule3: Contains similarity pattern keywords'
        
        
        if has_join_pattern and has_two_objects:
            return 'Moving objects join query', 'Rule4: Contains join pattern with two objects'
        
        
        if has_two_objects and self._is_trajectory_comparison(nlq_lower) and not has_strong_similarity:
            return 'Moving objects join query', 'Rule4b: Comparison between two specific objects'
        
        
        if has_moving_object and has_location_action and has_location_name:
            
            if not has_join_pattern and not has_similarity_pattern and not has_strong_similarity:
                return 'Moving objects range query', 'Rule5: Moving object with location action and location name'
        
        
        if has_moving_object and has_time and not has_location_action:
            
            if not has_join_pattern and not has_similarity_pattern and not has_strong_similarity:
                return 'Time interval query', 'Rule6: Moving object with time but no location action'
        
        
        if self._is_time_interval_query(nlq_lower):
            return 'Time interval query', 'Rule7: Time interval query patterns detected'
        
        
        if self._is_range_query(nlq_lower):
            return 'Moving objects range query', 'Rule8: Range query patterns detected'
        
        
        mapped_type = self.NLU_TO_SLM_MAPPING.get(nlu_predicted_type)
        if mapped_type:
            return mapped_type, f'Rule9: Direct mapping from NLU type {nlu_predicted_type}'
        
        
        return self._fallback_classification(nlq_lower, nlu_predicted_type)
    
    def _has_moving_object(self, nlq_lower: str) -> bool:
        """             """
        return any(obj in nlq_lower for obj in self.moving_objects)
    
    def _has_time(self, nlq_lower: str) -> bool:
        """          """
        return bool(self.time_pattern.search(nlq_lower))
    
    def _has_location_action(self, nlq_lower: str, nlq_original: str = None) -> bool:
        """            """
        if nlq_original is None:
            nlq_original = nlq_lower
            
        
        location_action_patterns = [
            r'pass(ed|ing|es)?\s+(through|by|near|at)\s+[A-Za-z]',
            r'pass(ed|ing|es)?\s+[A-Z][a-z]+',  
            r'visit(ed|ing|s)?\s+[A-Z][a-z]+',
            r'cross(ed|ing|es)?\s+[A-Z][a-z]+',
            r'reach(ed|ing|es)?\s+[A-Z][a-z]+',
            r'(go|went|gone|goes)\s+(through|to|near|along)\s+[A-Za-z]',
            r'travel(ed|ing|s)?\s+(through|to|near|along|past)\s+[A-Za-z]',
            r'mov(e|ed|ing|es)?\s+(through|to|near|along|past|across)\s+[A-Za-z]',
            r'enter(ed|ing|s)?\s+[A-Z][a-z]+',
            r'appear(ed|ing|s)?\s+(at|in)\s+[A-Z][a-z]+',
            r'present\s+at\s+[A-Z][a-z]+',
            r'at\s+[A-Z][a-z]+\s+(at|between|from|during)',  
            r'in\s+[A-Z][a-z]+\s+(at|between|from|during)',  
            r'near\s+[A-Z][a-z]+',
            r'along\s+[A-Z][a-z]+',
            r'across\s+[A-Z][a-z]+',
        ]
        for pattern in location_action_patterns:
            if re.search(pattern, nlq_original, re.IGNORECASE):
                return True
        return False
    
    def _has_location_name(self, nlq: str) -> bool:
        """                     """
        
        exclude_words = {'What', 'Which', 'Where', 'When', 'How', 'Find', 'Show', 
                        'List', 'Tell', 'Get', 'Retrieve', 'Display', 'Did', 'Are',
                        'Is', 'Can', 'Could', 'The', 'A', 'An', 'All', 'Any',
                        'Train', 'Taxi', 'Bus', 'Vehicle', 'Car', 'Trains', 'Taxis',
                        'Buses', 'Vehicles', 'Cars', 'Compare', 'Identify', 'Determine'}
        
        locations = self.location_pattern.findall(nlq)
        for loc in locations:
            if loc not in exclude_words and not loc.isdigit():
                return True
        return False
    
    def _has_nearest_keyword(self, nlq_lower: str) -> bool:
        """            """
        return any(kw in nlq_lower for kw in self.nearest_keywords)
    
    def _has_join_pattern(self, nlq_lower: str) -> bool:
        """      join    """
        return bool(self.join_pattern.search(nlq_lower))
    
    def _has_similarity_pattern(self, nlq_lower: str) -> bool:
        """             """
        return bool(self.similarity_pattern.search(nlq_lower))
    
    def _has_strong_similarity_pattern(self, nlq_lower: str) -> bool:
        """                        """
        strong_patterns = [
            r'similarity\s+score',
            r'similarity\s+between',
            r'similar\s+trajectory',
            r'similar\s+trajectories',
            r'similar\s+path',
            r'similar\s+paths',
            r'similar\s+route',
            r'similar\s+movement',
            r'trajectory\s+similarity',
            r'path\s+similarity',
            r'movement\s+similarity',
            r'similar\s+to\s+\w+\s+\d+',  
            r'similarity\s+of\s+\w+',
            r'resembles?\s+\w+\s+\d+',  
            r'similar\s+.*\s+pattern',
            r'compare.*similarity',
            r'similarity.*compare',
            r'matching\s+similar',
            r'similar\s+matching',
            r'demonstrate[sd]?\s+similar',
            r'high(er|est)?\s+similarity',
            r'trajectory\s+pattern',
            r'movement\s+pattern',
            r'motion\s+pattern',
            r'route\s+similarity',
            r'closest\s+trajectory\s+pattern',
            r'trajectory\s+is\s+closest',
            r'most\s+similar',
            r'highly\s+similar',
            r'nearly\s+identical',
            
            r'compare\s+(the\s+)?(trajectory|trajectories|movement|path)',
            r'compare\s+\w+\s+trajectories',
            r'trajectory\s+closest\s+to',
            r'which\s+one\s+had\s+a\s+trajectory\s+closest',
            r'among\s+all\s+\w+.*trajectory\s+closest',
        ]
        return any(re.search(p, nlq_lower) for p in strong_patterns)
    
    def _is_trajectory_similarity_nearest(self, nlq_lower: str) -> bool:
        """            nearest          """
        
        patterns = [
            r'closest\s+trajectory',
            r'closest\s+path',
            r'closest\s+route',
            r'closest\s+movement',
            r'nearest\s+trajectory',
            r'trajectory.*closest',
            r'path.*closest',
            r'closest.*pattern',
            r'nearest.*pattern',
            r'closest\s+motion',
            r'movement.*closest',
        ]
        return any(re.search(p, nlq_lower) for p in patterns)
    
    def _is_asking_same_trajectory(self, nlq_lower: str) -> bool:
        """                 join        """
        patterns = [
            r'which\s+\w+\s+have\s+(the\s+)?same\s+(trajectory|path|route)',
            r'which\s+\w+\s+had\s+(the\s+)?same\s+(trajectory|path|route)',
            r'which\s+\w+\s+share[sd]?\s+(the\s+)?(same\s+)?(trajectory|path|route)',
            r'did\s+\w+\s+have\s+(the\s+)?same\s+trajectory',
            r'what\s+\w+\s+share[sd]?\s+(the\s+)?(same\s+)?(trajectory|path)',
            r'find\s+\w+\s+.*same\s+trajectory',
        ]
        return any(re.search(p, nlq_lower) for p in patterns)
    
    def _has_strong_join_pattern(self, nlq_lower: str) -> bool:
        """       Join              /  /     """
        strong_patterns = [
            r'same\s+(trajectory|path|route)',
            r'shar(e|es|ed|ing)\s+(the\s+)?(trajectory|path|route|a\s+trajectory|a\s+path)',
            r'trajectory\s+(that\s+)?.*\s+shar(e|es|ed)\s+with',  
            r'path\s+(that\s+)?.*\s+shar(e|es|ed)\s+with',
            r'common\s+(trajectory|path|route)',
            r'identical\s+(trajectory|trajectories|path|paths|movement)',
            r'coincid(e|es|ed|ing)\s+(with\s+)?(trajectory|trajectories|path|paths)?',
            r'trajectory\s+(that\s+)?coincid',
            r'follow(s|ed|ing)?\s+the\s+same\s+(trajectory|path|route)',
            r'move(d|s)?\s+together',
            r'trajectory\s+intersection',
            r'overlapping\s+(trajectory|trajectories|path|paths)',
            r'overlap(s|ped|ping)?\s+(trajectory|trajectories|path|paths)',
            r'trajectory\s+overlapping\s+with',
            r'trajectory\s+matching\s+',  
            r'trajectories\s+matching\s+',
        ]
        return any(re.search(p, nlq_lower) for p in strong_patterns)
    
    def _has_two_objects(self, nlq_lower: str) -> bool:
        """             join   """
        return bool(self.two_objects_pattern.search(nlq_lower))
    
    def _is_trajectory_comparison(self, nlq_lower: str) -> bool:
        """         """
        comparison_patterns = [
            r'trajectory\s+of\s+\w+\s+\d+\s+and',
            r'compare\s+.*trajectory',
            r'between\s+\w+\s+\d+\s+and\s+\w+\s+\d+',
            r'matching\s+.*for\s+\w+\s+\d+\s+and',
            r'overlap.*between',
        ]
        return any(re.search(p, nlq_lower) for p in comparison_patterns)
    
    def _is_finding_similar_objects(self, nlq_lower: str) -> bool:
        """                        """
        finding_patterns = [
            r'which\s+\w+\s+(has|have|is|are|shows?|had)\s+.*similar',
            r'find\s+\w+\s+.*similar',
            r'what\s+\w+\s+.*similar',
            r'tell\s+me\s+which',
            r'identify\s+\w+\s+.*similar',
            r'retrieve\s+\w+\s+.*similar',
        ]
        return any(re.search(p, nlq_lower) for p in finding_patterns)
    
    def _is_time_interval_query(self, nlq_lower: str) -> bool:
        """           """
        
        time_only_patterns = [
            r'where\s+(was|were)\s+\w+\s+\d*\s*(between|from|at|during)',
            r'(what|where)\s+is\s+the\s+(trajectory|path|route|location|position)',
            r'(show|display|list|tell)\s+(me\s+)?(the\s+)?(trajectory|path|route|movement|location)',
            r'what\s+did\s+\w+\s+\d*\s+do\s+(at|between|from|during)',
            r'which\s+(places|locations)\s+did',
            r'trace\s+the\s+movement',
            r'how\s+(far|long)\s+did',
            r'(positions?|locations?)\s+of\s+\w+\s+\d*\s+(from|between|at)',
        ]
        return any(re.search(p, nlq_lower) for p in time_only_patterns)
    
    def _is_range_query(self, nlq_lower: str) -> bool:
        """     range  """
        range_patterns = [
            r'(pass|went|traveled|moved|visited|crossed)\s+(through|by|near|at|along)\s+[A-Za-z]',
            r'at\s+[A-Za-z]+\s+(at|between|from)\s+\d',
            r'in\s+[A-Za-z]+\s+(at|between|from)\s+\d',
        ]
        return any(re.search(p, nlq_lower, re.IGNORECASE) for p in range_patterns)
    
    def _fallback_classification(self, nlq_lower: str, nlu_type: str) -> Tuple[str, str]:
        """      """
        
        scores = {
            'Time interval query': 0,
            'Moving objects range query': 0,
            'Moving objects nearest neighbor query': 0,
            'Moving objects join query': 0,
            'Trajectory similarity query': 0
        }
        
        
        if self._has_time(nlq_lower):
            scores['Time interval query'] += 2
        if 'trajectory' in nlq_lower or 'path' in nlq_lower or 'route' in nlq_lower:
            scores['Time interval query'] += 1
        
        
        if self._has_location_action(nlq_lower):
            scores['Moving objects range query'] += 3
        
        
        if 'nearest' in nlq_lower or 'closest' in nlq_lower:
            scores['Moving objects nearest neighbor query'] += 5
        
        
        if 'same' in nlq_lower or 'matching' in nlq_lower or 'common' in nlq_lower:
            scores['Moving objects join query'] += 2
        if 'overlap' in nlq_lower or 'coincid' in nlq_lower:
            scores['Moving objects join query'] += 2
        
        
        if 'similar' in nlq_lower:
            scores['Trajectory similarity query'] += 3
        if 'resemble' in nlq_lower or 'comparable' in nlq_lower:
            scores['Trajectory similarity query'] += 2
        if 'pattern' in nlq_lower and ('trajectory' in nlq_lower or 'movement' in nlq_lower):
            scores['Trajectory similarity query'] += 1
        
        
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type, f'Rule10: Fallback scoring (score={scores[best_type]})'
        
        
        return 'Time interval query', 'Rule10: Default fallback to Time interval query'


def test_postprocessor():
    """      """
    processor = SLMPostProcessor()
    
    test_cases = [
        
        ("What is the trajectory of train 7 between 6am and 7am?", "Similarity Query"),
        ("Where was vehicle 42 located from 10:00 to 9am?", "Spatio-temporal Range Query"),
        ("Which places did taxi 96 go during 6am, 11:00?", "Spatio-temporal Range Query"),
        
        
        ("Did car 15 pass through the Tiergarten at 6:20?", "Time Point Query"),
        ("Which bus passed through Tiergarten at 7:15am?", "Time Point Query"),
        ("Show all car that visited Tiergarten at 9am.", "Time Point Query"),
        
        
        ("Find the 5 nearest taxi to bus 30 that move through Mehringdamm.", "Similarity Query"),
        ("Which car approaches train 42 most closely between 6am and 10am?", "Time Interval Query"),
        
        
        ("Which bus have the same trajectory.", "Spatial Aggregation-max Query"),
        ("Find the common trajectory between taxi 8 and train 7 from 7:15am to 9am.", "Time Interval Query"),
        ("Retrieve the matching trajectories for taxi 5 and bus 3 from 6am to 9am.", "Time Interval Query"),
        ("Did taxi 8 and train 12 share a path from 6am to 10am?", "Spatial Basic-direction Query"),
        
        
        ("Tell me which taxi has a similar trajectory to taxi 30.", "Normal Basic Query"),
        ("Which train has the closest trajectory pattern to bus 42?", "Moving Objects Nearest Neighbor Query"),
        ("Find vehicle whose movement paths are close to bus 96.", "Spatio-temporal Range Query"),
    ]
    
    print("=" * 80)
    print("      ")
    print("=" * 80)
    
    for nlq, nlu_type in test_cases:
        slm_type, reason = processor.postprocess(nlq, nlu_type)
        print(f"\nNLQ: {nlq}")
        print(f"NLU  : {nlu_type}")
        print(f"   : {slm_type}")
        print(f"  : {reason}")


if __name__ == "__main__":
    test_postprocessor()

