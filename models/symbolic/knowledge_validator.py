import re
import numpy as np
from collections import defaultdict

class KnowledgeValidator:
    """
    Validates knowledge consistency between different reasoning engines
    and ensures quality of knowledge transfers.
    """
    
    def __init__(self, prolog_engine=None, vector_engine=None):
        self.prolog = prolog_engine
        self.vector = vector_engine
        self.validation_history = []
        self.contradiction_patterns = {
            'direct_negation': [
                (r'(\w+)\s+(\w+)\s+(\w+)', r'not_\1\s+\2\s+\3'),
                (r'likes', r'hates'),
                (r'trusts', r'distrusts')
            ]
        }
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
    def validate_fact(self, fact, source_engine='prolog'):
        """
        Validate a single fact against existing knowledge.
        
        Args:
            fact: Dict with subject, predicate, object
            source_engine: Engine this fact came from
            
        Returns:
            Dict with validation results
        """
        # Basic structure for validation results
        validation = {
            'fact': fact,
            'is_valid': True,
            'confidence': fact.get('confidence', 0.5),
            'issues': [],
            'contradictions': [],
            'supporting_evidence': [],
            'provenance': source_engine
        }
        
        # Check for basic validity
        if not self._check_basic_validity(fact):
            validation['is_valid'] = False
            validation['issues'].append('Invalid fact structure')
            return validation
            
        # Check for contradictions
        contradictions = self._find_contradictions(fact)
        if contradictions:
            validation['contradictions'] = contradictions
            validation['confidence'] *= 0.5  # Reduce confidence if contradictions exist
            
        # Find supporting evidence
        supporting = self._find_supporting_evidence(fact, source_engine)
        validation['supporting_evidence'] = supporting
        
        # Adjust confidence based on evidence
        if supporting:
            # Boost confidence if we have supporting evidence
            evidence_confidence = np.mean([e['confidence'] for e in supporting])
            validation['confidence'] = min(1.0, validation['confidence'] * (1 + evidence_confidence))
            
        # Add validation record
        self.validation_history.append({
            'fact': fact,
            'source': source_engine,
            'result': validation,
            'timestamp': time.time()
        })
        
        return validation
        
    def _check_basic_validity(self, fact):
        """Check if fact has valid basic structure"""
        required = ['subject', 'predicate', 'object']
        return all(k in fact for k in required)
        
    def _find_contradictions(self, fact):
        """Find any contradicting facts in knowledge base"""
        contradictions = []
        
        # Check in Prolog KB
        if self.prolog:
            # Look for direct negations
            subject = fact['subject']
            predicate = fact['predicate']
            obj = fact['object']
            
            # Check for "not_" prefixed predicates
            if predicate.startswith('not_'):
                base_pred = predicate[4:]
                query = f"confident_fact({base_pred}, {subject}, {obj}, C)"
            else:
                query = f"confident_fact(not_{predicate}, {subject}, {obj}, C)"
                
            try:
                for result in self.prolog.prolog.query(query):
                    contradictions.append({
                        'type': 'direct_negation',
                        'fact': {
                            'subject': subject,
                            'predicate': str(result['P']),
                            'object': obj,
                            'confidence': float(result['C'])
                        },
                        'source': 'prolog'
                    })
            except Exception as e:
                print(f"Error querying Prolog: {e}")
                
        # Check in Vector KB
        if self.vector:
            # Query for semantically opposite facts
            try:
                antonym_preds = self._get_antonym_predicates(fact['predicate'])
                for anti_pred in antonym_preds:
                    anti_facts = self.vector.query_facts(
                        subject=fact['subject'],
                        predicate=anti_pred,
                        object=fact['object']
                    )
                    
                    for anti_fact in anti_facts:
                        contradictions.append({
                            'type': 'semantic_opposition',
                            'fact': anti_fact,
                            'source': 'vector'
                        })
            except Exception as e:
                print(f"Error querying Vector KB: {e}")
                
        return contradictions
        
    def _find_supporting_evidence(self, fact, source_engine):
        """Find evidence supporting this fact"""
        evidence = []
        
        # Skip checking the source engine
        if source_engine != 'prolog' and self.prolog:
            # Look for exact matches or implications in Prolog
            subject = fact['subject']
            predicate = fact['predicate']
            obj = fact['object']
            
            query = f"confident_fact({predicate}, {subject}, {obj}, C)"
            try:
                for result in self.prolog.prolog.query(query):
                    evidence.append({
                        'type': 'exact_match',
                        'fact': {
                            'subject': subject,
                            'predicate': predicate,
                            'object': obj,
                            'confidence': float(result['C'])
                        },
                        'source': 'prolog'
                    })
            except Exception as e:
                print(f"Error querying Prolog: {e}")
                
        if source_engine != 'vector' and self.vector:
            # Look for similar facts in vector space
            try:
                similar_facts = self.vector.query_facts(
                    subject=fact['subject'],
                    predicate=fact['predicate'],
                    object=fact['object'],
                    threshold=0.7
                )
                
                for similar in similar_facts:
                    evidence.append({
                        'type': 'semantic_similarity',
                        'fact': similar,
                        'source': 'vector'
                    })
            except Exception as e:
                print(f"Error querying Vector KB: {e}")
                
        return evidence
        
    def _get_antonym_predicates(self, predicate):
        """Get list of predicates that are antonyms"""
        antonyms = {
            'likes': ['hates', 'dislikes'],
            'trusts': ['distrusts', 'fears'],
            'helps': ['hinders', 'hurts'],
            'knows': ['doubts']
        }
        return antonyms.get(predicate, [f"not_{predicate}"])
    
    def analyze_validation_history(self):
        """Analyze validation history to improve validation rules"""
        if not self.validation_history:
            return None
            
        analysis = {
            'total_validations': len(self.validation_history),
            'success_rate': 0,
            'common_issues': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'improvement_suggestions': []
        }
        
        # Analyze validation results
        for validation in self.validation_history:
            # Track success rate
            if validation['result']['is_valid']:
                analysis['success_rate'] += 1
                
            # Track issues
            for issue in validation['result']['issues']:
                analysis['common_issues'][issue] += 1
                
            # Track confidence distribution
            conf = validation['result']['confidence']
            if conf >= self.confidence_thresholds['high']:
                analysis['confidence_distribution']['high'] += 1
            elif conf >= self.confidence_thresholds['medium']:
                analysis['confidence_distribution']['medium'] += 1
            else:
                analysis['confidence_distribution']['low'] += 1
                
        # Calculate success rate
        analysis['success_rate'] /= len(self.validation_history)
        
        # Generate improvement suggestions
        if analysis['success_rate'] < 0.8:
            analysis['improvement_suggestions'].append({
                'issue': 'Low validation success rate',
                'suggestion': 'Review and adjust validation rules'
            })
            
        # Find patterns in validation failures
        if analysis['common_issues']:
            top_issue = max(analysis['common_issues'].items(), key=lambda x: x[1])
            analysis['improvement_suggestions'].append({
                'issue': f'Common validation issue: {top_issue[0]}',
                'suggestion': 'Add specific validation rules for this case'
            })
            
        return analysis

    def adjust_thresholds(self, success_target=0.8):
        """Auto-adjust confidence thresholds based on validation history"""
        if len(self.validation_history) < 10:
            return False
            
        # Calculate current success rate
        current_rate = sum(1 for v in self.validation_history 
                         if v['result']['is_valid']) / len(self.validation_history)
                         
        if current_rate < success_target:
            # Too many failures - lower thresholds slightly
            self.confidence_thresholds = {
                k: max(0.3, v * 0.9) 
                for k, v in self.confidence_thresholds.items()
            }
        elif current_rate > success_target + 0.1:
            # Too many successes - raise thresholds slightly
            self.confidence_thresholds = {
                k: min(0.9, v * 1.1)
                for k, v in self.confidence_thresholds.items()
            }
            
        return True