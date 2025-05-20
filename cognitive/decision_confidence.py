from collections import defaultdict
import numpy as np

class DecisionConfidence:
    def __init__(self):
        self.confidence_history = defaultdict(list)
        self.decision_metrics = defaultdict(dict)
        self.uncertainty_threshold = 0.3
        self.high_confidence_threshold = 0.8
        
    def assess_confidence(self, decision_data):
        """Assess confidence in a decision based on multiple factors"""
        confidence_scores = {
            'evidence_support': self._evaluate_evidence_support(decision_data.get('evidence', [])),
            'consistency': self._evaluate_consistency(decision_data.get('process_trace', [])),
            'information_quality': self._evaluate_information_quality(decision_data.get('inputs', {})),
            'uncertainty': self._evaluate_uncertainty(decision_data)
        }
        
        # Calculate weighted confidence score
        weights = {
            'evidence_support': 0.4,
            'consistency': 0.3,
            'information_quality': 0.2,
            'uncertainty': 0.1
        }
        
        final_confidence = sum(score * weights[metric] 
                             for metric, score in confidence_scores.items())
                             
        # Record confidence assessment
        self.confidence_history[decision_data.get('type', 'general')].append({
            'confidence': final_confidence,
            'scores': confidence_scores,
            'context': decision_data.get('context', {})
        })
        
        return {
            'confidence': final_confidence,
            'component_scores': confidence_scores,
            'requires_verification': final_confidence < self.uncertainty_threshold,
            'high_confidence': final_confidence > self.high_confidence_threshold,
            'recommendations': self._generate_recommendations(confidence_scores)
        }

    def _evaluate_evidence_support(self, evidence):
        """Evaluate strength of supporting evidence"""
        if not evidence:
            return 0.0
            
        # Calculate evidence strength based on quantity and quality
        evidence_scores = [e.get('confidence', 0.0) for e in evidence]
        quantity_factor = min(1.0, len(evidence) / 5)  # Cap at 5 pieces of evidence
        
        return np.mean(evidence_scores) * quantity_factor

    def _evaluate_consistency(self, process_trace):
        """Evaluate consistency of the decision process"""
        if not process_trace:
            return 0.0
            
        # Check for consistent progression and lack of contradictions
        step_confidences = [step.get('confidence', 0.0) for step in process_trace]
        consistency = 1.0 - np.std(step_confidences)  # Higher std = lower consistency
        
        return max(0.0, consistency)

    def _evaluate_information_quality(self, inputs):
        """Evaluate quality and completeness of input information"""
        if not inputs:
            return 0.0
            
        required_fields = inputs.get('required_fields', [])
        provided_fields = inputs.get('provided_fields', [])
        
        if not required_fields:
            return 0.5  # Default if no requirements specified
            
        completeness = len(set(provided_fields) & set(required_fields)) / len(required_fields)
        quality_scores = [inputs.get('quality_scores', {}).get(f, 1.0) for f in provided_fields]
        
        return completeness * np.mean(quality_scores)

    def _evaluate_uncertainty(self, data):
        """Evaluate sources of uncertainty in the decision"""
        uncertainty_sources = data.get('uncertainty_sources', [])
        if not uncertainty_sources:
            return 1.0  # No explicit uncertainty
            
        # Calculate uncertainty impact
        total_impact = sum(u.get('impact', 0.0) for u in uncertainty_sources)
        uncertainty_score = 1.0 - min(1.0, total_impact)
        
        return uncertainty_score

    def _generate_recommendations(self, confidence_scores):
        """Generate recommendations for improving decision confidence"""
        recommendations = []
        
        if confidence_scores['evidence_support'] < 0.6:
            recommendations.append({
                'aspect': 'evidence',
                'suggestion': 'Gather additional supporting evidence',
                'priority': 'high'
            })
            
        if confidence_scores['consistency'] < 0.5:
            recommendations.append({
                'aspect': 'consistency',
                'suggestion': 'Review decision process for inconsistencies',
                'priority': 'medium'
            })
            
        if confidence_scores['information_quality'] < 0.7:
            recommendations.append({
                'aspect': 'information',
                'suggestion': 'Verify completeness of input information',
                'priority': 'high'
            })
            
        return recommendations
