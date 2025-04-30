import re
import time

class MetaReasoner:
    def __init__(self, bridge):
        self.bridge = bridge
        self.confidence_threshold = 0.7
        self.reasoning_history = []
        self.strategy_success_rates = {
            'symbolic': 0.8,  # Initial estimates
            'neural': 0.6,
            'abductive': 0.4,
            'hybrid': 0.7
        }
        
    def select_reasoning_strategy(self, query):
        """Select the best reasoning strategy for the given query"""
        # Features that influence strategy selection
        features = self._extract_features(query)
        
        # Strategy scoring
        scores = {}
        
        # Score symbolic reasoning
        if features['structured_query']:
            scores['symbolic'] = 0.8 * self.strategy_success_rates['symbolic']
        else:
            scores['symbolic'] = 0.4 * self.strategy_success_rates['symbolic']
            
        # Score neural reasoning
        if features['ambiguous']:
            scores['neural'] = 0.9 * self.strategy_success_rates['neural']
        else:
            scores['neural'] = 0.6 * self.strategy_success_rates['neural']
            
        # Score abductive reasoning
        if features['novel_concepts']:
            scores['abductive'] = 0.9 * self.strategy_success_rates['abductive']
        else:
            scores['abductive'] = 0.5 * self.strategy_success_rates['abductive']
            
        # Always consider hybrid approach
        scores['hybrid'] = 0.7 * self.strategy_success_rates['hybrid']
        
        # Select highest scoring strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])
        
        # Record the decision in reasoning history
        self.reasoning_history.append({
            'timestamp': time.time(),
            'query': query,
            'features': features,
            'scores': scores,
            'selected_strategy': best_strategy[0]
        })
        
        print(f"[Meta] Selected reasoning strategy: {best_strategy[0]} (score: {best_strategy[1]:.2f})")
        return best_strategy[0]
        
    def _extract_features(self, query):
        """Extract features from query that influence strategy selection"""
        query_lower = query.lower()
        
        features = {
            'structured_query': bool(re.search(r'\w+\s+(trusts|likes|fears)\s+\w+', query_lower)),
            'ambiguous': len(query_lower.split()) > 10,  # Simple heuristic for complexity
            'novel_concepts': not any(word in query_lower for word in 
                                  ['trust', 'like', 'hate', 'fear', 'distrust', 'avoid']),
            'is_question': '?' in query or any(word in query_lower for word in 
                                           ['who', 'what', 'when', 'where', 'why', 'how', 'does'])
        }
        
        return features
        
    def update_strategy_success(self, strategy, success_score):
        """Update success rate for a strategy based on feedback"""
        # Simple exponential moving average
        alpha = 0.2  # Learning rate
        current = self.strategy_success_rates.get(strategy, 0.5)
        updated = (1 - alpha) * current + alpha * success_score
        self.strategy_success_rates[strategy] = updated
        
        # Update the latest history entry with success information
        if self.reasoning_history:
            self.reasoning_history[-1]['success_score'] = success_score
            
        print(f"[Meta] Updated success rate for {strategy}: {current:.2f} → {updated:.2f}")
        
    def evaluate_result_quality(self, results):
        """Evaluate the quality of reasoning results"""
        # Simple quality metric based on confidence and result count
        if not results:
            return 0.0
            
        certainty_score = 0.0
        if results.get("certain"):
            # Higher score for certain results
            certainty_score = min(1.0, len(results["certain"]) * 0.5)
            # Add confidence bonus
            avg_confidence = sum(r['confidence'] for r in results["certain"]) / len(results["certain"])
            certainty_score *= (0.5 + 0.5 * avg_confidence)
            
        uncertainty_score = 0.0
        if results.get("uncertain"):
            # Some value for uncertain results, but less than certain
            uncertainty_score = min(0.5, len(results["uncertain"]) * 0.2)
            # Weight by confidence
            avg_confidence = sum(r['confidence'] for r in results["uncertain"]) / len(results["uncertain"])
            uncertainty_score *= (0.5 + 0.5 * avg_confidence)
            
        # Combined score favors certainty but values some uncertain results over none
        score = certainty_score + 0.5 * uncertainty_score
        
        # Normalize to 0-1 range
        return min(1.0, score)
        
    def get_reasoning_trace(self, limit=10):
        """Get recent reasoning traces"""
        return self.reasoning_history[-limit:]