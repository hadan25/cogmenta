from collections import defaultdict
import numpy as np

class StrategyLearner:
    def __init__(self):
        self.strategy_history = defaultdict(list)
        self.learning_rate = 0.2
        self.success_threshold = 0.7
        
    def record_strategy_outcome(self, strategy_name, features, success_score, execution_time):
        """Record outcome of a strategy execution"""
        self.strategy_history[strategy_name].append({
            'features': features,
            'success': success_score,
            'execution_time': execution_time,
            'feature_importance': self._calculate_feature_importance(features, success_score)
        })
        
    def get_strategy_recommendation(self, context_features):
        """Get recommended strategy based on historical performance"""
        scores = defaultdict(float)
        
        for strategy, history in self.strategy_history.items():
            if not history:
                continue
                
            # Calculate similarity-weighted success rate
            weighted_success = 0
            total_weight = 0
            
            for entry in history[-10:]:  # Consider recent history
                similarity = self._calculate_context_similarity(
                    context_features, 
                    entry['features']
                )
                weighted_success += similarity * entry['success']
                total_weight += similarity
                
            if total_weight > 0:
                scores[strategy] = weighted_success / total_weight
                
        return scores if scores else None
        
    def _calculate_feature_importance(self, features, success_score):
        """Calculate importance of different features for success"""
        importance = {}
        for feature, value in features.items():
            # Simple correlation between feature value and success
            importance[feature] = abs(float(value) - (1 - success_score))
        return importance
        
    def _calculate_context_similarity(self, features1, features2):
        """Calculate similarity between feature sets"""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
            
        similarities = []
        for feature in common_features:
            if isinstance(features1[feature], (int, float)) and \
               isinstance(features2[feature], (int, float)):
                diff = abs(features1[feature] - features2[feature])
                similarities.append(1.0 / (1.0 + diff))
            else:
                similarities.append(1.0 if features1[feature] == features2[feature] else 0.0)
                
        return np.mean(similarities)
