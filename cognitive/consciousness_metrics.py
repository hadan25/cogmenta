from typing import Dict, List, Optional
import numpy as np
import time

class ConsciousnessMetrics:
    """Collects and analyzes metrics related to consciousness-like behaviors"""
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.current_metrics: Dict = {}
        
    def calculate_information_complexity(self, states: List[Dict]) -> float:
        """Calculate complexity of information flow between states"""
        if not states:
            return 0.0
            
        # Calculate state transitions
        transitions = []
        for i in range(len(states) - 1):
            transition = self._calculate_state_difference(states[i], states[i+1])
            transitions.append(transition)
            
        # Calculate complexity using entropy and mutual information
        complexity = np.mean([self._calculate_entropy(t) for t in transitions])
        return complexity
        
    def _calculate_state_difference(self, state1: Dict, state2: Dict) -> Dict:
        """Calculate difference between two states"""
        diff = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            val1 = state1.get(key, 0)
            val2 = state2.get(key, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff[key] = abs(val2 - val1)
                
        return diff
        
    def _calculate_entropy(self, state: Dict) -> float:
        """Calculate entropy of a state"""
        values = [v for v in state.values() if isinstance(v, (int, float))]
        if not values:
            return 0.0
            
        # Normalize values
        total = sum(values)
        if total == 0:
            return 0.0
            
        probs = [v/total for v in values]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        return entropy

