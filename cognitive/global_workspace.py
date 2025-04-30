import numpy as np
import time
from collections import deque

class GlobalWorkspace:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.contents = deque(maxlen=capacity)
        self.broadcast_threshold = 0.5  # Lower threshold from 0.7
        self.attention_focus = None
        self.broadcast_history = []
        self.activation_decay = 0.95
        self.oscillation_frequency = 10  # Hz
        self.last_broadcast = time.time()
        
    def broadcast(self, content, strength):
        """Broadcast content if it exceeds threshold"""
        if strength >= self.broadcast_threshold:
            timestamp = time.time()
            
            # Check if enough time has passed (based on oscillation)
            if timestamp - self.last_broadcast >= (1.0 / self.oscillation_frequency):
                self.contents.append({
                    'content': content,
                    'strength': strength,
                    'timestamp': timestamp
                })
                self.broadcast_history.append({
                    'content': content,
                    'strength': strength,
                    'timestamp': timestamp
                })
                self.attention_focus = content
                self.last_broadcast = timestamp
                return True
        return False
        
    def get_current_focus(self):
        """Get current attention focus"""
        return self.attention_focus

    def update_activations(self):
        """Update activation strengths with decay"""
        for item in self.contents:
            item['strength'] *= self.activation_decay

    def get_workspace_state(self):
        """Get current state of the workspace"""
        return {
            'contents': list(self.contents),
            'focus': self.attention_focus,
            'broadcast_count': len(self.broadcast_history),
            'average_strength': np.mean([x['strength'] for x in self.contents]) if self.contents else 0
        }

    def update_attention(self, candidates, context=None):
        """Update attention focus based on candidates and context"""
        if not candidates:
            return None
            
        # Calculate attention scores
        scores = []
        for candidate in candidates:
            score = self._calculate_attention_score(candidate, context)
            scores.append((candidate, score))
            
        # Sort by score and get top candidate
        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidate, top_score = scores[0]
        
        # Only update if score exceeds threshold
        if top_score >= self.broadcast_threshold:
            self.broadcast(top_candidate, top_score)
            return top_candidate
            
        return None
        
    def _calculate_attention_score(self, candidate, context=None):
        """Calculate attention score for a candidate"""
        base_score = 0.5
        
        # Adjust based on existing content similarity
        if self.contents:
            similarity = self._calculate_similarity(candidate, self.contents[-1]['content'])
            base_score += similarity * 0.2
            
        # Consider context if provided
        if context:
            relevance = self._calculate_context_relevance(candidate, context)
            base_score += relevance * 0.3
            
        return min(1.0, base_score)

# Global Workspace exists with:
# - Broadcasting mechanism 
# - Attention focus tracking
# - Oscillatory control
# - Content management