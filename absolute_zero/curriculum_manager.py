import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import cosine

# Mock VSA engine for fallback
class MockVSAEngine:
    def encode(self, text):
        # Simple hash-based encoding for testing
        return np.array([hash(word) % 100 / 100.0 for word in text.split()])
    
    # Add encode_concept as an alias for encode to maintain compatibility with real VSA engine
    def encode_concept(self, text):
        return self.encode(text)
    
    def similarity(self, vec1, vec2):
        # Simple cosine similarity
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Make vectors the same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    # Add vector_similarity for direct vector comparison
    def vector_similarity(self, vec1, vec2):
        return self.similarity(vec1, vec2)

class VSACurriculum:
    def __init__(self, vsa_engine=None):
        self.vsa = vsa_engine if vsa_engine is not None else MockVSAEngine()
        self.task_history = []
        self.performance_history = []
        self.difficulty_level = 1.0
        
    def is_novel_task(self, task: Dict[str, Any]) -> bool:
        """Check if task is sufficiently different from recent history"""
        if not self.task_history:
            return True
        
        # Use encode_concept for real VSA engine or encode for mock engine
        if hasattr(self.vsa, 'encode_concept'):
            task_vector = self.vsa.encode_concept(str(task))
        else:
            task_vector = self.vsa.encode(str(task))
        
        for past_task in self.task_history[-100:]:
            # Use encode_concept for real VSA engine or encode for mock engine
            if hasattr(self.vsa, 'encode_concept'):
                past_vector = self.vsa.encode_concept(str(past_task))
            else:
                past_vector = self.vsa.encode(str(past_task))
            
            # Use direct vector similarity calculation
            similarity = self._calculate_vector_similarity(task_vector, past_vector)
            
            if similarity > 0.8:  # Too similar
                return False
        return True
    
    def _calculate_vector_similarity(self, vec1, vec2):
        """Calculate similarity between two vectors directly"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Make vectors the same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # Use cosine similarity (1 - cosine distance)
        try:
            return 1.0 - cosine(vec1, vec2)
        except:
            # Fallback to dot product similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
    
    def add_task_result(self, task: Dict, performance: Dict):
        """Record task and its performance"""
        self.task_history.append(task)
        self.performance_history.append(performance)
        
    def adjust_curriculum(self) -> str:
        """Adjust difficulty based on recent performance"""
        if len(self.performance_history) < 10:
            return 'maintain'
            
        recent_performance = self.performance_history[-10:]
        success_rate = np.mean([p['accuracy'] for p in recent_performance])
        
        if success_rate > 0.8:
            self.difficulty_level *= 1.1
            return 'increase_difficulty'
        elif success_rate < 0.4:
            self.difficulty_level *= 0.9
            return 'decrease_difficulty'
        
        return 'maintain'
    
    def get_difficulty_level(self) -> float:
        """Get current difficulty level"""
        return self.difficulty_level
        
    def set_adaptation_rate(self, rate: float):
        """Set the adaptation rate for curriculum adjustment"""
        # This would control how quickly the curriculum adapts
        # For now, just a stub implementation
        self.adaptation_rate = max(0.1, min(1.0, rate))
        
    def update(self, task_type: str, reward: float):
        """Update the curriculum based on task performance.
        
        Args:
            task_type: The type of task
            reward: The reward received
        """
        # Create a simple performance dictionary for tracking
        performance = {
            'task_type': task_type,
            'accuracy': reward,  # Using reward as a proxy for accuracy
            'timestamp': len(self.performance_history)
        }
        
        # Record performance
        self.performance_history.append(performance)
        
        # Adjust difficulty if we have enough data
        if len(self.performance_history) >= 10:
            recent_rewards = [p.get('accuracy', 0) for p in self.performance_history[-10:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            # Adjust difficulty based on recent performance
            if avg_reward > 0.7:
                # Doing well, increase difficulty
                self.difficulty_level = min(10.0, self.difficulty_level * 1.05)
            elif avg_reward < 0.3:
                # Struggling, decrease difficulty
                self.difficulty_level = max(0.5, self.difficulty_level * 0.95)
                
        return self.difficulty_level