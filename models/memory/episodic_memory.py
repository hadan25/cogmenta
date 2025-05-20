import time
from collections import deque
import re

class EpisodicMemory:
    def __init__(self, capacity=100):
        self.memory_buffer = deque(maxlen=capacity)
        self.importance_threshold = 0.3
        self.recent_activations = []  # Track recently activated memories
        
    def store_episode(self, content, importance=0.5):
        """Store an episodic memory with metadata"""
        episode = {
            'content': content,
            'timestamp': time.time(),
            'importance': importance,
            'recall_count': 0,
            'last_accessed': time.time()
        }
        self.memory_buffer.append(episode)
        return len(self.memory_buffer) - 1  # Return index
        
    def retrieve_relevant(self, query, limit=5):
        """Retrieve memories relevant to query"""
        # Simple relevance scoring (would use embeddings in real system)
        scored_memories = []
        for idx, memory in enumerate(self.memory_buffer):
            # Calculate relevance score
            relevance = self._calculate_relevance(query, memory)
            if relevance > self.importance_threshold:
                scored_memories.append((idx, memory, relevance))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Update retrieval metadata
        results = []
        self.recent_activations = []  # Reset recent activations
        
        for idx, memory, score in scored_memories[:limit]:
            # Update memory access data
            memory['recall_count'] += 1
            memory['last_accessed'] = time.time()
            results.append((memory['content'], score))
            self.recent_activations.append((memory['content'], score))
            
        return results
    
    def _calculate_relevance(self, query, memory):
        """Calculate relevance between query and memory"""
        query_lower = query.lower()
        memory_lower = memory['content'].lower()
        
        # For direct question-answer matching
        if '?' in query_lower:
            # Extract key entities from query
            query_entities = set()
            for entity in ['alice', 'bob', 'charlie', 'dave']:
                if entity in query_lower:
                    query_entities.add(entity)
            
            # Check if memory contains these entities
            matches = sum(1 for entity in query_entities if entity in memory_lower)
            if matches > 0:
                return 0.7 + (matches * 0.1)  # Higher relevance for more matching entities
        
        # Default calculation using word overlap
        query_words = set(query_lower.split())
        memory_words = set(memory_lower.split())
        
        intersection = len(query_words.intersection(memory_words))
        union = len(query_words.union(memory_words))
        
        base_similarity = intersection / union if union > 0 else 0
        
        # Boost relevance for certain types of queries
        if 'trust' in query_lower and 'trust' in memory_lower:
            base_similarity += 0.2
            
        return min(1.0, base_similarity)