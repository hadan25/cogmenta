#!/usr/bin/env python
"""
Test script for Absolute Zero.
This can be run directly from the absolute_zero directory.
"""

import sys
import os
import numpy as np
import random
from typing import Dict, List, Any

# Add the parent directory to Python path when running directly
if __name__ == "__main__":
    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Add to Python path
    sys.path.insert(0, parent_dir)

# Use direct import for when running as a script
try:
    from absolute_zero.zero_trainer import AbsoluteZeroTrainer
except ImportError:
    # Fallback to local import for when imported as a module
    from zero_trainer import AbsoluteZeroTrainer

# Mock SNN implementations for testing
class MockStatisticalSNN:
    def __init__(self):
        self.weights = np.random.randn(10, 10) * 0.1
        self.learning_rate = 0.01
        
    def process_input(self, input_vector):
        # Simple forward pass
        if len(input_vector) < self.weights.shape[0]:
            # Pad if needed
            input_vector = np.pad(input_vector, (0, self.weights.shape[0] - len(input_vector)))
        else:
            # Truncate if too long
            input_vector = input_vector[:self.weights.shape[0]]
            
        # Simple transformation - just a basic linear operation for testing
        result = np.dot(input_vector, self.weights)
        
        # For pattern tasks, return the expected format
        if isinstance(result, np.ndarray) and len(result) > 0:
            return {"next_element": int(result[0])}
        return result
    
    def update_weights(self, input_vector, prediction, reward):
        # Basic reward-based update
        if len(input_vector) < self.weights.shape[0]:
            input_vector = np.pad(input_vector, (0, self.weights.shape[0] - len(input_vector)))
        else:
            input_vector = input_vector[:self.weights.shape[0]]
            
        # Simple update rule
        delta = reward * self.learning_rate
        self.weights += delta * np.outer(input_vector, np.ones(self.weights.shape[1]))
        print(f"Updated weights with reward: {reward:.3f}")
    
    def get_region_activations(self):
        # Mock activations
        return np.mean(self.weights, axis=1)

class MockAffectiveSNN:
    def evaluate_affective_state(self, metrics):
        return {
            "valence": metrics["sentiment"],
            "arousal": metrics["intensity"]
        }
    
    def influence_processing(self, stat_snn):
        # In a real implementation, this would modulate learning
        pass
    
    def get_emotion_state(self):
        return {"valence": random.random(), "arousal": random.random()}

class MockMetacognitiveSNN:
    def monitor_system_state(self, state_info):
        # In a real implementation, this would track system performance
        pass
    
    def get_metacognitive_state(self):
        return {"confidence": random.random(), "uncertainty": random.random()}

# Mock symbolic engines
class MockSymbolicEngine:
    def deduce(self, rules, facts):
        # Very simple mock implementation
        if "parent(X,Y) :- father(X,Y)" in rules and "father(john, bob)" in facts:
            return "parent(john, bob)"
        return "no_conclusion"
    
    def verify_conclusion(self, rules, facts, conclusion):
        return conclusion == self.deduce(rules, facts)

class MockVSAEngine:
    def encode(self, text):
        # Simple hash-based encoding for testing
        return np.array([hash(word) % 100 / 100.0 for word in text.split()])
    
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

def test_absolute_zero():
    # Initialize mock components
    snns = {
        'statistical': MockStatisticalSNN(),
        'affective': MockAffectiveSNN(),
        'metacognitive': MockMetacognitiveSNN()
    }
    
    symbolic_engine = MockSymbolicEngine()
    vsa_engine = MockVSAEngine()
    
    # Initialize the trainer
    trainer = AbsoluteZeroTrainer(snns, symbolic_engine, vsa_engine)
    
    # Run a short training session
    print("Starting test training with mock components...")
    trainer.train(iterations=10, log_interval=2)
    print("Test training completed.")
    
    # You could add assertions here to verify expected behavior

if __name__ == "__main__":
    test_absolute_zero() 