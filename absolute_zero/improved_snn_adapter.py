"""
Improved SNN Adapters for Absolute Zero framework.

This module provides improved adapters to connect the Absolute Zero framework
with the actual SNN implementations from the models/snn directory.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Union

# Add the parent directory to the Python path to import the SNN modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the actual SNN module (will be caught in try/except if not available)
try:
    from models.snn.statistical_snn import StatisticalSNN
    HAS_STATISTICAL_SNN = True
except ImportError as e:
    print(f"Warning: Could not import StatisticalSNN. Error: {e}")
    HAS_STATISTICAL_SNN = False


class ImprovedStatisticalSNNAdapter:
    """
    Improved adapter for the StatisticalSNN to interface with Absolute Zero.
    
    This adapter properly bridges between the Absolute Zero framework's expected 
    interface and the actual StatisticalSNN implementation.
    """
    
    def __init__(self, use_real_snn=True):
        """
        Initialize the adapter with option to use real or mock SNN.
        
        Args:
            use_real_snn: If True, use the actual StatisticalSNN implementation.
                          If False or if the SNN is not available, use a mock implementation.
        """
        self.use_real_snn = use_real_snn and HAS_STATISTICAL_SNN
        
        if self.use_real_snn:
            # Initialize the actual StatisticalSNN with proper parameters
            self.snn = StatisticalSNN(
                neuron_count=1000,  # Use the actual default from StatisticalSNN
                embedding_dim=300,  # Use the actual default from StatisticalSNN
                learning_rate=0.01
            )
            print("Using real StatisticalSNN implementation")
        else:
            # Use a mock implementation
            print("Using mock StatisticalSNN implementation")
            self.weights = np.random.randn(10, 10) * 0.1
            self.learning_rate = 0.01
    
    def process_input(self, input_vector: np.ndarray) -> Dict:
        """
        Process the input vector and return a prediction.
        
        This method adapts between the expected interface in Absolute Zero and
        the actual StatisticalSNN.process_input method.
        
        Args:
            input_vector: The input vector to process
            
        Returns:
            Dictionary with prediction results as expected by Absolute Zero
        """
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array([0.0] if not input_vector else input_vector, dtype=float)
            
        if self.use_real_snn:
            # Call the real StatisticalSNN.process_input method
            # The real SNN expects a query_type parameter, use default 'similarity'
            result = self.snn.process_input(input_vector, query_type='similarity')
            
            # Extract information from the result that's useful for Absolute Zero
            if isinstance(result, dict):
                # Find most similar concept if available
                similar_concepts = result.get('similar_concepts', [])
                if similar_concepts:
                    # Return the most similar concept and its similarity score
                    concept, score = similar_concepts[0]
                    return {
                        "next_element": hash(concept) % 32,  # Use hash as a numerical representation
                        "concept": concept,
                        "confidence": score
                    }
                
                # If no similar concepts found, use a default
                return {"next_element": 0, "confidence": 0.1}
            else:
                # Fallback if result is not in expected format
                return {"next_element": 0, "confidence": 0.1}
        else:
            # Simple mock processing
            if len(input_vector) < self.weights.shape[0]:
                input_vector = np.pad(input_vector, 
                                     (0, self.weights.shape[0] - len(input_vector)))
            else:
                input_vector = input_vector[:self.weights.shape[0]]
                
            result = np.dot(input_vector, self.weights)
            return {"next_element": int(np.sum(result) % 32)}
    
    def update_weights(self, input_vector: np.ndarray, prediction: Dict, reward: float):
        """
        Update the weights based on reward signal.
        
        Args:
            input_vector: The input vector that was processed
            prediction: The prediction that was made
            reward: The reward signal (higher is better)
        """
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array([0.0] if not input_vector else input_vector, dtype=float)
            
        if self.use_real_snn:
            # Apply feedback to the real SNN using the apply_feedback method
            # Convert prediction to active concept(s) if available
            active_concepts = []
            if isinstance(prediction, dict) and "concept" in prediction:
                active_concepts.append(prediction["concept"])
            
            # Call apply_feedback with the reward value and active concepts
            feedback_result = self.snn.apply_feedback(reward, active_concepts)
            
            # Log the result
            print(f"Updated StatisticalSNN with reward: {reward:.3f}")
            if isinstance(feedback_result, dict) and feedback_result.get('success'):
                modified = feedback_result.get('modified_concepts', [])
                if modified:
                    print(f"  - Modified concepts: {', '.join([c for c, _ in modified])}")
        else:
            # Process input vector for mock implementation
            if len(input_vector) < self.weights.shape[0]:
                input_vector = np.pad(input_vector, 
                                     (0, self.weights.shape[0] - len(input_vector)))
            else:
                input_vector = input_vector[:self.weights.shape[0]]
                
            # Simple update rule
            delta = reward * self.learning_rate
            self.weights += delta * np.outer(input_vector, np.ones(self.weights.shape[1]))
            print(f"Updated mock weights with reward: {reward:.3f}")
    
    def get_region_activations(self) -> np.ndarray:
        """
        Get the current activation state of the SNN regions.
        
        Returns:
            Array of activation values for different regions
        """
        if self.use_real_snn:
            # Extract region activations from the real SNN
            activations = []
            for region_name, region_info in self.snn.regions.items():
                activations.append(region_info.get('activation', 0.0))
            return np.array(activations)
        
        # Mock implementation
        return np.mean(self.weights, axis=1)
    
    def learn_concept(self, name: str, features: np.ndarray):
        """
        Learn a new concept with the given name and features.
        
        Args:
            name: The name of the concept to learn
            features: The feature vector for the concept
        """
        if self.use_real_snn and hasattr(self.snn, 'learn_concept_embedding'):
            self.snn.learn_concept_embedding(name, features)
            print(f"Learned new concept: {name}")


def create_improved_statistical_snn(use_real_snn=True) -> ImprovedStatisticalSNNAdapter:
    """
    Create an improved StatisticalSNNAdapter.
    
    Args:
        use_real_snn: Whether to use the real SNN implementation
        
    Returns:
        A configured StatisticalSNNAdapter instance
    """
    return ImprovedStatisticalSNNAdapter(use_real_snn) 