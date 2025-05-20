"""
Improved Affective SNN Adapter for Absolute Zero framework.

This module provides an improved adapter to connect the Absolute Zero framework
with the AffectiveSNN implementation from the models/snn directory.
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
    from models.snn.affective_snn import AffectiveSNN
    HAS_AFFECTIVE_SNN = True
except ImportError as e:
    print(f"Warning: Could not import AffectiveSNN. Error: {e}")
    HAS_AFFECTIVE_SNN = False


class ImprovedAffectiveSNNAdapter:
    """
    Improved adapter for the AffectiveSNN to interface with Absolute Zero.
    
    This adapter properly bridges between the Absolute Zero framework's expected 
    interface and the actual AffectiveSNN implementation.
    """
    
    def __init__(self, use_real_snn=True):
        """
        Initialize the adapter with option to use real or mock SNN.
        
        Args:
            use_real_snn: If True, use the actual AffectiveSNN implementation.
                          If False or if the SNN is not available, use a mock implementation.
        """
        self.use_real_snn = use_real_snn and HAS_AFFECTIVE_SNN
        
        if self.use_real_snn:
            # Initialize the actual AffectiveSNN with default parameters
            self.snn = AffectiveSNN(
                neuron_count=400,  # Default from AffectiveSNN
                topology_type="flexible"
            )
            print("Using real AffectiveSNN implementation")
            
            # Store current affective state
            self.current_state = {
                "valence": 0.0,
                "arousal": 0.5,
                "emotion": None
            }
        else:
            # Mock implementation
            print("Using mock AffectiveSNN implementation")
            self.current_state = {
                "valence": 0.0,
                "arousal": 0.5,
                "emotion": None
            }
    
    def evaluate_affective_state(self, metrics: Dict) -> Dict:
        """
        Evaluate the affective state based on metrics.
        
        This method adapts between the expected interface in Absolute Zero and
        the actual AffectiveSNN.evaluate_affective_state method.
        
        Args:
            metrics: Dictionary with performance metrics
            
        Returns:
            Dictionary with affective state information
        """
        if not isinstance(metrics, dict):
            metrics = {}
            
        if self.use_real_snn:
            try:
                # Convert metrics to the format expected by AffectiveSNN
                input_features = self._convert_metrics_to_features(metrics)
                
                # Ensure input_features is a dict, not a tuple
                if not isinstance(input_features, dict):
                    input_features = {"sentiment": 0.0, "intensity": 0.5}
                
                # Get affective state evaluation from the real AffectiveSNN
                affective_state = self.snn.evaluate_affective_state(input_features)
                
                # Store current state for later reference
                self.current_state = affective_state.copy()
                
                # Format output as expected by Absolute Zero
                return {
                    "valence": affective_state.get("valence", 0.0),
                    "arousal": affective_state.get("arousal", 0.5),
                    "emotion": affective_state.get("emotion", None),
                    "confidence": affective_state.get("confidence", 0.5)
                }
            except Exception as e:
                print(f"Error in AffectiveSNN.evaluate_affective_state: {e}")
                # Fall back to mock implementation
                self.use_real_snn = False
                return self.evaluate_affective_state(metrics)
        else:
            # Mock implementation
            # Map sentiment (accuracy-based) to valence
            valence = metrics.get("sentiment", 0.0)
            if "accuracy" in metrics and "sentiment" not in metrics:
                valence = metrics["accuracy"] * 2 - 1.0  # Convert [0, 1] to [-1, 1]
            
            # Map intensity to arousal
            arousal = metrics.get("intensity", 0.5)
            if "combined_reward" in metrics and "intensity" not in metrics:
                arousal = min(1.0, max(0.0, 0.5 + abs(metrics["combined_reward"]) * 0.5))
            
            # Determine primary emotion based on valence-arousal
            emotion = self._mock_determine_emotion(valence, arousal)
            
            # Update and return current state
            self.current_state = {
                "valence": valence,
                "arousal": arousal,
                "emotion": emotion,
                "confidence": 0.7  # Fixed confidence for mock
            }
            
            return self.current_state
    
    def _convert_metrics_to_features(self, metrics: Dict) -> Dict:
        """
        Convert Absolute Zero metrics to features expected by AffectiveSNN.
        
        Args:
            metrics: Metrics from Absolute Zero framework
            
        Returns:
            Features dictionary for AffectiveSNN
        """
        features = {}
        
        # Map accuracy to sentiment (valence)
        if "accuracy" in metrics:
            features["sentiment"] = metrics["accuracy"] - 0.5  # Convert to range [-0.5, 0.5]
        elif "sentiment" in metrics:
            features["sentiment"] = metrics["sentiment"]
        else:
            features["sentiment"] = 0.0
            
        # Map combined_reward or intensity to arousal
        if "combined_reward" in metrics:
            features["intensity"] = abs(metrics["combined_reward"])
        elif "intensity" in metrics:
            features["intensity"] = metrics["intensity"]
        else:
            features["intensity"] = 0.5
            
        # Copy any additional features that might be useful
        for key in ["novelty", "success_rate", "task_type", "difficulty"]:
            if key in metrics:
                features[key] = metrics[key]
                
        # If there's a clear frustration or satisfaction signal, include it
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if accuracy < 0.3:
                features["frustration"] = 0.7
            elif accuracy > 0.8:
                features["satisfaction"] = 0.8
                
        return features
    
    def _mock_determine_emotion(self, valence: float, arousal: float) -> str:
        """
        Determine emotion from valence-arousal coordinates for mock implementation.
        
        Args:
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            
        Returns:
            String name of the primary emotion
        """
        # Simple emotion classification based on valence-arousal space
        if valence > 0.3:
            if arousal > 0.7:
                return "joy"
            elif arousal > 0.3:
                return "anticipation"
            else:
                return "trust"
        elif valence < -0.3:
            if arousal > 0.7:
                return "anger"
            elif arousal > 0.5:
                return "fear"
            elif arousal > 0.3:
                return "disgust"
            else:
                return "sadness"
        else:
            # Neutral valence
            if arousal > 0.7:
                return "surprise"
            else:
                return "neutral"
    
    def influence_processing(self, statistical_snn):
        """
        Apply affective influence to learning in another component.
        
        Args:
            statistical_snn: The component to influence
        """
        if not self.use_real_snn:
            # Mock implementation
            valence = self.current_state.get("valence", 0.0)
            arousal = self.current_state.get("arousal", 0.5)
            
            # Apply simple modulation to statistical_snn
            if hasattr(statistical_snn, "use_real_snn") and statistical_snn.use_real_snn:
                # If statistical_snn has a real SNN
                learning_rate_mod = 1.0 + (valence * 0.2)
                exploration_mod = 1.0 + (arousal * 0.3)
                
                if hasattr(statistical_snn.snn, "set_modulation"):
                    statistical_snn.snn.set_modulation({
                        "learning_rate_mod": learning_rate_mod,
                        "exploration_mod": exploration_mod
                    })
                    print(f"Applied mock affective modulation: learning={learning_rate_mod:.2f}, exploration={exploration_mod:.2f}")
            return
            
        # Real implementation
        if not hasattr(self.snn, "influence_processing"):
            print("AffectiveSNN does not support influence_processing")
            return
        
        # Check if the target component has a real SNN
        if not hasattr(statistical_snn, "use_real_snn") or not statistical_snn.use_real_snn:
            print("Cannot influence mock SNN with real AffectiveSNN")
            return
            
        # Apply influence to the target component
        target_component = statistical_snn.snn
        self.snn.influence_processing(target_component)
        
        # Log the influence
        current_state = self.current_state
        print(f"Applied affective modulation: valence={current_state.get('valence', 0.0):.2f}, " +
              f"arousal={current_state.get('arousal', 0.5):.2f}, " +
              f"emotion={current_state.get('emotion', 'unknown')}")
    
    def get_emotion_state(self) -> Dict:
        """
        Get the current emotional state.
        
        Returns:
            Dictionary with current emotional state
        """
        if self.use_real_snn:
            # Get the current emotion state from the real AffectiveSNN
            # The actual implementation might have a specific method for this
            # but we'll use our stored state as a fallback
            if hasattr(self.snn, "get_current_emotion"):
                return self.snn.get_current_emotion()
            else:
                return self.current_state
        else:
            # Mock implementation - return stored state
            return self.current_state
    
    def train_emotion(self, input_features: Dict, target_emotion: str, reward: float = None):
        """
        Train the AffectiveSNN to recognize and respond to emotional patterns.
        
        Args:
            input_features: Input features for training
            target_emotion: Target emotion to train
            reward: Optional reward signal
        """
        if not self.use_real_snn:
            print("Training not available for mock AffectiveSNN")
            return
            
        # Check if the SNN has the training method
        if not hasattr(self.snn, "train_emotion_assembly"):
            print("AffectiveSNN does not support train_emotion_assembly")
            return
            
        # Convert features if needed
        if not isinstance(input_features, dict):
            input_features = {}
            
        # Determine learning rate based on reward
        learn_rate = 0.02  # Default
        if reward is not None:
            learn_rate = max(0.01, min(0.05, 0.02 + 0.03 * abs(reward)))
            
        # Train the emotion assembly
        self.snn.train_emotion_assembly(input_features, target_emotion, learn_rate)
        print(f"Trained AffectiveSNN on emotion: {target_emotion} with learn_rate={learn_rate:.3f}")


def create_improved_affective_snn(use_real_snn=True) -> ImprovedAffectiveSNNAdapter:
    """
    Create an improved AffectiveSNNAdapter.
    
    Args:
        use_real_snn: Whether to use the real SNN implementation
        
    Returns:
        A configured AffectiveSNNAdapter instance
    """
    return ImprovedAffectiveSNNAdapter(use_real_snn) 