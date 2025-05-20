"""
Improved Decision SNN Adapter for Absolute Zero framework.

This module provides an improved adapter to connect the Absolute Zero framework
with the DecisionSNN implementation from the models/snn directory.
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
    from models.snn.decision_snn import DecisionSNN
    HAS_DECISION_SNN = True
except ImportError as e:
    print(f"Warning: Could not import DecisionSNN. Error: {e}")
    HAS_DECISION_SNN = False


class ImprovedDecisionSNNAdapter:
    """
    Improved adapter for the DecisionSNN to interface with Absolute Zero.
    
    This adapter properly bridges between the Absolute Zero framework's expected 
    interface and the actual DecisionSNN implementation.
    """
    
    def __init__(self, use_real_snn=True):
        """
        Initialize the adapter with option to use real or mock SNN.
        
        Args:
            use_real_snn: If True, use the actual DecisionSNN implementation.
                          If False or if the SNN is not available, use a mock implementation.
        """
        self.use_real_snn = use_real_snn and HAS_DECISION_SNN
        
        if self.use_real_snn:
            # Initialize the actual DecisionSNN with default parameters
            self.snn = DecisionSNN(
                neuron_count=400,        # Default from DecisionSNN
                topology_type="scale_free"
            )
            print("Using real DecisionSNN implementation")
            
            # Define action types that map to core decision types for Absolute Zero
            self.action_mapping = {
                "factual_response": "exploit",
                "hypothesis_selection": "explore",
                "memory_retrieval": "recall",
                "clarification_request": "clarify",
                "verification_check": "verify",
                "uncertainty_expression": "adapt",
                "confident_assertion": "reinforce"
            }
            self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        else:
            # Mock implementation
            print("Using mock DecisionSNN implementation")
            self.actions = ["explore", "exploit", "reinforce", "adapt", "verify", "clarify", "recall"]
            self.confidence = 0.5
            self.last_action = None
            self.last_reward = 0.0
    
    def select_action(self, context: Dict) -> Dict:
        """
        Select an action based on context.
        
        This method adapts between the expected interface in Absolute Zero and
        the actual DecisionSNN.process_input method.
        
        Args:
            context: Decision context including task type, complexity, etc.
            
        Returns:
            Dictionary with action, confidence, and parameters
        """
        if not isinstance(context, dict):
            context = {}
            
        if self.use_real_snn:
            # Convert context to neural activation pattern
            input_activation = self._encode_context(context)
            
            # Process through the real DecisionSNN
            result = self.snn.process_input(input_activation)
            
            # Extract decision information from the activation pattern
            decision_info = self._extract_decision(result)
            
            # Get the best action based on activation
            internal_action = decision_info.get("action", "factual_response")
            
            # Map internal action to Absolute Zero action type
            az_action = self.action_mapping.get(internal_action, "exploit")
            
            # Format output as expected by Absolute Zero
            return {
                'action': az_action,
                'confidence': decision_info.get('confidence', 0.5),
                'params': decision_info.get('parameters', {})
            }
        else:
            # Mock implementation - simple decision logic
            task_complexity = float(context.get('complexity', 1.0))
            
            # Select action based on mock logic
            if task_complexity > 0.7:
                action = "adapt" if np.random.random() < 0.7 else "explore"
                confidence = 0.4 + 0.3 * np.random.random()
            else:
                action = "exploit" if np.random.random() < 0.8 else "reinforce"
                confidence = 0.6 + 0.3 * np.random.random()
                
            self.last_action = action
            self.confidence = confidence
            
            return {
                'action': action,
                'confidence': confidence,
                'params': {'intensity': 0.5 + 0.5 * np.random.random()}
            }
    
    def _encode_context(self, context: Dict) -> np.ndarray:
        """
        Encode the decision context as a neural activation pattern.
        
        Args:
            context: Decision context including task type, complexity, etc.
            
        Returns:
            Neural activation pattern for the DecisionSNN
        """
        # Initialize activation pattern with zeros
        activation = np.zeros(self.snn.neuron_count)
        
        # Set activation in different regions based on context
        
        # 1. Task type affects sensory region
        if 'sensory' in self.snn.regions:
            sensory_neurons = self.snn.regions['sensory']['neurons']
            task_type = context.get('task_type', 'unknown')
            
            # Apply different activation patterns based on task type
            if task_type == 'deduction':
                pattern = 0.8  # Strong activation for logical tasks
            elif task_type == 'abduction':
                pattern = 0.7  # Moderate-high for hypothetical reasoning
            elif task_type == 'induction':
                pattern = 0.6  # Moderate for pattern generalization
            elif task_type == 'pattern_recognition':
                pattern = 0.5  # Medium for pattern tasks
            else:
                pattern = 0.4  # Default activation
                
            # Apply to a subset of sensory neurons
            subset_size = min(len(sensory_neurons) // 2, 50)
            subset = sensory_neurons[:subset_size]
            activation[subset] = pattern
        
        # 2. Task complexity affects decision region
        if 'decision' in self.snn.regions:
            decision_neurons = self.snn.regions['decision']['neurons']
            complexity = float(context.get('complexity', 0.5))
            
            # Higher complexity activates different parts of the decision region
            # Scale to neuron indices
            mid_point = len(decision_neurons) // 2
            
            if complexity > 0.7:  # High complexity
                # Activate later neurons (complex decision patterns)
                subset = decision_neurons[mid_point:]
                activation[subset] = 0.6 + 0.2 * complexity
            else:  # Lower complexity
                # Activate earlier neurons (simpler decision patterns)
                subset = decision_neurons[:mid_point]
                activation[subset] = 0.4 + 0.2 * complexity
        
        # 3. Prior results affect integration region
        if 'integration' in self.snn.regions:
            integration_neurons = self.snn.regions['integration']['neurons']
            reasoning_result = context.get('reasoning_result', {})
            
            # If there are prior results, activate integration neurons
            if reasoning_result:
                # Convert dictionary size to activation strength (more info = stronger signal)
                strength = min(0.8, 0.2 + 0.1 * len(reasoning_result))
                activation[integration_neurons] = strength
        
        # Add some noise for biological plausibility
        activation += np.random.normal(0, 0.05, activation.shape)
        
        # Ensure values stay in reasonable range
        activation = np.clip(activation, 0, 1)
        
        return activation
    
    def _extract_decision(self, result: Dict) -> Dict:
        """
        Extract decision information from neural activation patterns.
        
        Args:
            result: Result from DecisionSNN processing
            
        Returns:
            Dictionary with action, confidence, and parameters
        """
        decision_info = {}
        
        # Get action channel activations
        action_activations = {}
        for channel_name, neurons in self.snn.action_channels.items():
            if not neurons:
                continue
                
            # Calculate activation level for this channel
            active_neurons = result.get('active_neurons', set())
            channel_active = set(neurons).intersection(active_neurons)
            activation_ratio = len(channel_active) / len(neurons) if neurons else 0
            
            # Also consider membrane potentials for neurons that didn't quite spike
            membrane_potentials = result.get('membrane_potentials', [])
            if len(membrane_potentials) == self.snn.neuron_count:
                avg_potential = np.mean(membrane_potentials[neurons]) if neurons else 0
                # Combine spike count and potential
                total_activation = (0.7 * activation_ratio) + (0.3 * avg_potential)
            else:
                total_activation = activation_ratio
                
            action_activations[channel_name] = total_activation
        
        # Select best action based on activation
        if action_activations:
            best_action = max(action_activations.items(), key=lambda x: x[1])
            action_name, activation_level = best_action
            
            # Calculate confidence as ratio between best and average
            mean_activation = np.mean(list(action_activations.values()))
            if mean_activation > 0:
                confidence = min(0.95, activation_level / mean_activation)
            else:
                confidence = 0.5  # Default if no activations
                
            decision_info["action"] = action_name
            decision_info["confidence"] = confidence
            
            # Add parameters based on activation patterns
            parameters = {}
            
            # Intensity parameter based on overall activation
            parameters["intensity"] = min(0.95, activation_level * 1.5)
            
            # Add any region-specific parameters
            if 'affective' in self.snn.regions:
                affective_activation = self.snn.regions['affective']['activation']
                parameters["emotional_influence"] = affective_activation
                
            decision_info["parameters"] = parameters
        else:
            # Fallback if no action channels activated
            decision_info["action"] = "factual_response"  # Default action
            decision_info["confidence"] = 0.3  # Low confidence
            decision_info["parameters"] = {"intensity": 0.3}
        
        return decision_info
    
    def update_policy(self, reward: float):
        """
        Update decision policy based on reward signal.
        
        Args:
            reward: The reward signal (higher is better)
        """
        if not isinstance(reward, (int, float)):
            reward = 0.0
            
        if self.use_real_snn:
            # The DecisionSNN needs an input state, target action, and reward for training
            # Since we don't have the original input state and target action, we'll use
            # a simplified approach focusing just on reward-based adjustment
            
            # Apply the reward signal to adjust biases in the network
            # This will influence future decisions by strengthening/weakening recent activations
            
            # 1. Create a simple neutral input pattern for training
            # Make sure this is a numpy array of the right shape
            neutral_input = np.ones(self.snn.neuron_count, dtype=np.float32) * 0.1
            
            # 2. Train with the reward signal
            target_action_idx = 0  # Default (factual_response)
            
            # If we have a last action, use that as the target
            if hasattr(self, 'last_action') and self.last_action:
                # Convert AZ action to internal action
                internal_action = self.reverse_mapping.get(self.last_action, "factual_response")
                
                # Get index of this action
                action_types = list(self.snn.action_channels.keys())
                if internal_action in action_types:
                    target_action_idx = action_types.index(internal_action)
            else:
                # Initialize last_action if it doesn't exist yet
                self.last_action = "exploit"  # Default action
                self.last_reward = 0.0
            
            # Train the network with appropriate learning rate based on reward magnitude
            learn_rate = 0.01 * (1.0 + abs(reward))
            
            # Set the SNN to training mode
            self.snn.training_mode = True
            
            try:
                # Train decision making with the reward signal
                self.snn.train_decision(
                    neutral_input,  # This should be a numpy array
                    target_action_idx,  # Integer index
                    confidence_target=max(0.1, min(0.9, 0.5 + 0.5 * reward)),
                    learn_rate=learn_rate
                )
                # Update last reward
                self.last_reward = reward
                print(f"Updated DecisionSNN with reward: {reward:.3f}")
            except Exception as e:
                print(f"Error updating DecisionSNN: {e}")
            finally:
                # Turn off training mode
                self.snn.training_mode = False
        else:
            # Mock implementation - track reward and adjust confidence
            if not hasattr(self, 'last_reward'):
                self.last_reward = 0.0
            
            delta = reward - self.last_reward
            self.confidence = min(0.95, max(0.1, self.confidence + 0.05 * delta))
            self.last_reward = reward
            print(f"Updated mock decision policy with reward: {reward:.3f}")
    
    def set_affective_state(self, affective_state: Dict):
        """
        Set affective state to modulate decision-making.
        
        Args:
            affective_state: Dictionary with affective state information
        """
        if not isinstance(affective_state, dict):
            affective_state = {}
            
        if self.use_real_snn and hasattr(self.snn, 'receive_affective_influence'):
            # Convert affective state to influence signal
            influence_signal = {}
            
            # Map valence and arousal to influence signal parameters
            valence = affective_state.get('valence', 0.0)
            arousal = affective_state.get('arousal', 0.5)
            
            # Set influence parameters
            influence_signal['modulation_type'] = 'affective'
            influence_signal['valence'] = valence
            influence_signal['arousal'] = arousal
            
            # Additional parameters based on specific emotions
            if 'emotion' in affective_state:
                influence_signal['emotion'] = affective_state['emotion']
            
            # Apply influence to the SNN
            self.snn.receive_affective_influence(influence_signal)
            print(f"Applied affective modulation to DecisionSNN: valence={valence:.2f}, arousal={arousal:.2f}")
        elif self.use_real_snn:
            # Alternative if receive_affective_influence not available
            print("DecisionSNN does not support direct affective modulation.")
        else:
            # Mock implementation - adjust behavior based on affective state
            valence = affective_state.get('valence', 0.0)
            arousal = affective_state.get('arousal', 0.5)
            
            # Higher arousal = more exploration
            explore_bias = 0.3 * arousal
            
            # Positive valence = more confidence
            self.confidence = min(0.95, max(0.1, self.confidence + 0.1 * valence))
            print(f"Applied mock affective modulation: valence={valence:.2f}, arousal={arousal:.2f}")


def create_improved_decision_snn(use_real_snn=True) -> ImprovedDecisionSNNAdapter:
    """
    Create an improved DecisionSNNAdapter.
    
    Args:
        use_real_snn: Whether to use the real SNN implementation
        
    Returns:
        A configured DecisionSNNAdapter instance
    """
    return ImprovedDecisionSNNAdapter(use_real_snn) 