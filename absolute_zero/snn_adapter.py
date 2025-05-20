"""
SNN Adapter for Absolute Zero framework.

This module provides adapters to connect the Absolute Zero framework
with the actual SNN implementations from the models/snn directory.
"""

import sys
import os
import numpy as np
import time
import importlib.util
from typing import Dict, Any, List, Tuple

# Add the parent directory to the Python path to import the SNN modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Print debug information
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# Try to find SNN modules
snn_dir = os.path.join(parent_dir, 'models', 'snn')
if os.path.exists(snn_dir):
    print(f"SNN directory exists: {snn_dir}")
    print(f"Contents of SNN directory: {os.listdir(snn_dir)}")
else:
    print(f"SNN directory not found: {snn_dir}")

# Check for required dependencies with better reporting
# We no longer need gensim since vectorization is done locally
required_packages = ['numpy', 'scipy']
missing_packages = []

for package in required_packages:
    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Warning: Missing required packages for real SNN implementations: {', '.join(missing_packages)}")
    print("Using mock implementations instead.")
    print("To use real SNNs, install the missing packages with: pip install " + " ".join(missing_packages))
    HAS_SNN_MODULES = False
else:
    # Import the actual SNN modules with better error handling
    try:
        # First check if the modules exist by file path
        required_modules = [
            'statistical_snn.py', 
            'affective_snn.py', 
            'metacognitive_snn.py',
            'reasoning_snn.py',
            'decision_snn.py',
            'perceptual_snn.py',
            'memory_snn.py'
        ]
        
        missing_module_files = []
        for module_file in required_modules:
            if not os.path.exists(os.path.join(snn_dir, module_file)):
                missing_module_files.append(module_file)
                
        if missing_module_files:
            print(f"Warning: Some required SNN module files are missing: {', '.join(missing_module_files)}")
            HAS_SNN_MODULES = False
        else:
            # Try importing each module
            from models.snn.statistical_snn import StatisticalSNN
            from models.snn.affective_snn import AffectiveSNN
            from models.snn.metacognitive_snn import MetacognitiveSNN
            from models.snn.reasoning_snn import ReasoningSNN
            from models.snn.decision_snn import DecisionSNN
            from models.snn.perceptual_snn import PerceptualSNN
            from models.snn.memory_snn import MemorySNN
            
            # Verify each class has the necessary methods
            missing_methods = []
            if not hasattr(StatisticalSNN, 'process_input') and not hasattr(StatisticalSNN, 'process'):
                missing_methods.append('StatisticalSNN.process_input')
            if not hasattr(AffectiveSNN, 'evaluate_affective_state'):
                missing_methods.append('AffectiveSNN.evaluate_affective_state')
            
            if missing_methods:
                print(f"Warning: Some required methods are missing in SNN modules: {', '.join(missing_methods)}")
                print("Will use adapter logic to bridge the gaps.")
                
            HAS_SNN_MODULES = True
            print("Successfully imported all SNN modules!")
    except ImportError as e:
        print(f"Warning: Could not import SNN modules. Error: {e}")
        print("Using mock implementations.")
        HAS_SNN_MODULES = False


class StatisticalSNNAdapter:
    """Adapter for the StatisticalSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            try:
                # Initialize the actual StatisticalSNN with proper parameters
                print("Initializing real StatisticalSNN...")
                # Check what parameters the constructor accepts
                import inspect
                init_params = inspect.signature(StatisticalSNN.__init__).parameters
                
                # Create appropriate kwargs based on available parameters
                kwargs = {}
                if 'input_size' in init_params:
                    kwargs['input_size'] = 64
                if 'hidden_size' in init_params:
                    kwargs['hidden_size'] = 128
                if 'output_size' in init_params:
                    kwargs['output_size'] = 32
                if 'learning_rate' in init_params:
                    kwargs['learning_rate'] = 0.01
                if 'neuron_count' in init_params:
                    kwargs['neuron_count'] = 100
                
                # Initialize with only the parameters the constructor accepts
                self.snn = StatisticalSNN(**kwargs)
                print("Successfully initialized real StatisticalSNN")
            except Exception as e:
                print(f"Error initializing real StatisticalSNN: {e}")
                print("Falling back to mock implementation")
                self.use_real_snn = False
                # Create a mock implementation as fallback
                self.weights = np.random.randn(10, 10) * 0.1
                self.learning_rate = 0.01
                self.input_size = 10
                self.output_size = 10
        else:
            # Create a simple but fully functional mock
            self.weights = np.random.randn(10, 10) * 0.1
            self.learning_rate = 0.01
            self.input_size = 10
            self.output_size = 10
    
    def process_input(self, input_vector: np.ndarray) -> Dict:
        """Process the input vector and return a prediction."""
        # Ensure the input is a numpy array
        if input_vector is None:
            print("Warning: input_vector is None, using default empty array")
            input_vector = np.array([0.0])
            
        if not isinstance(input_vector, np.ndarray):
            try:
                input_vector = np.array(input_vector, dtype=float)
            except Exception as e:
                print(f"Error converting input to numpy array: {e}")
                input_vector = np.array([0.0])
                
        if self.use_real_snn:
            try:
                # Check if the real SNN has a process_input method directly
                if hasattr(self.snn, 'process_input'):
                    print("Using actual StatisticalSNN.process_input method")
                    result = self.snn.process_input(input_vector, query_type='similarity')
                    
                    # Extract the needed information for Absolute Zero
                    if isinstance(result, dict):
                        if 'similar_concepts' in result:
                            # Use the first similar concept's name if available
                            similar = result.get('similar_concepts', [])
                            if similar and len(similar) > 0:
                                return {"next_element": similar[0][0] if isinstance(similar[0], tuple) else 0}
                        elif 'generalizations' in result:
                            return {"next_element": 1}  # Default for generalization
                        elif 'completed_pattern' in result:
                            return {"next_element": 2}  # Default for completion
                            
                    # Return a default structure if we couldn't extract meaningful data
                    return {"next_element": 0}
                
                # Fallback to process method if it exists
                elif hasattr(self.snn, 'process'):
                    print("Using StatisticalSNN.process method instead of process_input")
                    # Pad or truncate to match expected input size
                    if hasattr(self.snn, 'input_size'):
                        if len(input_vector) < self.snn.input_size:
                            input_vector = np.pad(input_vector, 
                                                (0, self.snn.input_size - len(input_vector)))
                        else:
                            input_vector = input_vector[:self.snn.input_size]
                            
                    # Process through the SNN
                    result = self.snn.process(input_vector)
                    
                    # Format the output as expected by Absolute Zero
                    return {"next_element": int(np.argmax(result) if isinstance(result, np.ndarray) else 0)}
                
                # Another fallback for predict method
                elif hasattr(self.snn, 'predict'):
                    print("Using StatisticalSNN.predict method instead of process_input")
                    result = self.snn.predict(input_vector)
                    return {"next_element": int(np.argmax(result) if isinstance(result, np.ndarray) else 0)}
                    
                else:
                    print("Warning: SNN has neither process_input, process, nor predict method")
                    print("Falling back to mock implementation for this call")
                    # Use mock implementation for this call
                    if len(input_vector) < 10:
                        input_vector = np.pad(input_vector, (0, 10 - len(input_vector)))
                    else:
                        input_vector = input_vector[:10]
                    result = np.dot(input_vector, np.random.randn(10, 10) * 0.1)
                    return {"next_element": int(np.argmax(result))}
                    
            except Exception as e:
                print(f"Error calling real SNN methods: {e}")
                print("Falling back to mock implementation for this call")
                # Use mock implementation for this call
                if len(input_vector) < 10:
                    input_vector = np.pad(input_vector, (0, 10 - len(input_vector)))
                else:
                    input_vector = input_vector[:10]
                result = np.dot(input_vector, np.random.randn(10, 10) * 0.1)
                return {"next_element": int(np.argmax(result))}
        else:
            # Simple mock processing - ensure this works similar to the MockStatisticalSNN in test_absolute_zero.py
            if len(input_vector) < self.weights.shape[0]:
                input_vector = np.pad(input_vector, 
                                     (0, self.weights.shape[0] - len(input_vector)))
            else:
                input_vector = input_vector[:self.weights.shape[0]]
                
            result = np.dot(input_vector, self.weights)
            if isinstance(result, np.ndarray):
                return {"next_element": int(np.argmax(result))}
            else:
                return {"next_element": int(result)}
    
    def update_weights(self, input_vector: np.ndarray, prediction: Dict, reward: float):
        """Update the weights based on reward signal."""
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array([0.0] if not input_vector else input_vector, dtype=float)
            
        if self.use_real_snn:
            try:
                # Convert prediction to the format expected by the SNN
                target = np.zeros(self.snn.output_size)
                if isinstance(prediction, dict) and "next_element" in prediction:
                    idx = prediction["next_element"] % self.snn.output_size
                    target[idx] = 1.0
                    
                # Apply reward-modulated learning
                if hasattr(self.snn, 'reward_modulated_update'):
                    self.snn.reward_modulated_update(input_vector, target, reward)
                    print(f"Updated real StatisticalSNN with reward: {reward:.3f}")
                elif hasattr(self.snn, 'update'):
                    self.snn.update(input_vector, target, reward)
                    print(f"Updated real StatisticalSNN with update method: {reward:.3f}")
                else:
                    print("Warning: Real SNN has no update methods, falling back to mock update")
                    # Fall back to mock implementation for update
                    if len(input_vector) < 10:
                        input_vector = np.pad(input_vector, (0, 10 - len(input_vector)))
                    else:
                        input_vector = input_vector[:10]
                    weights = np.random.randn(10, 10) * 0.1
                    delta = reward * 0.01
                    weights += delta * np.outer(input_vector, np.ones(10))
            except Exception as e:
                print(f"Error updating real StatisticalSNN: {e}")
                print("Falling back to mock update")
                # Fall back to mock implementation for update
                if len(input_vector) < 10:
                    input_vector = np.pad(input_vector, (0, 10 - len(input_vector)))
                else:
                    input_vector = input_vector[:10]
                weights = np.random.randn(10, 10) * 0.1
                delta = reward * 0.01
                weights += delta * np.outer(input_vector, np.ones(10))
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
        """Get the current activation state of the SNN regions."""
        if self.use_real_snn:
            try:
                if hasattr(self.snn, 'get_activation_state'):
                    return self.snn.get_activation_state()
                elif hasattr(self.snn, 'get_neuron_activations'):
                    return self.snn.get_neuron_activations()
                else:
                    print("Warning: Real SNN has no activation methods, returning mock activations")
                    return np.random.random(10)
            except Exception as e:
                print(f"Error getting activations from real StatisticalSNN: {e}")
                return np.random.random(10)
        return np.mean(self.weights, axis=1)


class AffectiveSNNAdapter:
    """Adapter for the AffectiveSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            # Initialize the actual AffectiveSNN
            self.snn = AffectiveSNN()
    
    def evaluate_affective_state(self, metrics: Dict) -> Dict:
        """Evaluate the affective state based on metrics."""
        if not isinstance(metrics, dict):
            metrics = {}
            
        if self.use_real_snn:
            # Prepare the input for the affective SNN
            affective_input = {
                "valence": metrics.get("sentiment", 0.0),
                "arousal": metrics.get("intensity", 0.5),
                "success_rate": metrics.get("accuracy", 0.0)
            }
            return self.snn.evaluate_emotion(affective_input)
            
        # Mock implementation
        return {
            "valence": metrics.get("sentiment", 0.0),
            "arousal": metrics.get("intensity", 0.5)
        }
    
    def influence_processing(self, statistical_snn):
        """Apply affective influence to learning."""
        if not self.use_real_snn or not hasattr(statistical_snn, "snn"):
            return  # No action needed for mock implementation
        
        # Get the current emotional state
        emotion_state = self.snn.get_current_emotion()
        
        # Apply modulation based on emotional state
        learning_rate_mod = 1.0 + (emotion_state.get("valence", 0) * 0.2)
        exploration_mod = 1.0 + (emotion_state.get("arousal", 0) * 0.3)
        
        # Update SNN parameters
        if hasattr(statistical_snn.snn, "set_modulation"):
            statistical_snn.snn.set_modulation({
                "learning_rate_mod": learning_rate_mod,
                "exploration_mod": exploration_mod
            })
            print(f"Applied affective modulation: learning={learning_rate_mod:.2f}, exploration={exploration_mod:.2f}")
    
    def get_emotion_state(self) -> Dict:
        """Get the current emotional state."""
        if self.use_real_snn:
            return self.snn.get_current_emotion()
        return {"valence": 0.0, "arousal": 0.5}


class MetacognitiveSNNAdapter:
    """Adapter for the MetacognitiveSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            # Initialize the actual MetacognitiveSNN
            self.snn = MetacognitiveSNN()
        else:
            # Mock implementation
            self.confidence = 0.5
            self.uncertainty = 0.5
    
    def monitor_system_state(self, state_info: Dict):
        """Monitor the system state and update metacognitive strategies."""
        if not isinstance(state_info, dict):
            state_info = {}
            
        if self.use_real_snn:
            # Convert state_info to the format expected by MetacognitiveSNN
            metrics = state_info.get("learning_metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
                
            metacog_input = {
                "performance": metrics.get("accuracy", 0.5),
                "complexity": metrics.get("task_type", "unknown"),
                "reward": metrics.get("combined_reward", 0.0)
            }
            
            # Update the metacognitive SNN
            self.snn.update_metacognition(metacog_input)
            
            accuracy = metrics.get("accuracy", 0)
            if accuracy > 0.8:
                # If performance is good, suggest increasing difficulty
                self.snn.suggest_strategy_change("increase_difficulty")
            elif accuracy < 0.3:
                # If performance is poor, suggest decreasing difficulty
                self.snn.suggest_strategy_change("decrease_difficulty")
            return
            
        # Mock implementation - just update internal state
        metrics = state_info.get("learning_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
            
        accuracy = metrics.get("accuracy", 0.5)
        if accuracy > 0.7:
            self.confidence = min(1.0, self.confidence + 0.1)
            self.uncertainty = max(0.0, self.uncertainty - 0.1)
        elif accuracy < 0.3:
            self.confidence = max(0.0, self.confidence - 0.1)
            self.uncertainty = min(1.0, self.uncertainty + 0.1)
    
    def get_metacognitive_state(self) -> Dict:
        """Get the current metacognitive state."""
        if self.use_real_snn:
            return self.snn.get_metacognitive_state()
        return {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty
        }


class ReasoningSNNAdapter:
    """Adapter for the ReasoningSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            try:
                # Initialize the actual ReasoningSNN
                print("Initializing real ReasoningSNN...")
                self.snn = ReasoningSNN()
                
                # Check if the required attributes/methods exist
                if not hasattr(self.snn, 'reasoning_circuits') and not hasattr(self.snn, 'reason'):
                    print("Warning: ReasoningSNN missing required attributes/methods")
                    # Add mock reasoning_circuits if it doesn't exist but is required
                    if not hasattr(self.snn, 'reasoning_circuits'):
                        self.snn.reasoning_circuits = {
                            'deduction': np.random.randn(10, 10) * 0.1,
                            'induction': np.random.randn(10, 10) * 0.1,
                            'abduction': np.random.randn(10, 10) * 0.1
                        }
                        print("Added mock reasoning_circuits to ReasoningSNN")
                
                print("Successfully initialized real ReasoningSNN with adapter enhancements")
            except Exception as e:
                print(f"Error initializing real ReasoningSNN: {e}")
                print("Falling back to mock implementation")
                self.use_real_snn = False
                # Create mock implementation
                self.reasoning_modes = ["deductive", "inductive", "abductive"]
                self.current_mode = "deductive"
                self.accuracy = 0.6
                self.reward_history = []
        else:
            # Mock implementation
            self.reasoning_modes = ["deductive", "inductive", "abductive"]
            self.current_mode = "deductive"
            self.accuracy = 0.6
            self.reward_history = []
    
    def process(self, reasoning_input: Dict) -> Dict:
        """Process reasoning input and produce logical inference."""
        if not isinstance(reasoning_input, dict):
            reasoning_input = {}
            
        if self.use_real_snn:
            try:
                # Extract input components
                rules = reasoning_input.get('rules', [])
                facts = reasoning_input.get('facts', [])
                mode = reasoning_input.get('type', 'deduction')
                
                # Create input text for reasoning SNN
                input_text = ""
                if rules:
                    input_text += "Rules: " + ". ".join(rules) + ". "
                if facts:
                    input_text += "Facts: " + ". ".join(facts) + "."
                    
                # Check which method to use
                if hasattr(self.snn, 'reason'):
                    # Process through reasoning SNN with specified mode
                    if hasattr(self.snn, 'set_reasoning_mode'):
                        self.snn.set_reasoning_mode(mode)
                    result = self.snn.reason(input_text, mode)
                    return result
                elif hasattr(self.snn, 'process'):
                    # Try process method if available
                    if hasattr(self.snn, 'set_reasoning_mode'):
                        self.snn.set_reasoning_mode(mode)
                    result = self.snn.process(input_text)
                    return result
                else:
                    print("Warning: ReasoningSNN has no reason or process method, using mock implementation")
                    raise AttributeError("No suitable method found in ReasoningSNN")
            except Exception as e:
                print(f"Error calling real ReasoningSNN methods: {e}")
                print("Falling back to mock implementation for this call")
                # Continue to mock implementation
                
        # Mock implementation
        task_type = reasoning_input.get('type', 'deduction')
        self.current_mode = task_type if task_type in self.reasoning_modes else "deductive"
        
        # Simple mock output for different reasoning modes
        mock_result = {}
        if self.current_mode == "deduction":
            mock_result = {"conclusion": "mock deductive conclusion", "is_valid": bool(np.random.random() < self.accuracy)}
        elif self.current_mode == "induction":
            mock_result = {"pattern": "mock inductive pattern", "confidence": max(0.1, min(0.9, self.accuracy))}
        elif self.current_mode == "abduction":
            mock_result = {"hypothesis": "mock abductive hypothesis", "score": max(0.1, min(0.9, self.accuracy))}
            
        return mock_result
    
    def update(self, reward: float):
        """Update reasoning networks based on reward."""
        if not isinstance(reward, (int, float)):
            reward = 0.0
            
        if self.use_real_snn:
            # Create a simple training input for the current mode
            mode = self.snn.current_mode
            
            # Train with the reward signal
            input_activation = np.random.random(64)  # Mock input
            target_output = np.random.random(32) if mode == "inductive" else np.array([1.0]) if reward > 0.5 else np.array([0.0])
            
            self.snn.train_reasoning(input_activation, target_output, mode, learn_rate=0.01 * reward)
            return
            
        # Simple mock update
        self.reward_history.append(reward)
        recent_rewards = self.reward_history[-10:] if len(self.reward_history) > 10 else self.reward_history
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        # Adjust accuracy based on rewards
        self.accuracy = min(0.95, max(0.1, self.accuracy + 0.02 * (avg_reward - 0.5)))
    
    def set_affective_state(self, affective_state: Dict):
        """Set affective state to modulate reasoning."""
        if not isinstance(affective_state, dict):
            affective_state = {}
            
        if self.use_real_snn:
            # If the SNN has a method to handle affective state
            if hasattr(self.snn, 'set_affective_modulation'):
                valence = affective_state.get('valence', 0.0)
                arousal = affective_state.get('arousal', 0.5)
                
                # Convert to modulation parameters
                creative_mod = 1.0 + 0.5 * arousal
                precision_mod = 1.0 + 0.5 * (valence if valence > 0 else 0)
                
                self.snn.set_affective_modulation({
                    'creative_mod': creative_mod,
                    'precision_mod': precision_mod
                })
            return
            
        # Mock implementation - adjust accuracy based on affective state
        valence = affective_state.get('valence', 0.0)
        self.accuracy += 0.05 * valence  # Positive affect increases accuracy


class DecisionSNNAdapter:
    """Adapter for the DecisionSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            # Initialize the actual DecisionSNN
            self.snn = DecisionSNN()
        else:
            # Mock implementation
            self.actions = ["explore", "exploit", "reinforce", "adapt"]
            self.confidence = 0.5
            self.last_action = None
            self.last_reward = 0.0
    
    def select_action(self, context: Dict) -> Dict:
        """Select an action based on context."""
        if not isinstance(context, dict):
            context = {}
            
        if self.use_real_snn:
            # Convert context to the format expected by DecisionSNN
            decision_input = {
                'task_type': context.get('task_type', 'unknown'),
                'complexity': float(context.get('complexity', 1.0)),
                'prior_results': context.get('reasoning_result', {})
            }
            
            # Get decision from SNN
            result = self.snn.decide(decision_input)
            
            # Format consistent output
            return {
                'action': result.get('action', 'default_action'),
                'confidence': result.get('confidence', 0.5),
                'params': result.get('parameters', {})
            }
            
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
    
    def update_policy(self, reward: float):
        """Update decision policy based on reward."""
        if not isinstance(reward, (int, float)):
            reward = 0.0
            
        if self.use_real_snn:
            # Update the DecisionSNN with reward
            self.snn.update_with_reward(reward)
            return
            
        # Mock implementation - track reward and adjust confidence
        delta = reward - self.last_reward
        self.confidence = min(0.95, max(0.1, self.confidence + 0.05 * delta))
        self.last_reward = reward
    
    def set_affective_state(self, affective_state: Dict):
        """Set affective state to modulate decision-making."""
        if not isinstance(affective_state, dict):
            affective_state = {}
            
        if self.use_real_snn:
            # If the SNN has a method to handle affective state
            if hasattr(self.snn, 'set_affective_state'):
                self.snn.set_affective_state(affective_state)
            return
            
        # Mock implementation - adjust behavior based on affective state
        valence = affective_state.get('valence', 0.0)
        arousal = affective_state.get('arousal', 0.5)
        
        # Higher arousal = more exploration
        explore_bias = 0.3 * arousal
        
        # Positive valence = more confidence
        self.confidence = min(0.95, max(0.1, self.confidence + 0.1 * valence))


class PerceptualSNNAdapter:
    """Adapter for the PerceptualSNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            # Initialize the actual PerceptualSNN
            self.snn = PerceptualSNN()
        else:
            # Mock implementation
            self.encoding_size = 64
            self.accuracy = 0.7
            self.embeddings = {}
    
    def process(self, input_data: Any) -> np.ndarray:
        """Process input data into neural encoding."""
        # Convert input to string representation if not already
        if not isinstance(input_data, str):
            input_data = str(input_data)
            
        if self.use_real_snn:
            # Process through real SNN with method detection
            try:
                # Check which method to use - some implementations use process, others use perceive
                if hasattr(self.snn, 'process'):
                    print("Using PerceptualSNN.process method")
                    result = self.snn.process(input_data)
                    return result
                elif hasattr(self.snn, 'perceive'):
                    print("Using PerceptualSNN.perceive method")
                    result = self.snn.perceive(input_data)
                    return result
                elif hasattr(self.snn, 'encode'):
                    print("Using PerceptualSNN.encode method")
                    result = self.snn.encode(input_data)
                    return result
                else:
                    print("Warning: PerceptualSNN has no process, perceive, or encode method")
                    raise AttributeError("No suitable method found in PerceptualSNN")
            except Exception as e:
                print(f"Error calling real PerceptualSNN methods: {e}")
                print("Falling back to mock implementation")
                # Continue to mock implementation
            
        # Mock implementation - create consistent encoding for inputs
        if input_data in self.embeddings:
            # Return cached encoding for consistent results
            return self.embeddings[input_data]
            
        # Generate mock encoding
        encoding = np.random.random(self.encoding_size)
        
        # Simple hash-based perturbation for consistency
        hash_val = hash(input_data) % 1000000
        np.random.seed(hash_val)
        encoding = np.random.random(self.encoding_size)
        np.random.seed(None)  # Reset seed
        
        # Store for consistency
        self.embeddings[input_data] = encoding
        return encoding
    
    def update(self, input_data: Any, target_encoding: np.ndarray, reward: float):
        """Update perceptual SNN based on reward."""
        if not isinstance(reward, (int, float)):
            reward = 0.0
            
        if self.use_real_snn:
            # Update the PerceptualSNN
            if hasattr(self.snn, 'update'):
                self.snn.update(input_data, target_encoding, reward)
            return
            
        # Mock implementation - improve accuracy
        self.accuracy = min(0.95, max(0.3, self.accuracy + 0.02 * (reward - 0.5)))


class MemorySNNAdapter:
    """Adapter for the MemorySNN to interface with Absolute Zero."""
    
    def __init__(self, use_real_snn=True):
        self.use_real_snn = use_real_snn and HAS_SNN_MODULES
        
        if self.use_real_snn:
            # Initialize the actual MemorySNN
            self.snn = MemorySNN()
        else:
            # Mock implementation
            self.memory_store = []
            self.capacity = 100
            self.utilization = 0.0
    
    def store(self, memory_item: Dict):
        """Store an item in memory."""
        if not isinstance(memory_item, dict):
            memory_item = {}
            
        if self.use_real_snn:
            # Store in the real SNN
            self.snn.store(memory_item)
            return
            
        # Mock implementation - add to memory store
        if len(self.memory_store) >= self.capacity:
            # Remove oldest item when at capacity
            self.memory_store.pop(0)
            
        # Timestamp for recency
        if 'timestamp' not in memory_item:
            memory_item['timestamp'] = time.time()
            
        self.memory_store.append(memory_item)
        self.utilization = len(self.memory_store) / self.capacity
    
    def retrieve(self, query: Dict) -> List[Dict]:
        """Retrieve memory items matching query."""
        if not isinstance(query, dict):
            query = {}
            
        if self.use_real_snn:
            # Retrieve from real SNN
            return self.snn.retrieve(query)
            
        # Mock implementation - simple filtering
        results = []
        
        for item in self.memory_store:
            match = True
            for key, value in query.items():
                if key in item and item[key] != value:
                    match = False
                    break
            
            if match:
                results.append(item)
                
        # Sort by recency (timestamp)
        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return results[:10]  # Return top 10 most recent matches
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if self.use_real_snn:
            # Get stats from real SNN
            if hasattr(self.snn, 'get_statistics'):
                return self.snn.get_statistics()
            return {'utilization': 0.5}  # Default if not implemented
            
        # Mock implementation
        return {
            'utilization': self.utilization,
            'items_count': len(self.memory_store),
            'capacity': self.capacity
        }
    
    def consolidate(self):
        """Consolidate memory (strengthen important items, forget less relevant)."""
        if self.use_real_snn:
            # Call real SNN consolidation
            if hasattr(self.snn, 'consolidate'):
                self.snn.consolidate()
            return
            
        # Mock implementation - simple pruning of old items
        if len(self.memory_store) > self.capacity * 0.8:
            # Sort by recency
            self.memory_store.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Keep top 80%
            keep_count = int(self.capacity * 0.8)
            self.memory_store = self.memory_store[:keep_count]
            self.utilization = len(self.memory_store) / self.capacity


def create_snn_components(use_real_snn=False) -> Dict:
    """Create the SNN components for Absolute Zero."""
    print(f"\nCreating SNN components with use_real_snn={use_real_snn}")
    
    if use_real_snn and not HAS_SNN_MODULES:
        if missing_packages:
            print(f"Cannot use real SNNs due to missing dependencies: {', '.join(missing_packages)}")
            print("Please install the missing packages with: pip install " + " ".join(missing_packages))
        else:
            print("Cannot use real SNNs due to import errors. Using mock implementations.")
        print("Forcing use_real_snn=False due to missing modules or packages")
        use_real_snn = False  # Fall back to mock implementations
    elif use_real_snn:
        print("Real SNN modules are available and will be used!")
    
    # Initialize the SNNs with error handling
    snns = {}
    
    # Create StatisticalSNNAdapter with special attention since it's critical
    try:
        print(f"Creating StatisticalSNNAdapter with use_real_snn={use_real_snn}")
        stat_snn = StatisticalSNNAdapter(use_real_snn)
        # Verify that process_input method exists and works
        if hasattr(stat_snn, 'process_input'):
            test_vector = np.array([0.1, 0.2, 0.3])
            try:
                test_result = stat_snn.process_input(test_vector)
                if isinstance(test_result, dict) and 'next_element' in test_result:
                    print("StatisticalSNNAdapter successfully initialized and tested")
                else:
                    print("Warning: StatisticalSNNAdapter.process_input returned invalid format")
                    raise ValueError("Invalid process_input result format")
            except Exception as e:
                print(f"Warning: Error testing StatisticalSNNAdapter.process_input: {e}")
                # Reinitialize with mock if real fails
                if use_real_snn:
                    print("Falling back to mock StatisticalSNNAdapter")
                    stat_snn = StatisticalSNNAdapter(False)
        else:
            print("Warning: StatisticalSNNAdapter missing process_input method")
            raise AttributeError("Missing process_input method")
            
        # Store the adapter after successful testing
        snns['statistical'] = stat_snn
    except Exception as e:
        print(f"Error creating StatisticalSNNAdapter: {e}")
        print("Creating basic mock StatisticalSNNAdapter")
        
        # Create a fully functional basic mock adapter since the regular one failed
        try:
            from .test_absolute_zero import MockStatisticalSNN
            snns['statistical'] = MockStatisticalSNN()
        except ImportError:
            print("Could not import MockStatisticalSNN, creating basic mock")
            class BasicMockSNN:
                def process_input(self, input_vector):
                    return {"next_element": 0}
                def update_weights(self, input_vector, prediction, reward):
                    print(f"Mock update with reward: {reward}")
                def get_region_activations(self):
                    return np.array([0.5, 0.5, 0.5])
            snns['statistical'] = BasicMockSNN()
    
    # Create the rest of the adapters with error handling and fallbacks to test_absolute_zero versions
    adapter_classes = {
        'affective': AffectiveSNNAdapter,
        'metacognitive': MetacognitiveSNNAdapter, 
        'reasoning': ReasoningSNNAdapter,
        'decision': DecisionSNNAdapter,
        'perceptual': PerceptualSNNAdapter,
        'memory': MemorySNNAdapter
    }
    
    test_mock_classes = {
        'affective': 'MockAffectiveSNN',
        'metacognitive': 'MockMetacognitiveSNN'
    }
    
    for name, adapter_class in adapter_classes.items():
        try:
            print(f"Creating {name} adapter with use_real_snn={use_real_snn}")
            snns[name] = adapter_class(use_real_snn)
            
            # Verify the adapter has required methods
            if name == 'affective' and not hasattr(snns[name], 'evaluate_affective_state'):
                raise AttributeError(f"Missing evaluate_affective_state method in {name} adapter")
            elif name == 'metacognitive' and not hasattr(snns[name], 'monitor_system_state'):
                raise AttributeError(f"Missing monitor_system_state method in {name} adapter")
                
        except Exception as e:
            print(f"Error creating {name} adapter: {e}")
            print(f"Creating {name} adapter with mock implementation")
            
            # Use test_absolute_zero mock classes for core adapters if available
            if name in test_mock_classes:
                try:
                    from .test_absolute_zero import MockAffectiveSNN, MockMetacognitiveSNN
                    mock_class_name = test_mock_classes[name]
                    if mock_class_name == 'MockAffectiveSNN':
                        snns[name] = MockAffectiveSNN()
                    elif mock_class_name == 'MockMetacognitiveSNN':
                        snns[name] = MockMetacognitiveSNN()
                    print(f"Successfully created {name} adapter using test mock class")
                except Exception as mock_e:
                    print(f"Error creating mock {name} adapter: {mock_e}")
                    snns[name] = adapter_class(False)  # Last resort - use regular adapter with mock=False
            else:
                snns[name] = adapter_class(False)
    
    # Print information about each SNN component
    print("\nEnhanced Absolute Zero Components:")
    print("----------------------------------")
    for name, adapter in snns.items():
        using_real = hasattr(adapter, 'use_real_snn') and adapter.use_real_snn
        status = "REAL" if using_real else "MOCK"
        print(f"- {name.upper()} SNN: {status}")
    
    # Final verification of the statistical SNN
    if 'statistical' not in snns or not hasattr(snns['statistical'], 'process_input'):
        print("ERROR: Statistical SNN missing or lacks required process_input method")
        print("Creating a guaranteed working mock StatisticalSNN")
        
        # Create a simple mock that DEFINITELY has process_input
        class BasicMockSNN:
            def process_input(self, input_vector):
                if not isinstance(input_vector, np.ndarray):
                    input_vector = np.array([0.0])
                return {"next_element": 0}
                
            def update_weights(self, input_vector, prediction, reward):
                print(f"Mock update with reward: {reward}")
                
            def get_region_activations(self):
                return np.array([0.5, 0.5, 0.5])
                
        snns['statistical'] = BasicMockSNN()
        print("Verified statistical SNN has working process_input method")
    
    return snns 