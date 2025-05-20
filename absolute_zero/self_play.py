import random
import numpy as np
from typing import Dict
from .task_generator import CogmentaTaskGenerator
from .reward_calculator import RewardCalculator

class CogmentaSelfPlay:
    def __init__(self, snns=None, task_generator=None, reward_calculator=None, curriculum_manager=None):
        """
        Initialize the self-play environment with provided components.
        
        Args:
            snns: Dictionary of SNN adapters with at least 'statistical', 'affective', 'metacognitive' keys
            task_generator: Task generator instance
            reward_calculator: Reward calculator instance
            curriculum_manager: Curriculum manager instance
        """
        # Check if SNNs are provided correctly
        if not isinstance(snns, dict):
            print("Warning: SNNs not provided as a dictionary, creating empty dictionary")
            snns = {}
            
        self.task_generator = task_generator
        self.curriculum_manager = curriculum_manager
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        # Initialize SNNs with error checking
        self.stat_snn = snns.get('statistical')
        if self.stat_snn is None:
            print("ERROR: Statistical SNN is missing or None. Self-play will fail.")
            # Create a minimal mock to prevent crashes
            class BasicMockSNN:
                def process_input(self, input_vector):
                    return {"next_element": 0}
                def update_weights(self, input_vector, prediction, reward):
                    pass
                def get_region_activations(self):
                    return np.array([0.5, 0.5, 0.5])
            self.stat_snn = BasicMockSNN()
            print("Created a basic mock StatisticalSNN as fallback")
        
        self.affect_snn = snns.get('affective')
        if self.affect_snn is None:
            print("Warning: Affective SNN is missing or None. Using minimal mock.")
            class BasicAffectiveMock:
                def evaluate_affective_state(self, metrics):
                    return {"valence": 0, "arousal": 0.5}
                def influence_processing(self, stat_snn):
                    pass
                def get_emotion_state(self):
                    return {"valence": 0, "arousal": 0.5}
            self.affect_snn = BasicAffectiveMock()
            
        self.meta_snn = snns.get('metacognitive')
        if self.meta_snn is None:
            print("Warning: Metacognitive SNN is missing or None. Using minimal mock.")
            class BasicMetacognitiveMock:
                def monitor_system_state(self, state_info):
                    pass
                def get_metacognitive_state(self):
                    return {"confidence": 0.5, "uncertainty": 0.5}
            self.meta_snn = BasicMetacognitiveMock()
            
        # Print status of initialized components
        print(f"Self-play initialized with: statistical={'OK' if hasattr(self.stat_snn, 'process_input') else 'MISSING'}, " +
              f"affective={'OK' if hasattr(self.affect_snn, 'evaluate_affective_state') else 'MISSING'}, " +
              f"metacognitive={'OK' if hasattr(self.meta_snn, 'monitor_system_state') else 'MISSING'}")
            
        self.training_history = []
        
    def self_play_iteration(self, task=None) -> Dict[str, float]:
        """Execute one iteration of self-play learning"""
        # 1. PROPOSE: Generate task if not provided
        if task is None and self.task_generator:
            task = self.task_generator.generate_task()
        elif task is None:
            print("Warning: No task generator available to create a task")
            # Create a minimal mock task
            task = {
                "type": "pattern",
                "input": {"pattern": [0.1, 0.2, 0.3]},
                "output": {"next_element": 0}
            }
        
        # 2. Evaluate learnability with error checking
        try:
            if self.reward_calculator and hasattr(self.reward_calculator, 'calculate_learnability_reward'):
                learnability = self.reward_calculator.calculate_learnability_reward(task, self.stat_snn)
            else:
                print("Warning: No reward calculator available or missing calculate_learnability_reward method")
                learnability = 0.5  # Default value
        except Exception as e:
            print(f"Error calculating learnability: {e}")
            learnability = 0.5
        
        # 3. SOLVE: Attempt the task with error checking
        try:
            input_vector = self.encode_task_input(task.get("input", {}))
            
            if hasattr(self.stat_snn, 'process_input'):
                prediction = self.stat_snn.process_input(input_vector)
            else:
                print("Error: Statistical SNN has no process_input method")
                prediction = {"next_element": 0}
        except Exception as e:
            print(f"Error in task solving: {e}")
            prediction = {"next_element": 0}
        
        # 4. Get solving reward with error checking
        try:
            if hasattr(self.reward_calculator, 'verify_output'):
                solve_reward = float(self.reward_calculator.verify_output(prediction, task.get("output", {})))
            else:
                print("Warning: Reward calculator missing verify_output method")
                solve_reward = 0.0
        except Exception as e:
            print(f"Error calculating solve reward: {e}")
            solve_reward = 0.0
        
        # 5. UPDATE: Learn from both rewards
        combined_reward = learnability + 0.5 * solve_reward
        self._update_weights(input_vector, prediction, combined_reward)
        
        # 6. Metacognitive monitoring with error checking
        metrics = {
            'task_type': task.get('type', 'unknown'),
            'learnability': learnability,
            'accuracy': solve_reward,
            'combined_reward': combined_reward
        }
        
        try:
            if hasattr(self.meta_snn, 'monitor_system_state'):
                self.meta_snn.monitor_system_state({
                    'learning_metrics': metrics,
                    'component_states': self._get_component_states()
                })
        except Exception as e:
            print(f"Error in metacognitive monitoring: {e}")
        
        return metrics
    
    def encode_task_input(self, task_input: Dict) -> np.ndarray:
        """Encode task input for SNN processing"""
        # Implementation depends on task structure
        try:
            if isinstance(task_input, dict):
                # Simple example implementation
                return np.array([float(val) for val in task_input.values() if isinstance(val, (int, float))])
            elif isinstance(task_input, (list, tuple)):
                return np.array(task_input, dtype=float)
            else:
                # Default fallback
                return np.array([0.0])
        except Exception as e:
            print(f"Error encoding task input: {e}")
            return np.array([0.0])
    
    def _update_weights(self, input_vector, prediction, reward):
        """Update SNN weights based on reward"""
        try:
            # Use appropriate learning rule for your SNNs
            if hasattr(self.stat_snn, 'update_weights'):
                self.stat_snn.update_weights(input_vector, prediction, reward)
        except Exception as e:
            print(f"Error updating weights: {e}")
    
    def _get_component_states(self):
        """Get current states of all components"""
        try:
            return {
                'statistical': {'activation': self.stat_snn.get_region_activations() if hasattr(self.stat_snn, 'get_region_activations') else None},
                'affective': {'activation': self.affect_snn.get_emotion_state() if hasattr(self.affect_snn, 'get_emotion_state') else None},
                'metacognitive': {'activation': self.meta_snn.get_metacognitive_state() if hasattr(self.meta_snn, 'get_metacognitive_state') else None}
            }
        except Exception as e:
            print(f"Error getting component states: {e}")
            return {'statistical': None, 'affective': None, 'metacognitive': None}

class IntegratedCogmentaLearner:
    def __init__(self, stat_snn, affect_snn, meta_snn):
        self.stat_snn = stat_snn
        self.affect_snn = affect_snn
        self.meta_snn = meta_snn
        self.current_difficulty = 1.0
        
    def train_with_absolute_zero(self):
        converged = False
        while not converged:
            # Statistical SNN learns patterns
            stat_task = self.generate_pattern_task()
            stat_result = self.stat_snn.learn_from_task(stat_task)
            
            # Affective SNN provides learning modulation
            affect_state = self.affect_snn.evaluate_learning_emotion({
                'frustration': stat_result['error_rate'],
                'curiosity': stat_result['novelty'],
                'satisfaction': stat_result['progress']
            })
            
            # Metacognitive SNN adjusts strategy
            meta_decision = self.meta_snn.adjust_learning_strategy({
                'performance': stat_result,
                'affect': affect_state,
                'task_difficulty': self.current_difficulty
            })
            
            # Apply metacognitive adjustments
            if meta_decision['change_approach']:
                self.switch_task_type()
            if meta_decision['adjust_difficulty']:
                self.modify_task_complexity(meta_decision['direction'])
                
            # Check for convergence
            if self.check_convergence(stat_result):
                converged = True
    
    def generate_pattern_task(self):
        """Generate a pattern recognition task"""
        # Implementation would depend on the specific domain
        pass
        
    def switch_task_type(self):
        """Switch to a different type of task"""
        pass
        
    def modify_task_complexity(self, direction):
        """Modify task complexity based on direction"""
        if direction == 'increase':
            self.current_difficulty *= 1.1
        else:
            self.current_difficulty *= 0.9
            
    def check_convergence(self, result):
        """Check if training has converged"""
        # Simple example implementation
        return result.get('error_rate', 1.0) < 0.1