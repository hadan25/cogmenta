import random
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class HybridSelfPlay:
    """
    Enhanced self-play implementation that uses hybrid learning approaches
    by combining bio-inspired and supervised learning methods.
    """
    
    def __init__(self, task_generator, curriculum_manager=None, reward_calculator=None, snns=None):
        """
        Initialize the hybrid self-play system.
        
        Args:
            task_generator: Task generator component
            curriculum_manager: Optional curriculum manager
            reward_calculator: Optional reward calculator
            snns: Dictionary of SNN adapters
        """
        self.task_generator = task_generator
        self.curriculum_manager = curriculum_manager
        self.reward_calculator = reward_calculator
        
        # Initialize SNNs if provided
        if snns:
            self.stat_snn = snns.get('statistical')
            self.affect_snn = snns.get('affective')
            self.meta_snn = snns.get('metacognitive')
            self.decision_snn = snns.get('decision')
            self.perceptual_snn = snns.get('perceptual')
            self.memory_snn = snns.get('memory')
            self.reasoning_snn = snns.get('reasoning')
        else:
            self.stat_snn = None
            self.affect_snn = None
            self.meta_snn = None
            self.decision_snn = None
            self.perceptual_snn = None
            self.memory_snn = None
            self.reasoning_snn = None
        
        # Training history
        self.training_history = []
        
        # Specialized learning tracking
        self.specialized_synapses_by_task = {}
    
    def hybrid_self_play_iteration(self, task=None) -> Dict[str, Any]:
        """
        Execute one iteration of hybrid self-play learning using both
        bio-inspired and supervised learning approaches.
        
        Args:
            task: Optional task. If not provided, a task will be generated.
            
        Returns:
            Dictionary with metrics
        """
        # 1. PROPOSE: Generate task if not provided
        if task is None:
            task = self.task_generator.generate_task()
        
        # 2. PREPARATION: Prepare specialized synapses for this task type
        task_type = task.get('type', 'unknown')
        self._prepare_specialized_learning(task)
        
        # 3. BIO-INSPIRED PHASE: Evaluate learnability and process input
        learnability = 0.0
        if self.reward_calculator:
            learnability = self.reward_calculator.calculate_learnability_reward(
                task, self.stat_snn
            )
        
        input_vector = self._encode_task_input(task["input"])
        bio_prediction = self.stat_snn.process_input(input_vector)
        
        # 4. SUPERVISED PHASE: Extract targets and apply supervised learning
        supervised_targets = self._extract_targets_from_task(task)
        supervised_result = self._apply_supervised_learning(input_vector, supervised_targets)
        
        # 5. HYBRID INTEGRATION: Combine bio-inspired and supervised results
        hybrid_prediction = self._combine_predictions(bio_prediction, supervised_result)
        
        # 6. REWARD CALCULATION: Calculate multiple reward signals
        bio_reward = 0.0
        supervised_accuracy = 0.0
        combined_reward = 0.0
        
        if self.reward_calculator:
            bio_reward = float(self.reward_calculator.verify_output(
                bio_prediction, task["output"]
            ))
            
            supervised_accuracy = self._calculate_supervised_accuracy(
                supervised_result, task["output"]
            )
            
            combined_reward = self._combine_rewards(
                bio_reward, supervised_accuracy, learnability
            )
        
        # 7. UPDATE: Apply learning based on rewards
        self._update_weights(input_vector, hybrid_prediction, combined_reward)
        
        # 8. METACOGNITIVE MONITORING: Observe system state
        metrics = {
            'task_type': task_type,
            'learnability': learnability,
            'bio_accuracy': bio_reward,
            'supervised_accuracy': supervised_accuracy, 
            'combined_reward': combined_reward
        }
        
        if self.meta_snn:
            self.meta_snn.monitor_system_state({
                'learning_metrics': metrics,
                'component_states': self._get_component_states()
            })
        
        # 9. CLEANUP: Clean up specialized learning for this task
        self._cleanup_specialized_learning(task)
        
        # Record history
        self.training_history.append({
            'task': task,
            'metrics': metrics
        })
        
        return metrics
    
    def _encode_task_input(self, task_input: Any) -> np.ndarray:
        """
        Encode task input for SNN processing.
        
        Args:
            task_input: Input from task
            
        Returns:
            Encoded vector for SNN
        """
        # If input is already a numpy array, return it
        if isinstance(task_input, np.ndarray):
            return task_input
            
        # If input is a list of numbers, convert to numpy array
        if isinstance(task_input, (list, tuple)) and all(isinstance(x, (int, float)) for x in task_input):
            return np.array(task_input, dtype=float)
            
        # Handle dictionaries
        if isinstance(task_input, dict):
            if 'features' in task_input:
                return np.array(task_input['features'], dtype=float)
                
            if 'sequence' in task_input:
                return np.array(task_input['sequence'], dtype=float)
                
            if 'vector' in task_input:
                return np.array(task_input['vector'], dtype=float)
                
            # Extract all numeric values from dictionary
            numeric_values = []
            for k, v in task_input.items():
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                    
            if numeric_values:
                return np.array(numeric_values)
        
        # Default fallback - hash the input string into a vector
        input_str = str(task_input)
        hash_val = hash(input_str) % 10000
        random.seed(hash_val)
        encoded = np.array([random.random() for _ in range(10)])
        random.seed(None)  # Reset seed
        
        return encoded
    
    def _extract_targets_from_task(self, task: Dict) -> Dict:
        """
        Extract supervised learning targets from the task structure.
        
        Args:
            task: Task dictionary
            
        Returns:
            Dictionary of targets for supervised learning
        """
        targets = {}
        
        # Extract output as target
        if 'output' in task:
            targets['output'] = task['output']
            
        # Extract specific targets based on task type
        task_type = task.get('type', 'unknown')
        
        if task_type == 'pattern':
            if 'input' in task and 'sequence' in task['input']:
                sequence = task['input']['sequence']
                if len(sequence) > 1:
                    # Target is the next element in the sequence
                    targets['next_element'] = sequence[-1]
                    
        elif task_type == 'classification':
            if 'output' in task and 'category' in task['output']:
                targets['category'] = task['output']['category']
                
        elif task_type == 'association':
            if 'output' in task and 'associated_item' in task['output']:
                targets['associated_item'] = task['output']['associated_item']
                
        # Include any additional target information from task metadata
        if 'metadata' in task and 'targets' in task['metadata']:
            targets.update(task['metadata']['targets'])
            
        return targets
    
    def _apply_supervised_learning(self, input_vector: np.ndarray, targets: Dict) -> Dict:
        """
        Apply supervised learning using the appropriate SNN.
        
        Args:
            input_vector: Encoded input vector
            targets: Dictionary of targets
            
        Returns:
            Result of supervised learning
        """
        # Choose appropriate SNN based on target types
        result = {}
        
        # StatisticalSNN for general pattern recognition
        if self.stat_snn and hasattr(self.stat_snn, 'supervised_learning'):
            result['statistical'] = self.stat_snn.supervised_learning(input_vector, targets)
            
        # Use specialized SNNs if available and appropriate
        if 'category' in targets and self.perceptual_snn and hasattr(self.perceptual_snn, 'train_perception'):
            result['perceptual'] = self.perceptual_snn.train_perception(input_vector, targets['category'])
            
        if 'action' in targets and self.decision_snn and hasattr(self.decision_snn, 'train_decision'):
            result['decision'] = self.decision_snn.train_decision(
                input_vector, targets['action'], 
                targets.get('confidence', 0.8)
            )
            
        if 'reasoning_target' in targets and self.reasoning_snn and hasattr(self.reasoning_snn, 'train_reasoning'):
            result['reasoning'] = self.reasoning_snn.train_reasoning(
                input_vector, targets['reasoning_target']
            )
            
        return result
    
    def _combine_predictions(self, bio_prediction: Any, supervised_result: Dict) -> Any:
        """
        Combine bio-inspired and supervised predictions.
        
        Args:
            bio_prediction: Result from bio-inspired processing
            supervised_result: Result from supervised learning
            
        Returns:
            Combined prediction
        """
        # If no supervised result, use bio prediction
        if not supervised_result:
            return bio_prediction
            
        # If bio_prediction is a dictionary, merge with supervised results
        if isinstance(bio_prediction, dict):
            combined = bio_prediction.copy()
            
            # Add supervised results, prioritizing them over bio results
            for source, prediction in supervised_result.items():
                if isinstance(prediction, dict):
                    combined.update(prediction)
                else:
                    combined[f"{source}_prediction"] = prediction
                    
            return combined
            
        # If supervised result contains 'statistical' prediction, use it
        if 'statistical' in supervised_result:
            return supervised_result['statistical']
            
        # Fallback to bio prediction
        return bio_prediction
    
    def _calculate_supervised_accuracy(self, supervised_result: Dict, expected_output: Any) -> float:
        """
        Calculate accuracy of supervised learning.
        
        Args:
            supervised_result: Result from supervised learning
            expected_output: Expected output from task
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not supervised_result:
            return 0.0
            
        # Extract the most relevant prediction
        prediction = None
        
        # Priority order: statistical, perceptual, decision, reasoning
        if 'statistical' in supervised_result:
            prediction = supervised_result['statistical']
        elif 'perceptual' in supervised_result:
            prediction = supervised_result['perceptual']
        elif 'decision' in supervised_result:
            prediction = supervised_result['decision']
        elif 'reasoning' in supervised_result:
            prediction = supervised_result['reasoning']
            
        if prediction is None:
            return 0.0
            
        # Compare prediction to expected output
        if isinstance(expected_output, dict) and isinstance(prediction, dict):
            # Calculate proportion of matching key-value pairs
            matches = sum(1 for k, v in prediction.items() 
                         if k in expected_output and expected_output[k] == v)
            total = len(expected_output)
            return matches / total if total > 0 else 0.0
            
        # Direct comparison
        return 1.0 if prediction == expected_output else 0.0
    
    def _combine_rewards(self, bio_reward: float, supervised_accuracy: float, 
                         learnability: float) -> float:
        """
        Combine multiple reward signals into a single value.
        
        Args:
            bio_reward: Reward from bio-inspired processing
            supervised_accuracy: Accuracy from supervised learning
            learnability: Learnability score
            
        Returns:
            Combined reward
        """
        # Weighted combination
        # Bio-reward is weighted most heavily as it represents the core learning signal
        bio_weight = 0.6
        supervised_weight = 0.3
        learnability_weight = 0.1
        
        return (bio_reward * bio_weight + 
                supervised_accuracy * supervised_weight + 
                learnability * learnability_weight)
    
    def _update_weights(self, input_vector: np.ndarray, prediction: Any, reward: float):
        """
        Update SNN weights based on reward.
        
        Args:
            input_vector: Encoded input vector
            prediction: Combined prediction
            reward: Combined reward signal
        """
        # Update statistical SNN
        if self.stat_snn and hasattr(self.stat_snn, 'update_weights'):
            self.stat_snn.update_weights(input_vector, prediction, reward)
            
        # Update decision SNN
        if self.decision_snn and hasattr(self.decision_snn, 'update_policy'):
            try:
                self.decision_snn.update_policy(reward)
            except Exception as e:
                print(f"Warning: Error updating DecisionSNN: {e}")
            
        # Update affective SNN
        if self.affect_snn and hasattr(self.affect_snn, 'update_emotional_state'):
            # Create metrics dictionary for affective update
            metrics = {
                'reward': reward,
                'prediction_quality': 0.5 + (reward / 2)  # Scale to 0-1 range
            }
            self.affect_snn.update_emotional_state(metrics)
    
    def _prepare_specialized_learning(self, task: Dict):
        """
        Prepare specialized learning synapses based on task type.
        
        Args:
            task: Task dictionary
        """
        task_type = task.get('type', 'unknown')
        task_id = task.get('id', f"task_{len(self.training_history)}")
        
        # Identify task-relevant neurons and synapses
        if self.stat_snn and hasattr(self.stat_snn, 'get_task_relevant_neurons'):
            relevant_neurons = self.stat_snn.get_task_relevant_neurons(task)
            
            # Generate pairs of presynaptic-postsynaptic neurons
            synapses = []
            
            if relevant_neurons:
                # Create pairs for specialized learning
                for post_neuron in relevant_neurons['output']:
                    for pre_neuron in relevant_neurons['input']:
                        synapses.append((pre_neuron, post_neuron))
                
                # Register specialized synapses
                if hasattr(self.stat_snn, 'register_specialized_synapses'):
                    registered_count = self.stat_snn.register_specialized_synapses(synapses)
                    
                    # Store for cleanup
                    self.specialized_synapses_by_task[task_id] = synapses
    
    def _cleanup_specialized_learning(self, task: Dict):
        """
        Clean up specialized learning synapses after task.
        
        Args:
            task: Task dictionary
        """
        task_id = task.get('id', f"task_{len(self.training_history) - 1}")
        
        if task_id in self.specialized_synapses_by_task:
            synapses = self.specialized_synapses_by_task[task_id]
            
            # Unregister specialized synapses
            if self.stat_snn and hasattr(self.stat_snn, 'unregister_specialized_synapse'):
                for pre, post in synapses:
                    self.stat_snn.unregister_specialized_synapse(pre, post)
            
            # Remove from tracking
            del self.specialized_synapses_by_task[task_id]
    
    def _get_component_states(self) -> Dict:
        """
        Get current states of all components.
        
        Returns:
            Dictionary with component states
        """
        states = {}
        
        # Get statistical SNN state
        if self.stat_snn:
            if hasattr(self.stat_snn, 'get_region_activations'):
                states['statistical'] = {'activation': self.stat_snn.get_region_activations()}
            elif hasattr(self.stat_snn, 'get_neuron_activations'):
                states['statistical'] = {'activation': self.stat_snn.get_neuron_activations()}
        
        # Get affective SNN state
        if self.affect_snn and hasattr(self.affect_snn, 'get_emotion_state'):
            states['affective'] = {'activation': self.affect_snn.get_emotion_state()}
        
        # Get metacognitive SNN state
        if self.meta_snn and hasattr(self.meta_snn, 'get_metacognitive_state'):
            states['metacognitive'] = {'activation': self.meta_snn.get_metacognitive_state()}
        
        # Get decision SNN state
        if self.decision_snn and hasattr(self.decision_snn, 'get_policy_state'):
            states['decision'] = {'activation': self.decision_snn.get_policy_state()}
        
        return states 