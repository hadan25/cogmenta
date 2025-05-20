import random
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ModuleSelector:
    """
    Helper class that selects appropriate modules for a task.
    """
    
    def __init__(self, snns=None, task_generators=None, reward_calculators=None):
        """
        Initialize module selector.
        
        Args:
            snns: Dictionary of available SNNs
            task_generators: Dictionary of available task generators
            reward_calculators: Dictionary of available reward calculators
        """
        self.snns = snns or {}
        self.task_generators = task_generators or {}
        self.reward_calculators = reward_calculators or {}
        
        # SNN to task type mapping
        self.snn_task_mapping = {
            'statistical': ['pattern', 'classification', 'association', 'sequence'],
            'perceptual': ['classification', 'perceptual', 'feature_extraction'],
            'reasoning': ['reasoning', 'deductive', 'inductive', 'abductive'],
            'decision': ['decision', 'action_selection', 'policy'],
            'memory': ['memory', 'recall', 'recognition', 'association'],
            'metacognitive': ['metacognitive', 'confidence', 'strategy'],
            'affective': ['affective', 'emotion', 'motivation']
        }
        
        # Task type to SNN mapping (inverse of above)
        self.task_snn_mapping = self._create_inverse_mapping()
    
    def _create_inverse_mapping(self) -> Dict[str, List[str]]:
        """
        Create inverse mapping from task types to SNNs.
        
        Returns:
            Dictionary mapping task types to SNN names
        """
        inverse_mapping = defaultdict(list)
        
        for snn_name, task_types in self.snn_task_mapping.items():
            for task_type in task_types:
                inverse_mapping[task_type].append(snn_name)
        
        return inverse_mapping
    
    def select_modules(self, meta_task=None) -> Dict[str, Any]:
        """
        Select appropriate modules for a task.
        
        Args:
            meta_task: Optional task description guiding module selection
            
        Returns:
            Dictionary with selected modules
        """
        selected = {
            'snns': [],
            'generator': None,
            'calculator': None
        }
        
        # If meta_task is specified, use it to guide selection
        if meta_task is not None:
            task_type = meta_task.get('type', 'unknown')
            
            # Select appropriate SNNs for task type
            if task_type in self.task_snn_mapping:
                # Get list of appropriate SNNs for this task type
                appropriate_snns = self.task_snn_mapping[task_type]
                
                # Filter to available SNNs
                available_snns = [snn for snn in appropriate_snns if snn in self.snns]
                
                if available_snns:
                    selected['snns'] = available_snns
            
            # Select appropriate task generator
            if task_type in self.task_generators:
                selected['generator'] = task_type
            elif 'default' in self.task_generators:
                selected['generator'] = 'default'
            elif self.task_generators:
                selected['generator'] = next(iter(self.task_generators))
                
            # Select appropriate reward calculator
            if task_type in self.reward_calculators:
                selected['calculator'] = task_type
            elif 'default' in self.reward_calculators:
                selected['calculator'] = 'default'
            elif self.reward_calculators:
                selected['calculator'] = next(iter(self.reward_calculators))
                
        else:
            # Default: use all available SNNs and first generator/calculator
            selected['snns'] = list(self.snns.keys())
            
            if self.task_generators:
                # Prefer multi-modal generator if available
                if 'multi_modal' in self.task_generators:
                    selected['generator'] = 'multi_modal'
                else:
                    selected['generator'] = next(iter(self.task_generators))
            
            if self.reward_calculators:
                # Prefer multi-SNN calculator if available
                if 'multi_snn' in self.reward_calculators:
                    selected['calculator'] = 'multi_snn'
                else:
                    selected['calculator'] = next(iter(self.reward_calculators))
        
        return selected

class CrossSNNCommunicator:
    """
    Manages communication between different SNN components.
    """
    
    def __init__(self, snns=None):
        """
        Initialize the cross-SNN communicator.
        
        Args:
            snns: Dictionary of SNN adapters
        """
        self.snns = snns or {}
    
    def propagate_learning(self, source_snn: str, target_snn: str, learning_signal: float) -> bool:
        """
        Propagate learning from source SNN to target SNN.
        
        Args:
            source_snn: Name of source SNN
            target_snn: Name of target SNN
            learning_signal: Signal guiding adaptation strength
            
        Returns:
            Success status
        """
        # Ensure both SNNs exist
        if source_snn not in self.snns or target_snn not in self.snns:
            return False
            
        # Extract patterns from source
        patterns = self._extract_patterns(source_snn)
        
        if not patterns:
            return False
            
        # Transform patterns for target SNN
        transformed_patterns = self._transform_patterns(
            patterns, source_snn, target_snn
        )
        
        if not transformed_patterns:
            return False
            
        # Apply to target SNN
        return self._apply_patterns(target_snn, transformed_patterns, learning_signal)
    
    def _extract_patterns(self, snn_name: str) -> List[Dict]:
        """
        Extract learned patterns from SNN.
        
        Args:
            snn_name: Name of SNN
            
        Returns:
            List of pattern dictionaries
        """
        snn = self.snns.get(snn_name)
        
        if not snn:
            return []
            
        # Use appropriate method based on SNN type
        if hasattr(snn, 'get_learned_patterns'):
            return snn.get_learned_patterns()
        elif hasattr(snn, 'get_region_activations'):
            # Convert region activations to patterns
            activations = snn.get_region_activations()
            if activations is not None and len(activations) > 0:
                return [{'activations': activations, 'type': 'region_pattern'}]
        elif hasattr(snn, 'get_neuron_activations'):
            # Convert neuron activations to patterns
            activations = snn.get_neuron_activations()
            if activations is not None and len(activations) > 0:
                return [{'activations': activations, 'type': 'neuron_pattern'}]
        
        return []
    
    def _transform_patterns(self, patterns: List[Dict], source_type: str, 
                           target_type: str) -> List[Dict]:
        """
        Transform patterns from source SNN format to target SNN format.
        
        Args:
            patterns: List of pattern dictionaries
            source_type: Source SNN type
            target_type: Target SNN type
            
        Returns:
            List of transformed pattern dictionaries
        """
        transformed = []
        
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            
            # Apply appropriate transformation based on source/target types
            # This is a simplified implementation - real implementation would
            # need more sophisticated transformations based on SNN architectures
            
            # Create a copy with source/target info
            transformed_pattern = pattern.copy()
            transformed_pattern['source_snn'] = source_type
            transformed_pattern['target_snn'] = target_type
            
            # Apply specific transformations if needed
            if source_type == 'statistical' and target_type == 'perceptual':
                # Treat statistical patterns as feature patterns for perceptual SNN
                transformed_pattern['type'] = 'feature_pattern'
            elif source_type == 'statistical' and target_type == 'memory':
                # Treat statistical patterns as memory traces
                transformed_pattern['type'] = 'memory_trace'
            elif source_type == 'decision' and target_type == 'affective':
                # Transform decision patterns to affective biases
                transformed_pattern['type'] = 'affective_bias'
            
            transformed.append(transformed_pattern)
        
        return transformed
    
    def _apply_patterns(self, snn_name: str, patterns: List[Dict], signal: float) -> bool:
        """
        Apply transformed patterns to target SNN.
        
        Args:
            snn_name: Name of target SNN
            patterns: List of transformed pattern dictionaries
            signal: Learning signal guiding adaptation strength
            
        Returns:
            Success status
        """
        snn = self.snns.get(snn_name)
        
        if not snn:
            return False
            
        # Use appropriate method based on SNN type
        if hasattr(snn, 'apply_external_patterns'):
            return snn.apply_external_patterns(patterns, signal)
        elif hasattr(snn, 'integrate_patterns'):
            return snn.integrate_patterns(patterns, signal)
        
        return False
    
    def share_learning_state(self, source_snn: str, target_snn: str) -> bool:
        """
        Share learning state from source SNN to target SNN.
        
        Args:
            source_snn: Name of source SNN
            target_snn: Name of target SNN
            
        Returns:
            Success status
        """
        # Ensure both SNNs exist
        if source_snn not in self.snns or target_snn not in self.snns:
            return False
            
        # Extract learning state from source
        if not hasattr(self.snns[source_snn], 'get_learning_state'):
            return False
            
        learning_state = self.snns[source_snn].get_learning_state()
        
        if not learning_state:
            return False
            
        # Apply to target SNN
        if hasattr(self.snns[target_snn], 'integrate_learning_state'):
            return self.snns[target_snn].integrate_learning_state(learning_state)
        
        return False

class ModularSelfPlay:
    """
    Modular self-play system that can dynamically select and combine
    different SNNs, task generators, and reward calculators.
    """
    
    def __init__(self, snns=None, task_generators=None, reward_calculators=None):
        """
        Initialize the modular self-play system.
        
        Args:
            snns: Dictionary of SNN adapters
            task_generators: Dictionary of task generators
            reward_calculators: Dictionary of reward calculators
        """
        self.snns = snns or {}
        self.task_generators = task_generators or {}
        self.reward_calculators = reward_calculators or {}
        
        # Initialize module selector
        self.module_selector = ModuleSelector(
            self.snns, self.task_generators, self.reward_calculators
        )
        
        # Initialize cross-SNN communicator
        self.communicator = CrossSNNCommunicator(self.snns)
        
        # Training history
        self.training_history = []
        
        # Performance tracking
        self.performance_by_module = defaultdict(list)
        self.active_modules_history = []
    
    def self_play_iteration(self, meta_task=None) -> Dict[str, Any]:
        """
        Execute one iteration of modular self-play.
        
        Args:
            meta_task: Optional task description guiding module selection
            
        Returns:
            Dictionary with results and metrics
        """
        # 1. Select appropriate modules for the task
        active_modules = self.module_selector.select_modules(meta_task)
        
        # 2. Generate task using selected generator
        generator_name = active_modules['generator']
        generator = self.task_generators.get(generator_name)
        
        if not generator:
            raise ValueError(f"Selected generator '{generator_name}' not found")
            
        task = generator.generate_task()
        
        # 3. Process through selected SNNs
        results = {}
        for snn_name in active_modules['snns']:
            snn = self.snns.get(snn_name)
            
            if snn:
                # Encode task input appropriately
                task_input = self._encode_task_for_snn(task, snn_name)
                
                # Process input through SNN
                results[snn_name] = snn.process_input(task_input)
        
        # 4. Calculate rewards using selected calculator
        calculator_name = active_modules['calculator']
        calculator = self.reward_calculators.get(calculator_name)
        
        if not calculator:
            raise ValueError(f"Selected calculator '{calculator_name}' not found")
            
        reward_info = calculator.calculate_comprehensive_reward(task, results)
        
        # 5. Update all participating SNNs
        self._update_snns(active_modules['snns'], task, results, reward_info)
        
        # 6. Facilitate cross-SNN communication
        self._facilitate_cross_snn_communication(active_modules['snns'], reward_info)
        
        # 7. Track performance
        self._track_performance(active_modules, reward_info)
        
        # 8. Build result dictionary
        iteration_result = {
            'task': task,
            'results': results,
            'reward_info': reward_info,
            'active_modules': active_modules
        }
        
        # Add to history
        self.training_history.append(iteration_result)
        self.active_modules_history.append(active_modules)
        
        return iteration_result
    
    def _encode_task_for_snn(self, task: Dict, snn_name: str) -> Any:
        """
        Encode task input appropriately for specific SNN.
        
        Args:
            task: Task dictionary
            snn_name: SNN name
            
        Returns:
            Encoded task input
        """
        snn = self.snns.get(snn_name)
        
        if not snn:
            return None
            
        # Use SNN-specific encoding method if available
        if hasattr(snn, 'encode_task'):
            return snn.encode_task(task)
            
        # General encoding based on SNN type
        task_type = task.get('type', 'unknown')
        task_input = task.get('input', {})
        
        if snn_name == 'statistical':
            # Statistical SNN expects numeric arrays
            if isinstance(task_input, dict):
                if 'features' in task_input:
                    return np.array(task_input['features'])
                elif 'sequence' in task_input:
                    return np.array(task_input['sequence'])
                elif 'vector' in task_input:
                    return np.array(task_input['vector'])
            
            # Fallback: extract all numeric values
            if isinstance(task_input, dict):
                numeric_values = []
                for k, v in task_input.items():
                    if isinstance(v, (int, float)):
                        numeric_values.append(float(v))
                
                if numeric_values:
                    return np.array(numeric_values)
        
        elif snn_name == 'perceptual':
            # Perceptual SNN expects feature dict or array
            if isinstance(task_input, dict) and 'features' in task_input:
                return np.array(task_input['features'])
        
        elif snn_name == 'reasoning':
            # Reasoning SNN expects premises or patterns
            if isinstance(task_input, dict):
                if 'premises' in task_input:
                    return task_input['premises']
                elif 'sequence' in task_input:
                    return np.array(task_input['sequence'])
        
        elif snn_name == 'decision':
            # Decision SNN expects state features
            if isinstance(task_input, dict) and 'state_features' in task_input:
                return np.array(task_input['state_features'])
        
        elif snn_name == 'memory':
            # Memory SNN expects items to remember
            if task_type == 'memory':
                if isinstance(task_input, dict):
                    if 'items' in task_input:
                        return task_input['items']
                    elif 'study_pairs' in task_input:
                        return task_input['study_pairs']
        
        # Default: convert task input to string and hash to vector
        input_str = str(task_input)
        hash_val = hash(input_str) % 10000
        random.seed(hash_val)
        encoded = np.array([random.random() for _ in range(10)])
        random.seed(None)  # Reset seed
        
        return encoded
    
    def _update_snns(self, snn_names: List[str], task: Dict, results: Dict, reward_info: Dict) -> None:
        """
        Update all participating SNNs based on results and rewards.
        
        Args:
            snn_names: List of SNN names
            task: Task dictionary
            results: Results dictionary
            reward_info: Reward information
        """
        combined_reward = reward_info.get('combined_reward', 0.0)
        individual_rewards = reward_info.get('individual_rewards', {})
        
        for snn_name in snn_names:
            snn = self.snns.get(snn_name)
            
            if not snn:
                continue
                
            # Get appropriate reward for this SNN
            reward = individual_rewards.get(snn_name, combined_reward)
            
            try:
                # Use appropriate update method based on SNN type
                if snn_name == 'statistical' and hasattr(snn, 'update_weights'):
                    snn.update_weights(
                        self._encode_task_for_snn(task, snn_name),
                        results.get(snn_name),
                        reward
                    )
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
                elif snn_name == 'decision' and hasattr(snn, 'update_policy'):
                    snn.update_policy(reward)
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
                elif snn_name == 'perceptual' and hasattr(snn, 'update_perception'):
                    snn.update_perception(
                        self._encode_task_for_snn(task, snn_name),
                        task.get('output', {}).get('category'),
                        reward
                    )
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
                elif snn_name == 'memory' and hasattr(snn, 'update_memory'):
                    snn.update_memory(
                        self._encode_task_for_snn(task, snn_name),
                        task.get('output', {}),
                        reward
                    )
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
                elif snn_name == 'affective' and hasattr(snn, 'update_emotional_state'):
                    snn.update_emotional_state({
                        'reward': reward,
                        'task_type': task.get('type')
                    })
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
                elif hasattr(snn, 'update'):
                    # Generic update method
                    snn.update(
                        self._encode_task_for_snn(task, snn_name),
                        results.get(snn_name),
                        reward
                    )
                    print(f"Updated {snn_name}SNN with reward: {reward:.3f}")
            except Exception as e:
                print(f"Warning: Error updating {snn_name}SNN: {e}")
    
    def _facilitate_cross_snn_communication(self, snn_names: List[str], reward_info: Dict) -> None:
        """
        Facilitate communication between different SNNs.
        
        Args:
            snn_names: List of SNN names
            reward_info: Reward information
        """
        combined_reward = reward_info.get('combined_reward', 0.0)
        
        # Only propagate learning on successful trials
        if combined_reward < 0.6:
            return
            
        # Propagate learning between relevant SNN pairs
        relevant_pairs = [
            ('statistical', 'perceptual'),
            ('statistical', 'memory'),
            ('perceptual', 'reasoning'),
            ('decision', 'affective'),
            ('metacognitive', 'decision')
        ]
        
        # Filter to active pairs
        active_pairs = [
            (src, tgt) for src, tgt in relevant_pairs
            if src in snn_names and tgt in snn_names
        ]
        
        # Propagate learning for each pair
        for source, target in active_pairs:
            self.communicator.propagate_learning(source, target, combined_reward)
            
            # Bidirectional for some pairs
            if (source, target) in [('statistical', 'memory'), ('decision', 'affective')]:
                self.communicator.propagate_learning(target, source, combined_reward * 0.8)
            
            # Share learning state for metacognitive pairs
            if source == 'metacognitive' or target == 'metacognitive':
                self.communicator.share_learning_state(source, target)
    
    def _track_performance(self, active_modules: Dict, reward_info: Dict) -> None:
        """
        Track performance of modules.
        
        Args:
            active_modules: Dictionary of active modules
            reward_info: Reward information
        """
        combined_reward = reward_info.get('combined_reward', 0.0)
        
        # Track generator performance
        generator_name = active_modules['generator']
        self.performance_by_module[f"generator:{generator_name}"].append(combined_reward)
        
        # Track calculator performance
        calculator_name = active_modules['calculator']
        self.performance_by_module[f"calculator:{calculator_name}"].append(combined_reward)
        
        # Track SNN performance
        for snn_name in active_modules['snns']:
            reward = reward_info.get('individual_rewards', {}).get(snn_name, combined_reward)
            self.performance_by_module[f"snn:{snn_name}"].append(reward)
        
        # Keep history manageable
        for module, history in self.performance_by_module.items():
            if len(history) > 100:
                self.performance_by_module[module] = history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all modules.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {}
        
        # Calculate stats for each module
        for module, history in self.performance_by_module.items():
            if not history:
                continue
                
            # Calculate average performance
            avg_perf = sum(history) / len(history)
            
            # Calculate trend (last 20 vs previous 20)
            trend = 0.0
            if len(history) >= 40:
                recent_avg = sum(history[-20:]) / 20
                previous_avg = sum(history[-40:-20]) / 20
                trend = recent_avg - previous_avg
            
            stats[module] = {
                'average_performance': avg_perf,
                'trend': trend,
                'samples': len(history)
            }
        
        # Calculate module selection stats
        module_selection_counts = defaultdict(int)
        
        for history in self.active_modules_history:
            generator = history.get('generator')
            calculator = history.get('calculator')
            snns = history.get('snns', [])
            
            if generator:
                module_selection_counts[f"generator:{generator}"] += 1
            
            if calculator:
                module_selection_counts[f"calculator:{calculator}"] += 1
            
            for snn_name in snns:
                module_selection_counts[f"snn:{snn_name}"] += 1
        
        # Add selection counts to stats
        stats['selection_counts'] = dict(module_selection_counts)
        
        # Add overall performance
        all_rewards = []
        for history in self.performance_by_module.values():
            all_rewards.extend(history)
        
        if all_rewards:
            stats['overall'] = {
                'average_performance': sum(all_rewards) / len(all_rewards),
                'samples': len(all_rewards)
            }
        
        return stats 