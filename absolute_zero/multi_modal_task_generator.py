import random
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .improved_task_generator import ImprovedTaskGenerator

class MultiModalTaskGenerator(ImprovedTaskGenerator):
    """
    Extended task generator that creates tasks targeting different SNN modalities.
    This generator can create perceptual, reasoning, decision, and memory tasks
    appropriate for training the corresponding specialized SNNs.
    """
    
    def __init__(self, symbolic_engine=None, vsa_engine=None, snns=None):
        """
        Initialize the multi-modal task generator.
        
        Args:
            symbolic_engine: Optional symbolic reasoning engine
            vsa_engine: Optional Vector Symbolic Architecture engine
            snns: Dictionary of SNN adapters to target
        """
        super().__init__(symbolic_engine, vsa_engine)
        self.snns = snns or {}
        
        # Initialize task counter
        self.task_counter = 0
        
        # Initialize complexity (default value)
        self.complexity = 1.0
        
        # Track available task types based on SNNs
        self.available_task_types = self._determine_available_task_types()
        
        # Task distribution weights (will be adapted based on SNN performance)
        self.task_type_weights = self._initialize_task_weights()
        
        # Track performance on different task types
        self.task_performance = defaultdict(list)
    
    def _determine_available_task_types(self) -> Dict[str, bool]:
        """
        Determine which task types are available based on SNNs.
        
        Returns:
            Dictionary of task types and availability
        """
        task_types = {
            'pattern': True,  # Always available (statistical SNN)
            'classification': True,  # Always available (statistical SNN)
            'association': True,  # Always available (statistical SNN)
            'perceptual': 'perceptual' in self.snns,
            'reasoning': 'reasoning' in self.snns,
            'decision': 'decision' in self.snns,
            'memory': 'memory' in self.snns,
            'metacognitive': 'metacognitive' in self.snns
        }
        return task_types
    
    def _initialize_task_weights(self) -> Dict[str, float]:
        """
        Initialize weights for task type selection.
        
        Returns:
            Dictionary of task types and selection weights
        """
        weights = {}
        
        # Base weights for all task types
        for task_type, available in self.available_task_types.items():
            if available:
                # Start with equal weights for available types
                weights[task_type] = 1.0
            else:
                weights[task_type] = 0.0
        
        return weights
    
    def generate_task(self) -> Dict:
        """
        Generate a task based on weighted selection of task types.
        
        Returns:
            Task dictionary
        """
        # Select task type based on weights
        task_type = self._select_task_type()
        
        # Generate appropriate task for selected type
        if task_type == 'perceptual':
            return self.generate_perceptual_task()
        elif task_type == 'reasoning':
            return self.generate_reasoning_task()
        elif task_type == 'decision':
            return self.generate_decision_task()
        elif task_type == 'memory':
            return self.generate_memory_task()
        elif task_type == 'metacognitive':
            return self.generate_metacognitive_task()
        else:
            # Fall back to parent implementation for standard tasks
            return super().generate_task()
    
    def _select_task_type(self) -> str:
        """
        Select task type based on weights.
        
        Returns:
            Selected task type
        """
        # Get available task types
        available_types = [t for t, available in self.available_task_types.items() if available]
        weights = [self.task_type_weights.get(t, 0.0) for t in available_types]
        
        # Ensure all weights are positive
        if all(w <= 0.0 for w in weights):
            weights = [1.0] * len(available_types)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(available_types)] * len(available_types)
        
        # Select task type
        selected_type = random.choices(available_types, weights=weights, k=1)[0]
        
        return selected_type
    
    def update_task_weights(self, task_type: str, performance: float):
        """
        Update task weights based on performance.
        
        Args:
            task_type: Task type
            performance: Performance on task (0.0 to 1.0)
        """
        # Record performance for this task type
        self.task_performance[task_type].append(performance)
        
        # Keep only last 100 performances for each type
        if len(self.task_performance[task_type]) > 100:
            self.task_performance[task_type] = self.task_performance[task_type][-100:]
        
        # Calculate average performance for each task type
        avg_performance = {}
        for t, scores in self.task_performance.items():
            if scores:
                avg_performance[t] = sum(scores) / len(scores)
            else:
                avg_performance[t] = 0.5  # Default to 0.5 if no data
        
        # Update weights based on inverse performance
        # Lower performing tasks get higher weights to encourage learning
        for t, perf in avg_performance.items():
            if perf < 0.3:
                # Very low performance - strongly prioritize
                self.task_type_weights[t] = 3.0
            elif perf < 0.7:
                # Medium performance - somewhat prioritize
                self.task_type_weights[t] = 2.0
            else:
                # High performance - lower priority
                self.task_type_weights[t] = 1.0
    
    def generate_perceptual_task(self) -> Dict:
        """
        Generate a perceptual task for the PerceptualSNN.
        
        Returns:
            Perceptual task dictionary
        """
        # Create a task ID
        task_id = f"perceptual_{self.task_counter}"
        self.task_counter += 1
        
        # Define categories for perception
        categories = ['circle', 'square', 'triangle', 'line', 'cross']
        
        # Parameters based on complexity
        num_features = 5
        noise_level = min(0.3, self.complexity * 0.1)
        
        # Generate random features for a category
        category = random.choice(categories)
        
        # Create prototype feature vectors for categories
        prototypes = {
            'circle': [0.9, 0.1, 0.1, 0.1, 0.1],
            'square': [0.1, 0.9, 0.1, 0.1, 0.1],
            'triangle': [0.1, 0.1, 0.9, 0.1, 0.1],
            'line': [0.1, 0.1, 0.1, 0.9, 0.1],
            'cross': [0.1, 0.1, 0.1, 0.1, 0.9]
        }
        
        # Generate a feature vector for the selected category
        # Add some noise to make it challenging
        features = []
        for value in prototypes[category]:
            # Add noise based on complexity
            noise = (random.random() * 2 - 1) * noise_level
            feature_val = max(0.0, min(1.0, value + noise))
            features.append(feature_val)
        
        # Create task
        task = {
            'id': task_id,
            'type': 'perceptual',
            'complexity': self.complexity,
            'input': {
                'features': features
            },
            'output': {
                'category': category
            },
            'metadata': {
                'concept': category,
                'feature_meaning': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
                'targets': {
                    'category': category
                }
            }
        }
        
        return task
    
    def generate_reasoning_task(self) -> Dict:
        """
        Generate a reasoning task for the ReasoningSNN.
        
        Returns:
            Reasoning task dictionary
        """
        # Create a task ID
        task_id = f"reasoning_{self.task_counter}"
        self.task_counter += 1
        
        # Define reasoning task types
        reasoning_types = ['deductive', 'inductive', 'abductive']
        
        # Select reasoning type based on complexity
        if self.complexity < 3.0:
            reasoning_type = 'deductive'  # Simpler
        elif self.complexity < 6.0:
            reasoning_type = random.choice(['deductive', 'inductive'])
        else:
            reasoning_type = random.choice(reasoning_types)
        
        # Generate appropriate reasoning task
        if reasoning_type == 'deductive':
            task = self._generate_deductive_task(task_id)
        elif reasoning_type == 'inductive':
            task = self._generate_inductive_task(task_id)
        else:
            task = self._generate_abductive_task(task_id)
        
        return task
    
    def _generate_deductive_task(self, task_id: str) -> Dict:
        """Generate a deductive reasoning task"""
        # Simple syllogism pattern
        entities = ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan']
        properties = ['tall', 'smart', 'kind', 'strong', 'fast']
        
        # Create premise 1: All X are Y
        entity_group = random.choice(entities)
        property1 = random.choice(properties)
        
        # Create premise 2: Z is an X
        entity = random.choice(entities)
        
        # Create task with premises and conclusion
        task = {
            'id': task_id,
            'type': 'reasoning',
            'subtype': 'deductive',
            'complexity': self.complexity,
            'input': {
                'premises': [
                    f"All {entity_group} are {property1}",
                    f"{entity} is a {entity_group}"
                ]
            },
            'output': {
                'conclusion': f"{entity} is {property1}"
            },
            'metadata': {
                'concept': 'deduction',
                'reasoning_type': 'deductive',
                'targets': {
                    'reasoning_target': f"{entity} is {property1}"
                }
            }
        }
        
        return task
    
    def _generate_inductive_task(self, task_id: str) -> Dict:
        """Generate an inductive reasoning task"""
        # Pattern completion task
        pattern_templates = [
            [1, 3, 5, 7, 9],  # Odd numbers
            [2, 4, 6, 8, 10],  # Even numbers
            [1, 2, 4, 8, 16],  # Powers of 2
            [1, 4, 9, 16, 25],  # Perfect squares
            [3, 6, 9, 12, 15]   # Multiples of 3
        ]
        
        # Select a pattern
        pattern = random.choice(pattern_templates).copy()
        
        # Create a partial pattern as input
        input_length = min(4, max(2, int(self.complexity)))
        partial_pattern = pattern[:input_length]
        
        # Expected next element
        next_element = pattern[input_length]
        
        # Create task
        task = {
            'id': task_id,
            'type': 'reasoning',
            'subtype': 'inductive',
            'complexity': self.complexity,
            'input': {
                'sequence': partial_pattern,
                'pattern_type': 'numeric',
            },
            'output': {
                'next_element': next_element
            },
            'metadata': {
                'concept': 'induction',
                'reasoning_type': 'inductive',
                'targets': {
                    'reasoning_target': next_element
                }
            }
        }
        
        return task
    
    def _generate_abductive_task(self, task_id: str) -> Dict:
        """Generate an abductive reasoning task"""
        # Cause inference task
        effects = {
            'wet_ground': ['rain', 'sprinkler', 'water_spill'],
            'lights_off': ['power_outage', 'switch_off', 'broken_bulb'],
            'dog_barking': ['intruder', 'other_animal', 'playing'],
            'late_arrival': ['traffic', 'overslept', 'car_trouble'],
            'plants_dying': ['no_water', 'disease', 'too_much_sun']
        }
        
        # Select an effect
        effect = random.choice(list(effects.keys()))
        
        # Get possible causes
        possible_causes = effects[effect]
        
        # Select actual cause
        actual_cause = random.choice(possible_causes)
        
        # Create additional context clues based on complexity
        context_clues = []
        if self.complexity > 3.0:
            if effect == 'wet_ground' and actual_cause == 'rain':
                context_clues.append('clouds_in_sky')
            elif effect == 'lights_off' and actual_cause == 'power_outage':
                context_clues.append('other_electronics_off')
        
        # Create task
        task = {
            'id': task_id,
            'type': 'reasoning',
            'subtype': 'abductive',
            'complexity': self.complexity,
            'input': {
                'observed_effect': effect,
                'context_clues': context_clues
            },
            'output': {
                'most_likely_cause': actual_cause
            },
            'metadata': {
                'concept': 'abduction',
                'reasoning_type': 'abductive',
                'targets': {
                    'reasoning_target': actual_cause
                }
            }
        }
        
        return task
    
    def generate_decision_task(self) -> Dict:
        """
        Generate a decision task for the DecisionSNN.
        
        Returns:
            Decision task dictionary
        """
        # Create a task ID
        task_id = f"decision_{self.task_counter}"
        self.task_counter += 1
        
        # Define possible actions for decision tasks
        actions = ['left', 'right', 'up', 'down', 'wait']
        
        # Create state features (simplified for demonstration)
        features = []
        for _ in range(5):
            features.append(random.random())
        
        # Determine correct action based on features
        # Simplified logic: action is determined by the highest feature value
        correct_action_idx = np.argmax(features)
        correct_action = actions[correct_action_idx]
        
        # Add some contextual information based on complexity
        context = {}
        if self.complexity > 3.0:
            # Add rewards for each action
            rewards = {action: 0.0 for action in actions}
            rewards[correct_action] = 1.0
            
            # Add some reward for suboptimal actions to make task more nuanced
            for action in actions:
                if action != correct_action:
                    rewards[action] = random.random() * 0.5
            
            context['rewards'] = rewards
        
        # Create task
        task = {
            'id': task_id,
            'type': 'decision',
            'complexity': self.complexity,
            'input': {
                'state_features': features,
                'available_actions': actions,
                'context': context
            },
            'output': {
                'optimal_action': correct_action
            },
            'metadata': {
                'concept': 'action_selection',
                'targets': {
                    'action': correct_action,
                    'confidence': 0.9
                }
            }
        }
        
        return task
    
    def generate_memory_task(self) -> Dict:
        """
        Generate a memory task for the MemorySNN.
        
        Returns:
            Memory task dictionary
        """
        # Create a task ID
        task_id = f"memory_{self.task_counter}"
        self.task_counter += 1
        
        # Define memory task types
        memory_types = ['recall', 'recognition', 'association']
        
        # Select memory type based on complexity
        if self.complexity < 3.0:
            memory_type = 'recall'  # Simpler
        elif self.complexity < 6.0:
            memory_type = random.choice(['recall', 'recognition'])
        else:
            memory_type = random.choice(memory_types)
        
        # Generate appropriate memory task
        if memory_type == 'recall':
            task = self._generate_recall_task(task_id)
        elif memory_type == 'recognition':
            task = self._generate_recognition_task(task_id)
        else:
            task = self._generate_association_memory_task(task_id)
        
        return task
    
    def _generate_recall_task(self, task_id: str) -> Dict:
        """Generate a recall memory task"""
        # Generate items to remember
        items = ['apple', 'banana', 'cherry', 'date', 'elderberry', 
                'fig', 'grape', 'honeydew', 'kiwi', 'lemon']
        
        # Select subset of items based on complexity
        items_count = min(7, max(2, int(self.complexity)))
        selected_items = random.sample(items, items_count)
        
        # Item to recall (randomly selected from the list)
        target_position = random.randint(0, items_count - 1)
        target_item = selected_items[target_position]
        
        # Create task
        task = {
            'id': task_id,
            'type': 'memory',
            'subtype': 'recall',
            'complexity': self.complexity,
            'input': {
                'items': selected_items,
                'recall_position': target_position
            },
            'output': {
                'recalled_item': target_item
            },
            'metadata': {
                'concept': 'recall',
                'memory_type': 'recall',
                'targets': {
                    'memory_target': target_item
                }
            }
        }
        
        return task
    
    def _generate_recognition_task(self, task_id: str) -> Dict:
        """Generate a recognition memory task"""
        # Generate items to remember
        items = ['dog', 'cat', 'fish', 'bird', 'snake', 
                'rabbit', 'hamster', 'turtle', 'lizard', 'frog']
        
        # Select subset of items based on complexity
        items_count = min(7, max(3, int(self.complexity)))
        study_items = random.sample(items, items_count)
        
        # Generate test items (some from study list, some new)
        test_set_size = min(5, items_count)
        include_count = random.randint(1, test_set_size)
        exclude_count = test_set_size - include_count
        
        # Select items from study list
        test_items_from_study = random.sample(study_items, include_count)
        
        # Select items not in study list
        new_items = [item for item in items if item not in study_items]
        test_items_new = random.sample(new_items, exclude_count)
        
        # Combine and shuffle
        test_items = test_items_from_study + test_items_new
        random.shuffle(test_items)
        
        # Expected recognition results
        expected_recognition = {item: item in study_items for item in test_items}
        
        # Create task
        task = {
            'id': task_id,
            'type': 'memory',
            'subtype': 'recognition',
            'complexity': self.complexity,
            'input': {
                'study_items': study_items,
                'test_items': test_items
            },
            'output': {
                'recognition_results': expected_recognition
            },
            'metadata': {
                'concept': 'recognition',
                'memory_type': 'recognition',
                'targets': {
                    'memory_target': expected_recognition
                }
            }
        }
        
        return task
    
    def _generate_association_memory_task(self, task_id: str) -> Dict:
        """Generate an associative memory task"""
        # Generate pairs to remember
        word_pairs = [
            ('bread', 'butter'),
            ('pen', 'paper'),
            ('salt', 'pepper'),
            ('knife', 'fork'),
            ('shoe', 'sock'),
            ('hammer', 'nail'),
            ('lock', 'key'),
            ('needle', 'thread'),
            ('cup', 'saucer'),
            ('bow', 'arrow')
        ]
        
        # Select subset of pairs based on complexity
        pairs_count = min(7, max(2, int(self.complexity)))
        selected_pairs = random.sample(word_pairs, pairs_count)
        
        # Create study list
        study_pairs = {first: second for first, second in selected_pairs}
        
        # Select a pair for testing
        test_pair_idx = random.randint(0, pairs_count - 1)
        test_first, test_second = selected_pairs[test_pair_idx]
        
        # Create task
        task = {
            'id': task_id,
            'type': 'memory',
            'subtype': 'association',
            'complexity': self.complexity,
            'input': {
                'study_pairs': study_pairs,
                'test_cue': test_first
            },
            'output': {
                'associated_item': test_second
            },
            'metadata': {
                'concept': 'association',
                'memory_type': 'association',
                'targets': {
                    'memory_target': test_second
                }
            }
        }
        
        return task
    
    def generate_metacognitive_task(self) -> Dict:
        """
        Generate a metacognitive task for the MetacognitiveSNN.
        
        Returns:
            Metacognitive task dictionary
        """
        # Create a task ID
        task_id = f"metacognitive_{self.task_counter}"
        self.task_counter += 1
        
        # Generate a base task using any other task type
        base_task_type = random.choice(['pattern', 'classification', 'reasoning', 'memory', 'decision'])
        
        if base_task_type == 'pattern':
            base_task = super().generate_pattern_task()
        elif base_task_type == 'classification':
            base_task = super().generate_classification_task()
        elif base_task_type == 'reasoning':
            base_task = self.generate_reasoning_task()
        elif base_task_type == 'memory':
            base_task = self.generate_memory_task()
        else:
            base_task = self.generate_decision_task()
        
        # Add noise/difficulty based on complexity
        difficulty_level = min(1.0, self.complexity / 10.0)
        
        # Calculate expected confidence (inverse of difficulty)
        expected_confidence = 1.0 - difficulty_level
        
        # Create metacognitive task
        task = {
            'id': task_id,
            'type': 'metacognitive',
            'complexity': self.complexity,
            'input': {
                'base_task': base_task,
                'difficulty_level': difficulty_level
            },
            'output': {
                'expected_confidence': expected_confidence,
                'base_task_output': base_task['output']
            },
            'metadata': {
                'concept': 'confidence_estimation',
                'base_task_type': base_task['type'],
                'targets': {
                    'confidence': expected_confidence
                }
            }
        }
        
        return task 