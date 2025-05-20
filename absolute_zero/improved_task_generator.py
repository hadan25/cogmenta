"""
Improved Task Generator for Absolute Zero framework.

This module provides an improved task generator that creates tasks
better suited for SNN learning, with clear conceptual patterns
and meaningful reward signals.
"""

import random
import numpy as np
import time
from typing import Dict, Tuple, Any, List, Optional

# These imports should be properly configured based on your project structure
try:
    # Try importing directly if they're in the Python path
    from models.symbolic.prolog_engine import PrologEngine as SymbolicEngine
    from models.symbolic.vector_symbolic import VectorSymbolicEngine as VSAEngine
except ImportError:
    # Fallback - these will be provided via constructor if imports fail
    pass

# Mock VSA engine for fallback
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

class ImprovedTaskGenerator:
    """
    Improved task generator with better conceptual patterns and reward potential.
    
    This generator creates tasks that have clearer learning trajectories
    and conceptual patterns, making them more suitable for SNN learning.
    """
    
    def __init__(self, symbolic_engine=None, vsa_engine=None):
        """
        Initialize the task generator with symbolic and VSA engines
        
        Args:
            symbolic_engine: Symbolic reasoning engine (e.g., Prolog-based)
            vsa_engine: Vector Symbolic Architecture engine for encoding
        """
        # Use provided symbolic engine or create a simple mock
        if symbolic_engine is None:
            class MockSymbolicEngine:
                def deduce(self, rules, facts):
                    return "mock_conclusion"
                    
                def verify(self, rules, facts, conclusion):
                    return random.random() > 0.3
            self.symbolic_engine = MockSymbolicEngine()
        else:
            self.symbolic_engine = symbolic_engine
        
        # Use provided VSA engine or fallback to mock implementation
        self.vector_symbolic = vsa_engine if vsa_engine is not None else MockVSAEngine()
        
        # Task generation parameters
        self.complexity_level = 1.0
        self.task_types = ['pattern', 'classification', 'sequence', 'association']
        
        # Concept pools for building tasks around consistent patterns
        self.initialize_concept_pools()
        
        # Track concept usage for curriculum
        self.concept_usage = {}
        self.concept_success = {}
        
        # Task history for analyzing performance trends
        self.task_history = []
        
        # Random seed to ensure reproducibility when needed
        self.seed_val = int(time.time())
    
    def initialize_concept_pools(self):
        """Initialize pools of concepts for various tasks"""
        # Patterns for pattern recognition tasks
        self.pattern_concepts = {
            'alternating': {'pattern': [1, 0, 1, 0], 'description': 'alternating binary values'},
            'increasing': {'pattern': [1, 2, 3, 4], 'description': 'steadily increasing values'},
            'decreasing': {'pattern': [4, 3, 2, 1], 'description': 'steadily decreasing values'},
            'oscillating': {'pattern': [1, 3, 2, 4], 'description': 'oscillating values'},
            'fibonacci': {'pattern': [1, 1, 2, 3, 5], 'description': 'fibonacci sequence'},
            'squares': {'pattern': [1, 4, 9, 16], 'description': 'square numbers'},
            'cubes': {'pattern': [1, 8, 27, 64], 'description': 'cube numbers'},
            'primes': {'pattern': [2, 3, 5, 7, 11], 'description': 'prime numbers'},
            'powers_of_2': {'pattern': [1, 2, 4, 8, 16], 'description': 'powers of 2'},
            'even': {'pattern': [2, 4, 6, 8], 'description': 'even numbers'},
            'odd': {'pattern': [1, 3, 5, 7], 'description': 'odd numbers'}
        }
        
        # Categories for classification tasks
        self.classification_concepts = {
            'animals': {
                'mammals': ['dog', 'cat', 'horse', 'elephant', 'lion', 'bear'],
                'birds': ['eagle', 'sparrow', 'penguin', 'hawk', 'robin'],
                'reptiles': ['snake', 'lizard', 'turtle', 'crocodile', 'alligator'],
                'fish': ['salmon', 'tuna', 'shark', 'goldfish', 'bass']
            },
            'colors': {
                'warm': ['red', 'orange', 'yellow', 'pink', 'brown'],
                'cool': ['blue', 'green', 'purple', 'teal', 'cyan']
            },
            'shapes': {
                '2d': ['circle', 'square', 'triangle', 'rectangle', 'pentagon'],
                '3d': ['sphere', 'cube', 'pyramid', 'cylinder', 'cone']
            }
        }
        
        # Association pairs for mapping tasks
        self.association_concepts = {
            'opposites': [
                ('hot', 'cold'), ('big', 'small'), ('fast', 'slow'),
                ('up', 'down'), ('left', 'right'), ('light', 'dark')
            ],
            'related': [
                ('boat', 'water'), ('car', 'road'), ('bird', 'sky'),
                ('fish', 'sea'), ('tree', 'forest'), ('book', 'read')
            ],
            'causes': [
                ('rain', 'wet'), ('fire', 'heat'), ('sugar', 'sweet'),
                ('ice', 'cold'), ('sleep', 'rest'), ('run', 'tired')
            ]
        }
    
    def generate_task(self, target_type=None) -> Dict:
        """
        Generate a task of specified or random type based on current complexity
        
        Args:
            target_type: Optional task type to generate
            
        Returns:
            Task definition dictionary
        """
        # Select task type, either specified or random
        task_type = target_type if target_type in self.task_types else random.choice(self.task_types)
        
        # Generate task by type
        if task_type == 'pattern':
            task = self.generate_pattern_task()
        elif task_type == 'classification':
            task = self.generate_classification_task()
        elif task_type == 'sequence':
            task = self.generate_sequence_task()
        else:  # association
            task = self.generate_association_task()
            
        # Add task metadata
        task['type'] = task_type
        task['complexity'] = self.complexity_level
        task['id'] = f"{task_type}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Add to task history
        self.task_history.append({'id': task['id'], 'type': task_type, 'success': None})
        
        return task
        
    def generate_pattern_task(self) -> Dict:
        """
        Generate a pattern recognition task with clear conceptual basis.
        
        Returns:
            Task definition dictionary
        """
        # Select a pattern concept, preferring less-used concepts
        pattern_name = self.select_concept(list(self.pattern_concepts.keys()))
        pattern_info = self.pattern_concepts[pattern_name]
        base_pattern = pattern_info['pattern']
        description = pattern_info['description']
        
        # Scale difficulty based on complexity level
        pattern_length = max(4, min(10, int(3 + self.complexity_level)))
        
        # Generate full pattern by extending the base pattern as needed
        full_pattern = []
        while len(full_pattern) < pattern_length:
            full_pattern.extend(base_pattern)
        full_pattern = full_pattern[:pattern_length]
        
        # Create input pattern (missing the last element)
        input_pattern = full_pattern[:-1]
        
        # The output to predict is the last element
        output_element = full_pattern[-1]
        
        # Add some noise or transformations based on complexity
        if self.complexity_level > 3:
            # Transform the pattern (scaling, shifting) while preserving structure
            transform_type = random.choice(['scale', 'shift', 'both'])
            
            if transform_type in ['scale', 'both']:
                scale_factor = random.randint(2, 5)
                full_pattern = [x * scale_factor for x in full_pattern]
                input_pattern = [x * scale_factor for x in input_pattern]
                output_element = output_element * scale_factor
                
            if transform_type in ['shift', 'both']:
                shift_amount = random.randint(1, 10)
                full_pattern = [x + shift_amount for x in full_pattern]
                input_pattern = [x + shift_amount for x in input_pattern]
                output_element = output_element + shift_amount
        
        # Format task
        task = {
            "input": {
                "sequence": input_pattern,
                "description": description
            },
            "output": {
                "next_element": output_element
            },
            "metadata": {
                "concept": pattern_name,
                "original_pattern": base_pattern
            }
        }
        
        # Update concept usage
        self.increment_concept_usage(pattern_name)
        
        return task
    
    def generate_classification_task(self) -> Dict:
        """
        Generate a classification task with clear category boundaries.
        
        Returns:
            Task definition dictionary
        """
        # Select a classification domain
        domain = random.choice(list(self.classification_concepts.keys()))
        
        # Select two categories within the domain
        available_categories = list(self.classification_concepts[domain].keys())
        if len(available_categories) < 2:
            # Fallback if not enough categories
            domain = random.choice(list(self.classification_concepts.keys()))
            available_categories = list(self.classification_concepts[domain].keys())
        
        cat1, cat2 = random.sample(available_categories, 2)
        
        # Number of examples based on complexity
        num_examples = max(3, min(10, int(2 + self.complexity_level)))
        
        # Create examples from both categories
        examples = []
        correct_category = random.choice([cat1, cat2])
        
        for i in range(num_examples):
            # Alternating examples from each category
            category = cat1 if i % 2 == 0 else cat2
            item = random.choice(self.classification_concepts[domain][category])
            
            examples.append({
                "item": item,
                "category": category
            })
            
        # Create a test item from the correct category
        test_item = random.choice(self.classification_concepts[domain][correct_category])
        
        # Ensure test item doesn't appear in examples
        while any(example["item"] == test_item for example in examples):
            test_item = random.choice(self.classification_concepts[domain][correct_category])
        
        # Format task
        task = {
            "input": {
                "domain": domain,
                "examples": examples,
                "test_item": test_item
            },
            "output": {
                "category": correct_category
            },
            "metadata": {
                "concept": f"{domain}_{correct_category}",
                "categories": [cat1, cat2]
            }
        }
        
        # Update concept usage
        self.increment_concept_usage(f"{domain}_{correct_category}")
        
        return task
    
    def generate_sequence_task(self) -> Dict:
        """
        Generate a numeric sequence task with a clear pattern.
        
        Returns:
            Task definition dictionary
        """
        # Use pattern concepts as the basis for sequences
        pattern_name = self.select_concept(list(self.pattern_concepts.keys()))
        base_pattern = self.pattern_concepts[pattern_name]['pattern']
        
        # Scale difficulty based on complexity level
        sequence_length = max(4, min(10, int(3 + self.complexity_level)))
        
        # Select a sequence style based on complexity
        if self.complexity_level < 3:
            # Simple arithmetic or geometric sequence
            style = random.choice(['arithmetic', 'geometric'])
            
            if style == 'arithmetic':
                # Arithmetic sequence: a, a+d, a+2d, ...
                start = random.randint(1, 10)
                diff = random.randint(1, 5)
                sequence = [start + i * diff for i in range(sequence_length)]
                
                next_element = sequence[-1] + diff
                rule = f"Add {diff} to each element"
                
            else:  # geometric
                # Geometric sequence: a, a*r, a*r^2, ...
                start = random.randint(1, 5)
                ratio = random.randint(2, 4)
                sequence = [start * (ratio ** i) for i in range(sequence_length)]
                
                next_element = sequence[-1] * ratio
                rule = f"Multiply each element by {ratio}"
                
        else:
            # More complex patterns
            style = random.choice(['polynomial', 'recursive', 'pattern_based'])
            
            if style == 'polynomial':
                # Quadratic or cubic
                degree = random.randint(2, 3)
                coeffs = [random.randint(1, 3) for _ in range(degree + 1)]
                
                def poly(x):
                    return sum(coef * (x ** power) for power, coef in enumerate(coeffs))
                
                sequence = [poly(i+1) for i in range(sequence_length)]
                next_element = poly(sequence_length + 1)
                
                if degree == 2:
                    rule = f"Quadratic sequence with coefficients {coeffs}"
                else:
                    rule = f"Cubic sequence with coefficients {coeffs}"
                
            elif style == 'recursive':
                # Recursive pattern like Fibonacci
                sequence = [random.randint(1, 5), random.randint(1, 5)]
                
                recursion_type = random.choice(['sum', 'product', 'difference'])
                
                for i in range(sequence_length - 2):
                    if recursion_type == 'sum':
                        sequence.append(sequence[-1] + sequence[-2])
                    elif recursion_type == 'product':
                        sequence.append(sequence[-1] * sequence[-2] % 100)  # Mod to keep numbers manageable
                    else:  # difference
                        sequence.append(abs(sequence[-1] - sequence[-2]))
                
                if recursion_type == 'sum':
                    next_element = sequence[-1] + sequence[-2]
                    rule = "Each element is the sum of the two previous elements"
                elif recursion_type == 'product':
                    next_element = sequence[-1] * sequence[-2] % 100
                    rule = "Each element is the product of the two previous elements (mod 100)"
                else:
                    next_element = abs(sequence[-1] - sequence[-2])
                    rule = "Each element is the absolute difference of the two previous elements"
                
            else:  # pattern_based
                # Use the base pattern but extend it
                sequence = []
                while len(sequence) < sequence_length:
                    sequence.extend(base_pattern)
                sequence = sequence[:sequence_length]
                
                # The next element follows the pattern
                base_idx = sequence_length % len(base_pattern)
                next_element = base_pattern[base_idx]
                
                rule = f"Pattern repeats: {base_pattern}"
        
        # Format task
        task = {
            "input": {
                "sequence": sequence[:-1],  # Leave off the last element
                "style": style
            },
            "output": {
                "next_element": sequence[-1],  # Current last element is what should be predicted
                "future_element": next_element  # Next element (for curriculum assessment)
            },
            "metadata": {
                "concept": pattern_name if style == 'pattern_based' else style,
                "rule": rule
            }
        }
        
        # Update concept usage
        self.increment_concept_usage(pattern_name if style == 'pattern_based' else style)
        
        return task
    
    def generate_association_task(self) -> Dict:
        """
        Generate an association task where items need to be mapped.
        
        Returns:
            Task definition dictionary
        """
        # Select an association type
        assoc_type = random.choice(list(self.association_concepts.keys()))
        pairs = self.association_concepts[assoc_type]
        
        # Number of examples based on complexity
        num_examples = max(2, min(len(pairs) - 1, int(2 + self.complexity_level)))
        
        # Select pairs for examples and test
        selected_pairs = random.sample(pairs, num_examples + 1)
        example_pairs = selected_pairs[:-1]
        test_pair = selected_pairs[-1]
        
        # Format examples and test
        examples = []
        for first, second in example_pairs:
            examples.append({
                "first": first,
                "second": second
            })
        
        test_item = test_pair[0]
        correct_response = test_pair[1]
        
        # Format task
        task = {
            "input": {
                "association_type": assoc_type,
                "examples": examples,
                "test_item": test_item
            },
            "output": {
                "associated_item": correct_response
            },
            "metadata": {
                "concept": assoc_type,
                "full_pair": test_pair
            }
        }
        
        # Update concept usage
        self.increment_concept_usage(assoc_type)
        
        return task
    
    def select_concept(self, concepts):
        """
        Select a concept, biasing toward less-used concepts for better training.
        
        Args:
            concepts: List of concept keys to choose from
            
        Returns:
            Selected concept key
        """
        # Initialize usage stats for any new concepts
        for concept in concepts:
            if concept not in self.concept_usage:
                self.concept_usage[concept] = 0
                self.concept_success[concept] = 0.0
        
        if random.random() < 0.7:
            # 70% of the time, choose based on least usage
            concept_usage = [(concept, self.concept_usage[concept]) for concept in concepts]
            concept_usage.sort(key=lambda x: x[1])  # Sort by usage (ascending)
            
            # Select from the least used 50% of concepts
            top_half = max(1, len(concept_usage) // 2)
            return concept_usage[random.randint(0, top_half-1)][0]
        else:
            # 30% of the time, choose randomly
            return random.choice(concepts)
    
    def increment_concept_usage(self, concept):
        """
        Increment the usage counter for a concept.
        
        Args:
            concept: The concept key
        """
        if concept not in self.concept_usage:
            self.concept_usage[concept] = 0
            self.concept_success[concept] = 0.0
        
        self.concept_usage[concept] += 1
    
    def update_concept_success(self, concept, success):
        """
        Update the success rate for a concept.
        
        Args:
            concept: The concept key
            success: Boolean or float indicating success (0.0-1.0)
        """
        if concept not in self.concept_success:
            self.concept_usage[concept] = 1
            self.concept_success[concept] = 0.0
        
        # Convert boolean to float if needed
        if isinstance(success, bool):
            success = 1.0 if success else 0.0
        
        # Calculate running average
        usage = self.concept_usage[concept]
        current = self.concept_success[concept]
        
        # Update with new success value (weighted average)
        self.concept_success[concept] = current * (usage - 1) / usage + success / usage
    
    def record_task_result(self, task_id, success):
        """
        Record the success/failure of a task for curriculum planning.
        
        Args:
            task_id: The ID of the task
            success: Boolean or float indicating success (0.0-1.0)
        """
        # Find the task in history
        for task in self.task_history:
            if task['id'] == task_id:
                task['success'] = success
                
                # If the task has concept metadata, update concept success
                if 'metadata' in task and 'concept' in task['metadata']:
                    self.update_concept_success(task['metadata']['concept'], success)
                break
    
    def adjust_complexity(self, success_rate, min_adjust=0.9, max_adjust=1.1):
        """
        Adjust complexity based on success rate.
        
        Args:
            success_rate: Rate of successful task completion (0.0-1.0)
            min_adjust: Minimum adjustment factor
            max_adjust: Maximum adjustment factor
        """
        # Calculate adjustment factor: more success = higher complexity
        # Keep changes gradual for stability
        adjustment = min_adjust + (max_adjust - min_adjust) * success_rate
        
        # Apply adjustment with limits
        self.complexity_level = max(0.5, min(10.0, self.complexity_level * adjustment))
        
        return self.complexity_level
    
    def get_complexity(self):
        """Get the current complexity level"""
        return self.complexity_level
    
    def set_complexity(self, complexity):
        """
        Set the complexity level directly.
        
        Args:
            complexity: New complexity level
        """
        self.complexity_level = max(0.5, min(10.0, complexity))
        return self.complexity_level
    
    def get_concept_stats(self):
        """
        Get statistics about concept usage and success.
        
        Returns:
            Dictionary with concept statistics
        """
        stats = {
            'usage': self.concept_usage.copy(),
            'success': self.concept_success.copy()
        }
        
        # Add derived statistics
        stats['least_used'] = sorted(self.concept_usage.items(), key=lambda x: x[1])[:5]
        stats['most_used'] = sorted(self.concept_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['most_successful'] = sorted(self.concept_success.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['least_successful'] = sorted(self.concept_success.items(), key=lambda x: x[1])[:5]
        
        return stats
    
    def set_seed(self, seed):
        """
        Set a random seed for reproducible task generation.
        
        Args:
            seed: Integer seed value
        """
        self.seed_val = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def reset_seed(self):
        """Reset the random seed to variable behavior"""
        random.seed(None)
        np.random.seed(None) 