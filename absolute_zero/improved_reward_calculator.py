"""
Improved Reward Calculator for Absolute Zero framework.

This module provides an improved reward calculator that gives more meaningful
and granular rewards to facilitate better SNN learning.
"""

import numpy as np
from typing import Dict, Any, List, Union, Tuple, Optional
import math

class ImprovedRewardCalculator:
    """
    Improved reward calculator with more granular and meaningful rewards.
    
    This calculator provides various reward signals based on prediction accuracy,
    conceptual understanding, learning progress, and task complexity.
    """
    
    def __init__(self):
        """Initialize the reward calculator with history tracking"""
        # Track history of task results for measuring progress
        self.task_history = {}
        self.concept_history = {}
        
        # Task type specific reward functions
        self.task_reward_functions = {
            'pattern': self.calculate_pattern_reward,
            'classification': self.calculate_classification_reward,
            'sequence': self.calculate_sequence_reward,
            'association': self.calculate_association_reward,
            # Fallbacks for standard Absolute Zero tasks
            'deduction': self.calculate_deduction_reward,
            'abduction': self.calculate_abduction_reward,
            'induction': self.calculate_induction_reward,
            'pattern_recognition': self.calculate_pattern_recognition_reward
        }
    
    def calculate_reward(self, task: Dict[str, Any], 
                         prediction: Any, 
                         task_id: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate a comprehensive reward for a task prediction.
        
        Args:
            task: The task definition
            prediction: The model's prediction
            task_id: Optional task identifier for history tracking
            
        Returns:
            Dictionary with multiple reward components and combined reward
        """
        # Get task type and ensure it exists
        task_type = task.get('type', 'unknown')
        
        # Use specific reward function based on task type
        if task_type in self.task_reward_functions:
            reward_components = self.task_reward_functions[task_type](task, prediction)
        else:
            # Fallback to basic accuracy reward
            accuracy = float(self.verify_output(prediction, task.get('output', {})))
            reward_components = {
                'accuracy': accuracy,
                'learning': 0.0,
                'novelty': 0.0
            }
        
        # Calculate combined reward - weighted sum of components
        combined_reward = (
            0.6 * reward_components.get('accuracy', 0.0) +
            0.3 * reward_components.get('learning', 0.0) +
            0.1 * reward_components.get('novelty', 0.0)
        )
        
        # Add combined reward to components
        reward_components['combined_reward'] = combined_reward
        
        # Track this result if task_id is provided
        if task_id is not None:
            self.update_history(task, reward_components, task_id)
        
        return reward_components
    
    def calculate_pattern_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """
        Calculate reward for pattern recognition tasks.
        
        Args:
            task: The pattern task
            prediction: The model's prediction
            
        Returns:
            Dictionary with reward components
        """
        # Get ground truth and input
        ground_truth = task.get('output', {})
        next_element = ground_truth.get('next_element', None)
        
        # Get prediction value
        if isinstance(prediction, dict):
            predicted_element = prediction.get('next_element', None)
        else:
            predicted_element = prediction
        
        # Basic accuracy component
        if next_element is not None and predicted_element is not None:
            # Exact match
            exact_match = predicted_element == next_element
            
            # Near match (for numeric predictions)
            if isinstance(next_element, (int, float)) and isinstance(predicted_element, (int, float)):
                # Calculate how close the prediction is
                max_val = max(abs(next_element), 1)  # Avoid division by zero
                normalized_diff = abs(predicted_element - next_element) / max_val
                closeness = max(0, 1 - normalized_diff)  # 1 for perfect, 0 for far off
                
                # Accuracy is 1.0 for exact match, scaled by closeness otherwise
                accuracy = 1.0 if exact_match else closeness
            else:
                # Non-numeric prediction - binary accuracy
                accuracy = 1.0 if exact_match else 0.0
        else:
            accuracy = 0.0
        
        # Learning component - reward if the prediction follows the correct pattern
        # but might not be numerically exact
        learning = 0.0
        
        # Get pattern concept and original pattern
        metadata = task.get('metadata', {})
        concept = metadata.get('concept', '')
        original_pattern = metadata.get('original_pattern', [])
        
        if original_pattern and concept:
            # Check if the prediction is consistent with the pattern
            input_sequence = task.get('input', {}).get('sequence', [])
            if input_sequence:
                # Infer what the pattern would predict
                pattern_type = concept
                
                if pattern_type == 'alternating':
                    # Check if the predicted element alternates with the last input
                    last_input = input_sequence[-1]
                    expected_alternation = not bool(last_input) if isinstance(last_input, (bool, int)) and last_input in (0, 1) else None
                    
                    if expected_alternation is not None and predicted_element is not None:
                        learning = 0.8 if bool(predicted_element) == expected_alternation else 0.0
                
                elif pattern_type in ('increasing', 'decreasing'):
                    # Check if the trend is correct
                    if len(input_sequence) >= 2:
                        increasing = input_sequence[-1] > input_sequence[-2]
                        if (increasing and pattern_type == 'increasing') or (not increasing and pattern_type == 'decreasing'):
                            if isinstance(predicted_element, (int, float)) and isinstance(input_sequence[-1], (int, float)):
                                correct_direction = (predicted_element > input_sequence[-1]) == increasing
                                learning = 0.8 if correct_direction else 0.0
                
                elif pattern_type == 'fibonacci' and len(input_sequence) >= 2:
                    # Check if the prediction is close to the sum of the last two inputs
                    expected_sum = input_sequence[-1] + input_sequence[-2]
                    if isinstance(predicted_element, (int, float)):
                        normalized_diff = abs(predicted_element - expected_sum) / max(expected_sum, 1)
                        learning = max(0, 0.8 * (1 - normalized_diff))
                
                elif pattern_type == 'powers_of_2' and len(input_sequence) >= 1:
                    # Check if the prediction is close to doubling the last input
                    expected = input_sequence[-1] * 2
                    if isinstance(predicted_element, (int, float)):
                        normalized_diff = abs(predicted_element - expected) / max(expected, 1)
                        learning = max(0, 0.8 * (1 - normalized_diff))
        
        # Novelty component - reward for solving new pattern types
        # Track how often this pattern type has been presented
        novelty = 0.0
        if concept in self.concept_history:
            # Less novelty if we've seen this concept a lot
            exposure_count = self.concept_history.get(concept, {}).get('count', 0)
            novelty = max(0.0, 1.0 - (exposure_count / 10.0))
        else:
            # First time seeing this concept
            novelty = 1.0
        
        return {
            'accuracy': accuracy,
            'learning': learning,
            'novelty': novelty
        }
    
    def calculate_classification_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """
        Calculate reward for classification tasks.
        
        Args:
            task: The classification task
            prediction: The model's prediction
            
        Returns:
            Dictionary with reward components
        """
        # Get ground truth
        ground_truth = task.get('output', {})
        correct_category = ground_truth.get('category', None)
        
        # Get prediction value
        if isinstance(prediction, dict):
            predicted_category = prediction.get('category', None)
        else:
            predicted_category = prediction
        
        # Basic accuracy component - exact match of category
        accuracy = 1.0 if predicted_category == correct_category else 0.0
        
        # Learning component - reward for conceptual understanding
        learning = 0.0
        
        # Check if prediction is consistent with examples
        input_data = task.get('input', {})
        examples = input_data.get('examples', [])
        test_item = input_data.get('test_item', None)
        domain = input_data.get('domain', None)
        
        # Get possible categories from examples
        categories = set()
        for example in examples:
            if isinstance(example, dict) and 'category' in example:
                categories.add(example['category'])
        
        if predicted_category in categories:
            # The prediction is at least a valid category from examples
            learning = 0.5
            
            # Check if metadata has concept information
            metadata = task.get('metadata', {})
            if 'categories' in metadata:
                valid_categories = metadata.get('categories', [])
                if predicted_category in valid_categories:
                    learning = 0.7  # Higher reward for selecting a valid category
        
        # Novelty component - reward for classifying new domains/categories
        novelty = 0.0
        concept = task.get('metadata', {}).get('concept', '')
        
        if concept in self.concept_history:
            # Less novelty if we've seen this concept a lot
            exposure_count = self.concept_history.get(concept, {}).get('count', 0)
            novelty = max(0.0, 1.0 - (exposure_count / 10.0))
        else:
            # First time seeing this concept
            novelty = 1.0
        
        return {
            'accuracy': accuracy,
            'learning': learning,
            'novelty': novelty
        }
    
    def calculate_sequence_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """
        Calculate reward for sequence tasks.
        
        Args:
            task: The sequence task
            prediction: The model's prediction
            
        Returns:
            Dictionary with reward components
        """
        # Get ground truth
        ground_truth = task.get('output', {})
        next_element = ground_truth.get('next_element', None)
        future_element = ground_truth.get('future_element', None)
        
        # Get prediction value
        if isinstance(prediction, dict):
            predicted_element = prediction.get('next_element', None)
        else:
            predicted_element = prediction
        
        # Basic accuracy component
        if next_element is not None and predicted_element is not None:
            # Exact match
            exact_match = predicted_element == next_element
            
            # Near match (for numeric predictions)
            if isinstance(next_element, (int, float)) and isinstance(predicted_element, (int, float)):
                # Calculate how close the prediction is
                max_val = max(abs(next_element), 1)  # Avoid division by zero
                normalized_diff = abs(predicted_element - next_element) / max_val
                closeness = max(0, 1 - normalized_diff)  # 1 for perfect, 0 for far off
                
                # Accuracy is 1.0 for exact match, scaled by closeness otherwise
                accuracy = 1.0 if exact_match else closeness
            else:
                # Non-numeric prediction - binary accuracy
                accuracy = 1.0 if exact_match else 0.0
        else:
            accuracy = 0.0
        
        # Learning component - reward for understanding the sequence rule
        learning = 0.0
        metadata = task.get('metadata', {})
        rule = metadata.get('rule', '')
        
        # Check if prediction follows the rule
        if rule and 'input' in task:
            input_sequence = task.get('input', {}).get('sequence', [])
            style = task.get('input', {}).get('style', '')
            
            if style == 'arithmetic' and len(input_sequence) >= 2:
                # Check if prediction follows arithmetic pattern
                diff = input_sequence[-1] - input_sequence[-2]
                expected = input_sequence[-1] + diff
                
                if isinstance(predicted_element, (int, float)):
                    normalized_diff = abs(predicted_element - expected) / max(abs(expected), 1)
                    learning = max(0, 0.8 * (1 - normalized_diff))
            
            elif style == 'geometric' and len(input_sequence) >= 2:
                # Check if prediction follows geometric pattern
                if input_sequence[-2] != 0:  # Avoid division by zero
                    ratio = input_sequence[-1] / input_sequence[-2]
                    expected = input_sequence[-1] * ratio
                    
                    if isinstance(predicted_element, (int, float)):
                        normalized_diff = abs(predicted_element - expected) / max(abs(expected), 1)
                        learning = max(0, 0.8 * (1 - normalized_diff))
            
            elif style == 'pattern_based':
                # Use similar calculation as pattern task
                concept = metadata.get('concept', '')
                return self.calculate_pattern_reward(task, prediction)
        
        # Novelty component - reward for solving new sequence types
        novelty = 0.0
        concept = task.get('metadata', {}).get('concept', '')
        
        if concept in self.concept_history:
            # Less novelty if we've seen this concept a lot
            exposure_count = self.concept_history.get(concept, {}).get('count', 0)
            novelty = max(0.0, 1.0 - (exposure_count / 10.0))
        else:
            # First time seeing this concept
            novelty = 1.0
        
        return {
            'accuracy': accuracy,
            'learning': learning,
            'novelty': novelty
        }
    
    def calculate_association_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """
        Calculate reward for association tasks.
        
        Args:
            task: The association task
            prediction: The model's prediction
            
        Returns:
            Dictionary with reward components
        """
        # Get ground truth
        ground_truth = task.get('output', {})
        associated_item = ground_truth.get('associated_item', None)
        
        # Get prediction value
        if isinstance(prediction, dict):
            predicted_item = prediction.get('associated_item', None)
        else:
            predicted_item = prediction
        
        # Basic accuracy component - exact match
        accuracy = 1.0 if predicted_item == associated_item else 0.0
        
        # Learning component - reward for associated item being consistent with examples
        learning = 0.0
        
        # Check if prediction is consistent with association pattern
        input_data = task.get('input', {})
        examples = input_data.get('examples', [])
        test_item = input_data.get('test_item', None)
        association_type = input_data.get('association_type', None)
        
        # Collect first and second items from examples
        firsts = [ex.get('first', '') for ex in examples if isinstance(ex, dict)]
        seconds = [ex.get('second', '') for ex in examples if isinstance(ex, dict)]
        
        # Check if prediction matches any valid second
        if predicted_item in seconds:
            learning = 0.5  # At least predicted a valid second item
        
        # Check metadata for full pair info
        metadata = task.get('metadata', {})
        full_pair = metadata.get('full_pair', ())
        
        if full_pair and test_item == full_pair[0] and predicted_item != associated_item:
            # Calculate string similarity to correct answer
            if isinstance(predicted_item, str) and isinstance(associated_item, str):
                similarity = self._calculate_string_similarity(predicted_item, associated_item)
                learning = max(learning, 0.3 * similarity)
        
        # Novelty component - reward for new association types
        novelty = 0.0
        concept = task.get('metadata', {}).get('concept', '')
        
        if concept in self.concept_history:
            # Less novelty if we've seen this concept a lot
            exposure_count = self.concept_history.get(concept, {}).get('count', 0)
            novelty = max(0.0, 1.0 - (exposure_count / 10.0))
        else:
            # First time seeing this concept
            novelty = 1.0
        
        return {
            'accuracy': accuracy,
            'learning': learning,
            'novelty': novelty
        }
    
    # Fallback methods for standard Absolute Zero tasks
    def calculate_deduction_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """Calculate reward for deduction tasks from standard Absolute Zero"""
        # Basic verification
        ground_truth = task.get('output', {})
        accuracy = float(self.verify_output(prediction, ground_truth))
        
        # Simple learning and novelty signals
        learning = 0.3 if accuracy > 0 else 0.0
        novelty = 0.2
        
        return {
            'accuracy': accuracy,
            'learning': learning,
            'novelty': novelty
        }
    
    def calculate_abduction_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """Calculate reward for abduction tasks from standard Absolute Zero"""
        # Similar to deduction for now
        return self.calculate_deduction_reward(task, prediction)
    
    def calculate_induction_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """Calculate reward for induction tasks from standard Absolute Zero"""
        # Similar structure to pattern reward
        return self.calculate_pattern_reward(task, prediction)
    
    def calculate_pattern_recognition_reward(self, task: Dict[str, Any], prediction: Any) -> Dict[str, float]:
        """Calculate reward for pattern_recognition tasks from standard Absolute Zero"""
        # Similar structure to sequence reward
        return self.calculate_sequence_reward(task, prediction)
    
    def verify_output(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Verify if prediction matches ground truth.
        
        Args:
            prediction: The model's prediction
            ground_truth: The expected output
            
        Returns:
            Boolean indicating whether prediction is correct
        """
        # Handle different types of predictions and ground truths
        if isinstance(ground_truth, dict) and isinstance(prediction, dict):
            # For dictionary outputs (like pattern tasks)
            if 'next_element' in ground_truth and 'next_element' in prediction:
                return prediction['next_element'] == ground_truth['next_element']
            elif 'category' in ground_truth and 'category' in prediction:
                return prediction['category'] == ground_truth['category']
            elif 'associated_item' in ground_truth and 'associated_item' in prediction:
                return prediction['associated_item'] == ground_truth['associated_item']
            elif set(ground_truth.keys()) == set(prediction.keys()):
                return all(prediction[k] == ground_truth[k] for k in ground_truth)
        elif isinstance(ground_truth, (list, tuple)) and isinstance(prediction, (list, tuple)):
            # For list outputs (like abduce tasks returning facts)
            if len(ground_truth) == len(prediction):
                return all(str(a) == str(b) for a, b in zip(prediction, ground_truth))
        elif isinstance(ground_truth, (int, float)) and isinstance(prediction, (int, float)):
            # For numeric outputs
            return abs(prediction - ground_truth) < 1e-6
        elif isinstance(ground_truth, str) and isinstance(prediction, str):
            # For string outputs (like symbolic conclusions)
            return prediction == ground_truth
        elif isinstance(ground_truth, np.ndarray) and isinstance(prediction, np.ndarray):
            # For array outputs
            return np.allclose(prediction, ground_truth)
            
        # Fallback: string comparison
        return str(prediction) == str(ground_truth)
    
    def update_history(self, task: Dict[str, Any], rewards: Dict[str, float], task_id: str):
        """
        Update history with task results for tracking learning progress.
        
        Args:
            task: The task that was attempted
            rewards: The rewards that were given
            task_id: Unique identifier for the task
        """
        # Record task results
        task_type = task.get('type', 'unknown')
        
        # Initialize task type in history if not present
        if task_type not in self.task_history:
            self.task_history[task_type] = {
                'attempts': 0,
                'successes': 0,
                'recent_results': [],
                'last_reward': 0.0
            }
            
        # Update task history
        accuracy = rewards.get('accuracy', 0.0)
        combined = rewards.get('combined_reward', 0.0)
        
        self.task_history[task_type]['attempts'] += 1
        self.task_history[task_type]['successes'] += 1 if accuracy >= 0.9 else 0
        self.task_history[task_type]['recent_results'].append(accuracy)
        self.task_history[task_type]['last_reward'] = combined
        
        # Keep only the last 10 results
        if len(self.task_history[task_type]['recent_results']) > 10:
            self.task_history[task_type]['recent_results'].pop(0)
        
        # Update concept history if available
        metadata = task.get('metadata', {})
        concept = metadata.get('concept', '')
        
        if concept:
            if concept not in self.concept_history:
                self.concept_history[concept] = {
                    'count': 0,
                    'successes': 0,
                    'recent_results': []
                }
                
            self.concept_history[concept]['count'] += 1
            self.concept_history[concept]['successes'] += 1 if accuracy >= 0.9 else 0
            self.concept_history[concept]['recent_results'].append(accuracy)
            
            # Keep only the last 10 results
            if len(self.concept_history[concept]['recent_results']) > 10:
                self.concept_history[concept]['recent_results'].pop(0)
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        Get learning progress statistics.
        
        Returns:
            Dictionary with progress statistics
        """
        stats = {
            'task_types': {},
            'concepts': {},
            'overall': {
                'total_attempts': 0,
                'success_rate': 0.0,
                'learning_trend': 0.0
            }
        }
        
        # Calculate task type statistics
        total_attempts = 0
        total_successes = 0
        
        for task_type, history in self.task_history.items():
            attempts = history['attempts']
            successes = history['successes']
            recent = history['recent_results']
            
            # Success rate
            success_rate = successes / max(1, attempts)
            
            # Learning trend (positive if improving)
            trend = 0.0
            if len(recent) >= 5:
                early = sum(recent[:len(recent)//2]) / (len(recent)//2)
                late = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
                trend = late - early
            
            stats['task_types'][task_type] = {
                'attempts': attempts,
                'success_rate': success_rate,
                'trend': trend,
                'recent': recent
            }
            
            total_attempts += attempts
            total_successes += successes
        
        # Calculate concept statistics
        for concept, history in self.concept_history.items():
            count = history['count']
            successes = history['successes']
            recent = history['recent_results']
            
            # Success rate
            success_rate = successes / max(1, count)
            
            # Learning trend
            trend = 0.0
            if len(recent) >= 5:
                early = sum(recent[:len(recent)//2]) / (len(recent)//2)
                late = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
                trend = late - early
            
            stats['concepts'][concept] = {
                'count': count,
                'success_rate': success_rate,
                'trend': trend
            }
        
        # Overall statistics
        stats['overall']['total_attempts'] = total_attempts
        stats['overall']['success_rate'] = total_successes / max(1, total_attempts)
        
        # Calculate overall learning trend
        all_recent = []
        for history in self.task_history.values():
            all_recent.extend(history['recent_results'])
        
        if len(all_recent) >= 10:
            early = sum(all_recent[:len(all_recent)//2]) / (len(all_recent)//2)
            late = sum(all_recent[len(all_recent)//2:]) / (len(all_recent) - len(all_recent)//2)
            stats['overall']['learning_trend'] = late - early
        
        return stats
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for strings
        if not str1 or not str2:
            return 0.0
            
        # Convert to character sets
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        # Calculate Jaccard similarity (intersection / union)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0 