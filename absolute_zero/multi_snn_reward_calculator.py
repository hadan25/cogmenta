import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class MultiSNNRewardCalculator:
    """
    Enhanced reward calculator that leverages multiple specialized SNNs
    to calculate more nuanced and comprehensive reward signals.
    """
    
    def __init__(self, snns=None):
        """
        Initialize the multi-SNN reward calculator.
        
        Args:
            snns: Dictionary of SNN adapters
        """
        self.snns = snns or {}
        
        # Track performance history
        self.performance_history = defaultdict(list)
        self.task_rewards = defaultdict(lambda: defaultdict(list))
        
        # Task type weights for reward calculation
        self.task_weights = self._initialize_task_weights()
    
    def _initialize_task_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize weights for different SNNs based on task types.
        
        Returns:
            Dictionary mapping task types to SNN weight dictionaries
        """
        # Default weights for different SNNs across task types
        weights = {
            'pattern': {
                'statistical': 0.7,
                'perceptual': 0.1,
                'memory': 0.1,
                'reasoning': 0.1
            },
            'classification': {
                'statistical': 0.4,
                'perceptual': 0.5,
                'memory': 0.0,
                'reasoning': 0.1
            },
            'reasoning': {
                'statistical': 0.2,
                'perceptual': 0.0,
                'memory': 0.1,
                'reasoning': 0.7
            },
            'decision': {
                'statistical': 0.2,
                'decision': 0.6,
                'affective': 0.1,
                'reasoning': 0.1
            },
            'memory': {
                'statistical': 0.2,
                'perceptual': 0.1,
                'memory': 0.6,
                'reasoning': 0.1
            },
            'perceptual': {
                'statistical': 0.2,
                'perceptual': 0.7,
                'memory': 0.0,
                'reasoning': 0.1
            },
            'metacognitive': {
                'statistical': 0.1,
                'metacognitive': 0.6,
                'affective': 0.1,
                'reasoning': 0.2
            }
        }
        
        return weights
    
    def calculate_comprehensive_reward(self, task: Dict, results: Dict) -> Dict:
        """
        Calculate comprehensive reward using multiple SNNs.
        
        Args:
            task: Task dictionary
            results: Dictionary of results from different SNNs
            
        Returns:
            Dictionary with comprehensive reward information
        """
        task_type = task.get('type', 'unknown')
        task_id = task.get('id', 'unknown')
        
        # Calculate rewards from each specialized SNN
        rewards = {}
        
        # Statistical SNN reward (basic pattern learning)
        if 'statistical' in results and 'statistical' in self.snns:
            rewards['statistical'] = self.calculate_statistical_reward(
                task, results['statistical']
            )
        
        # Perceptual SNN reward (classification and feature extraction)
        if 'perceptual' in results and 'perceptual' in self.snns:
            rewards['perceptual'] = self.calculate_perceptual_reward(
                task, results['perceptual']
            )
        
        # Memory SNN reward (recall and recognition)
        if 'memory' in results and 'memory' in self.snns:
            rewards['memory'] = self.calculate_memory_reward(
                task, results['memory']
            )
        
        # Reasoning SNN reward (logical reasoning)
        if 'reasoning' in results and 'reasoning' in self.snns:
            rewards['reasoning'] = self.calculate_reasoning_reward(
                task, results['reasoning']
            )
        
        # Decision SNN reward (action selection)
        if 'decision' in results and 'decision' in self.snns:
            rewards['decision'] = self.calculate_decision_reward(
                task, results['decision']
            )
        
        # Metacognitive SNN reward (confidence estimation)
        if 'metacognitive' in results and 'metacognitive' in self.snns:
            rewards['metacognitive'] = self.calculate_metacognitive_reward(
                task, results['metacognitive']
            )
        
        # Use appropriate task weights
        if task_type in self.task_weights:
            task_weight_dict = self.task_weights[task_type]
        else:
            task_weight_dict = {'statistical': 1.0}
        
        # Calculate weighted reward
        weighted_rewards = {}
        for snn_type, reward in rewards.items():
            if snn_type in task_weight_dict:
                weight = task_weight_dict[snn_type]
                weighted_rewards[snn_type] = reward * weight
        
        # Calculate total reward
        total_weight = sum(task_weight_dict.get(snn_type, 0.0) for snn_type in rewards.keys())
        if total_weight > 0:
            combined_reward = sum(weighted_rewards.values()) / total_weight
        else:
            combined_reward = sum(rewards.values()) / len(rewards) if rewards else 0.0
        
        # Track performance
        self.performance_history[task_type].append(combined_reward)
        for snn_type, reward in rewards.items():
            self.task_rewards[task_type][snn_type].append(reward)
        
        # Create response with detailed breakdown
        result = {
            'combined_reward': combined_reward,
            'individual_rewards': rewards,
            'weighted_rewards': weighted_rewards,
            'task_type': task_type,
            'task_id': task_id
        }
        
        return result
    
    def calculate_statistical_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for statistical pattern learning.
        
        Args:
            task: Task dictionary
            result: Result from StatisticalSNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        task_type = task.get('type', 'unknown')
        expected_output = task.get('output', {})
        
        # Default reward calculation
        if hasattr(self, f'_reward_for_{task_type}_statistical'):
            method = getattr(self, f'_reward_for_{task_type}_statistical')
            return method(task, result, expected_output)
        
        # Generic reward calculation
        return self._verify_output_match(result, expected_output)
    
    def calculate_perceptual_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for perceptual tasks.
        
        Args:
            task: Task dictionary
            result: Result from PerceptualSNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        # Perceptual reward is primarily based on classification accuracy
        expected_category = task.get('output', {}).get('category')
        
        if expected_category is None:
            return 0.5  # Neutral reward if no expected category
        
        # Extract predicted category from result
        predicted_category = None
        if isinstance(result, dict) and 'category' in result:
            predicted_category = result['category']
        elif isinstance(result, str):
            predicted_category = result
        
        # Exact match
        if predicted_category == expected_category:
            return 1.0
            
        # Partial matches
        if predicted_category and expected_category:
            # Check string similarity for partial credit
            similarity = self._calculate_string_similarity(
                str(predicted_category), str(expected_category)
            )
            return max(0.0, similarity)
        
        return 0.0
    
    def calculate_memory_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for memory tasks.
        
        Args:
            task: Task dictionary
            result: Result from MemorySNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        task_subtype = task.get('subtype', 'recall')
        
        if task_subtype == 'recall':
            # Recall tasks check if item was correctly recalled
            expected_item = task.get('output', {}).get('recalled_item')
            
            if isinstance(result, dict) and 'recalled_item' in result:
                predicted_item = result['recalled_item']
            else:
                predicted_item = result
            
            return 1.0 if predicted_item == expected_item else 0.0
            
        elif task_subtype == 'recognition':
            # Recognition tasks check proportion of correct recognitions
            expected_results = task.get('output', {}).get('recognition_results', {})
            
            if isinstance(result, dict) and 'recognition_results' in result:
                predicted_results = result['recognition_results']
                
                # Calculate proportion of correct recognitions
                correct = 0
                total = len(expected_results)
                
                for item, expected in expected_results.items():
                    if item in predicted_results and predicted_results[item] == expected:
                        correct += 1
                
                return correct / total if total > 0 else 0.0
                
            return 0.0
            
        elif task_subtype == 'association':
            # Association tasks check if the associated item was correctly recalled
            expected_item = task.get('output', {}).get('associated_item')
            
            if isinstance(result, dict) and 'associated_item' in result:
                predicted_item = result['associated_item']
            else:
                predicted_item = result
            
            return 1.0 if predicted_item == expected_item else 0.0
        
        return 0.0
    
    def calculate_reasoning_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for reasoning tasks.
        
        Args:
            task: Task dictionary
            result: Result from ReasoningSNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        task_subtype = task.get('subtype', 'deductive')
        
        if task_subtype == 'deductive':
            # Deductive tasks check if conclusion matches
            expected_conclusion = task.get('output', {}).get('conclusion')
            
            if isinstance(result, dict) and 'conclusion' in result:
                predicted_conclusion = result['conclusion']
            else:
                predicted_conclusion = result
            
            # Check for exact match
            if predicted_conclusion == expected_conclusion:
                return 1.0
                
            # Check for similarity if strings
            if isinstance(predicted_conclusion, str) and isinstance(expected_conclusion, str):
                similarity = self._calculate_string_similarity(
                    predicted_conclusion, expected_conclusion
                )
                return max(0.0, similarity)
            
            return 0.0
            
        elif task_subtype == 'inductive':
            # Inductive tasks check if pattern was correctly continued
            expected_next = task.get('output', {}).get('next_element')
            
            if isinstance(result, dict) and 'next_element' in result:
                predicted_next = result['next_element']
            else:
                predicted_next = result
            
            # For numeric predictions, allow for close matches
            if isinstance(expected_next, (int, float)) and isinstance(predicted_next, (int, float)):
                difference = abs(expected_next - predicted_next)
                max_difference = max(1.0, abs(expected_next) * 0.2)  # Allow 20% error
                
                if difference == 0:
                    return 1.0
                elif difference <= max_difference:
                    return 1.0 - (difference / max_difference)
                else:
                    return 0.0
            
            # For non-numeric, exact match only
            return 1.0 if predicted_next == expected_next else 0.0
            
        elif task_subtype == 'abductive':
            # Abductive tasks check if cause was correctly identified
            expected_cause = task.get('output', {}).get('most_likely_cause')
            
            if isinstance(result, dict) and 'most_likely_cause' in result:
                predicted_cause = result['most_likely_cause']
            else:
                predicted_cause = result
            
            return 1.0 if predicted_cause == expected_cause else 0.0
        
        return 0.0
    
    def calculate_decision_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for decision tasks.
        
        Args:
            task: Task dictionary
            result: Result from DecisionSNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        # Extract expected optimal action
        expected_action = task.get('output', {}).get('optimal_action')
        
        # Extract predicted action
        predicted_action = None
        if isinstance(result, dict):
            if 'action' in result:
                predicted_action = result['action']
            elif 'selected_action' in result:
                predicted_action = result['selected_action']
            elif 'decision' in result:
                predicted_action = result['decision']
        else:
            predicted_action = result
        
        # If there are explicit rewards in the task, use those
        if 'input' in task and 'context' in task['input'] and 'rewards' in task['input']['context']:
            rewards = task['input']['context']['rewards']
            
            if predicted_action in rewards:
                return rewards[predicted_action]
            return 0.0
        
        # Otherwise, simple match/non-match
        return 1.0 if predicted_action == expected_action else 0.0
    
    def calculate_metacognitive_reward(self, task: Dict, result: Any) -> float:
        """
        Calculate reward specifically for metacognitive tasks.
        
        Args:
            task: Task dictionary
            result: Result from MetacognitiveSNN
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        # Metacognitive tasks evaluate confidence estimation
        expected_confidence = task.get('output', {}).get('expected_confidence', 0.5)
        
        # Extract predicted confidence
        predicted_confidence = None
        if isinstance(result, dict):
            if 'confidence' in result:
                predicted_confidence = result['confidence']
            elif 'expected_confidence' in result:
                predicted_confidence = result['expected_confidence']
        elif isinstance(result, (int, float)):
            predicted_confidence = result
        
        if predicted_confidence is None:
            return 0.5  # Neutral reward if no confidence prediction
            
        # Calculate reward based on how close the confidence estimation is
        difference = abs(expected_confidence - predicted_confidence)
        
        # Reward diminishes as difference increases
        return max(0.0, 1.0 - difference)
    
    def calculate_learnability_reward(self, task: Dict, snn: Any) -> float:
        """
        Calculate learnability reward for a task.
        
        Args:
            task: Task dictionary
            snn: SNN to evaluate learnability for
            
        Returns:
            Learnability reward (0.0 to 1.0)
        """
        # If SNN has a learnability method, use it
        if hasattr(snn, 'calculate_learnability'):
            return snn.calculate_learnability(task)
        
        # Default implementation
        # Estimate learnability based on task complexity and SNN capabilities
        task_type = task.get('type', 'unknown')
        task_complexity = task.get('complexity', 1.0)
        
        # Base learnability decreases as complexity increases
        base_learnability = max(0.0, 1.0 - (task_complexity / 10.0))
        
        # Adjust based on task type and SNN type
        snn_type = getattr(snn, 'snn_type', 'unknown')
        
        # Get appropriate weights for this task type
        if task_type in self.task_weights and snn_type in self.task_weights[task_type]:
            # Higher weight means the SNN is more suited for this task
            snn_suitability = self.task_weights[task_type][snn_type]
        else:
            # Default to moderate suitability
            snn_suitability = 0.5
        
        # Combine factors - both complexity and suitability affect learnability
        learnability = base_learnability * snn_suitability
        
        return learnability
    
    def verify_output(self, prediction: Any, expected_output: Any) -> float:
        """
        Verify if prediction matches expected output.
        
        Args:
            prediction: Prediction from SNN
            expected_output: Expected output from task
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        return self._verify_output_match(prediction, expected_output)
    
    def _verify_output_match(self, prediction: Any, expected_output: Any) -> float:
        """
        Helper method to verify output match with various data types.
        
        Args:
            prediction: Prediction from SNN
            expected_output: Expected output from task
            
        Returns:
            Match score (0.0 to 1.0)
        """
        # Handle different data types
        if isinstance(expected_output, dict) and isinstance(prediction, dict):
            # For dictionaries, check key-value matches
            matches = 0
            total = len(expected_output)
            
            for key, expected_value in expected_output.items():
                if key in prediction:
                    predicted_value = prediction[key]
                    
                    if expected_value == predicted_value:
                        matches += 1
                    elif isinstance(expected_value, (int, float)) and isinstance(predicted_value, (int, float)):
                        # For numeric values, allow close matches
                        difference = abs(expected_value - predicted_value)
                        max_difference = max(1.0, abs(expected_value) * 0.2)  # Allow 20% error
                        
                        if difference <= max_difference:
                            matches += 1 - (difference / max_difference)
            
            return matches / total if total > 0 else 0.0
            
        elif isinstance(expected_output, (list, tuple)) and isinstance(prediction, (list, tuple)):
            # For lists, check element-wise matches
            matches = 0
            total = len(expected_output)
            
            for i, expected_value in enumerate(expected_output):
                if i < len(prediction):
                    predicted_value = prediction[i]
                    
                    if expected_value == predicted_value:
                        matches += 1
                    elif isinstance(expected_value, (int, float)) and isinstance(predicted_value, (int, float)):
                        # For numeric values, allow close matches
                        difference = abs(expected_value - predicted_value)
                        max_difference = max(1.0, abs(expected_value) * 0.2)  # Allow 20% error
                        
                        if difference <= max_difference:
                            matches += 1 - (difference / max_difference)
            
            return matches / total if total > 0 else 0.0
            
        elif isinstance(expected_output, (int, float)) and isinstance(prediction, (int, float)):
            # For numeric values, allow close matches
            difference = abs(expected_output - prediction)
            max_difference = max(1.0, abs(expected_output) * 0.2)  # Allow 20% error
            
            if difference == 0:
                return 1.0
            elif difference <= max_difference:
                return 1.0 - (difference / max_difference)
            else:
                return 0.0
                
        elif isinstance(expected_output, str) and isinstance(prediction, str):
            # For strings, check similarity
            similarity = self._calculate_string_similarity(expected_output, prediction)
            return similarity
        
        # Direct comparison for other types
        return 1.0 if prediction == expected_output else 0.0
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple longest common subsequence similarity
        if not str1 or not str2:
            return 0.0
            
        # Use longest common subsequence
        lcs_length = self._longest_common_subsequence(str1, str2)
        
        # Normalize by average length
        avg_length = (len(str1) + len(str2)) / 2
        similarity = lcs_length / avg_length if avg_length > 0 else 0.0
        
        return similarity
    
    def _longest_common_subsequence(self, str1: str, str2: str) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def get_performance_stats(self) -> Dict:
        """
        Get statistics on performance across task types and SNNs.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'task_types': {},
            'snn_types': {},
            'overall': {}
        }
        
        # Calculate overall statistics
        all_rewards = []
        for rewards in self.performance_history.values():
            all_rewards.extend(rewards)
        
        if all_rewards:
            stats['overall']['average_reward'] = sum(all_rewards) / len(all_rewards)
            stats['overall']['reward_trend'] = self._calculate_trend(all_rewards)
        else:
            stats['overall']['average_reward'] = 0.0
            stats['overall']['reward_trend'] = 0.0
        
        # Calculate task type statistics
        for task_type, rewards in self.performance_history.items():
            if rewards:
                stats['task_types'][task_type] = {
                    'average_reward': sum(rewards) / len(rewards),
                    'reward_trend': self._calculate_trend(rewards)
                }
        
        # Calculate SNN type statistics
        for task_type, snn_rewards in self.task_rewards.items():
            for snn_type, rewards in snn_rewards.items():
                if snn_type not in stats['snn_types']:
                    stats['snn_types'][snn_type] = {'rewards': []}
                
                stats['snn_types'][snn_type]['rewards'].extend(rewards)
        
        # Calculate averages and trends for SNN types
        for snn_type, data in stats['snn_types'].items():
            rewards = data['rewards']
            if rewards:
                data['average_reward'] = sum(rewards) / len(rewards)
                data['reward_trend'] = self._calculate_trend(rewards)
            else:
                data['average_reward'] = 0.0
                data['reward_trend'] = 0.0
            
            # Remove raw data to avoid huge objects
            del data['rewards']
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Trend value (positive for upward trend, negative for downward)
        """
        if len(values) < 10:
            return 0.0
            
        # Use last 10 values
        recent_values = values[-10:]
        
        # Simple linear regression
        n = len(recent_values)
        x_mean = (n - 1) / 2  # Average index
        y_mean = sum(recent_values) / n  # Average value
        
        # Calculate slope
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator 