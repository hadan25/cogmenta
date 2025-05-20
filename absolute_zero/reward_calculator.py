import numpy as np
from typing import Dict, Any, List, Union

class RewardCalculator:
    """Calculates rewards for different aspects of learning."""
    
    def calculate_learnability_reward(self, task, snn) -> float:
        """
        Calculate how learnable the task is for the SNN.
        
        Args:
            task: Task dictionary containing input and output
            snn: SNN with process_input method
            
        Returns:
            Learnability reward (0.0 to 1.0)
        """
        # Check for valid inputs
        if not task or not isinstance(task, dict):
            print("Warning: Task is None or not a dictionary")
            return 0.0
            
        if snn is None:
            print("Error: SNN is None in calculate_learnability_reward")
            return 0.0
            
        if not hasattr(snn, 'process_input'):
            print("Error: SNN has no process_input method")
            return 0.0
        
        try:
            # Extract input and encode
            task_input = task.get('input', {})
            if not task_input:
                print("Warning: Task input is empty")
                return 0.0
                
            # Convert to vector
            input_encoded = self._encode_input(task_input)
            
            # Get prediction from SNN
            prediction = snn.process_input(input_encoded)
            
            # Check if the prediction has the expected format
            if not isinstance(prediction, dict) or 'next_element' not in prediction:
                print(f"Warning: Unexpected prediction format: {prediction}")
                return 0.5  # Default value
            
            # Apply sigmoid to magnitude as a simple learnability score
            # (larger magnitudes might indicate more confident predictions)
            if isinstance(prediction['next_element'], (int, float, np.number)):
                confidence = 1.0 / (1.0 + np.exp(-abs(prediction['next_element']) / 10.0))
                return min(0.9, max(0.1, confidence))  # Clip to reasonable range
            else:
                return 0.5  # Default value
                
        except Exception as e:
            print(f"Error in calculate_learnability_reward: {e}")
            return 0.5  # Default value
    
    def verify_output(self, prediction, expected_output) -> float:
        """
        Verify if prediction matches expected output.
        
        Args:
            prediction: Model's prediction
            expected_output: Expected output for the task
            
        Returns:
            Accuracy (0.0 or 1.0)
        """
        try:
            # Handle different output formats
            if not isinstance(prediction, dict) or not isinstance(expected_output, dict):
                print(f"Warning: Prediction or expected_output is not a dictionary")
                return 0.0
                
            # For pattern tasks, check the next_element
            if 'next_element' in prediction and 'next_element' in expected_output:
                return float(prediction['next_element'] == expected_output['next_element'])
                
            # For classification tasks, check the class
            if 'class' in prediction and 'class' in expected_output:
                return float(prediction['class'] == expected_output['class'])
                
            # Default case - keys don't match
            return 0.0
            
        except Exception as e:
            print(f"Error in verify_output: {e}")
            return 0.0
    
    def _encode_input(self, task_input: Any) -> np.ndarray:
        """Encode task input for SNN processing."""
        try:
            if isinstance(task_input, dict):
                # Extract values that can be converted to float
                values = []
                for value in task_input.values():
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                    elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                        values.extend(value)
                return np.array(values) if values else np.array([0.0])
                
            elif isinstance(task_input, list):
                # Flatten if nested
                if any(isinstance(x, list) for x in task_input):
                    flat_list = []
                    for item in task_input:
                        if isinstance(item, list):
                            flat_list.extend(item)
                        else:
                            flat_list.append(item)
                    return np.array([float(x) for x in flat_list if isinstance(x, (int, float))])
                else:
                    return np.array([float(x) for x in task_input if isinstance(x, (int, float))])
                    
            elif isinstance(task_input, (int, float)):
                return np.array([float(task_input)])
                
            elif isinstance(task_input, str):
                # Simple string encoding - use character codes
                return np.array([ord(c)/128.0 for c in task_input])
                
            else:
                print(f"Warning: Unsupported input type: {type(task_input)}")
                return np.array([0.0])
                
        except Exception as e:
            print(f"Error encoding input: {e}")
            return np.array([0.0])