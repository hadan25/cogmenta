"""
Reward Propagation Bridge Across Paradigms

This module implements a bidirectional bridge between symbolic and neural components
of the SNN framework, allowing rewards and error signals to propagate across these
different paradigms. It translates symbolic reasoning outcomes into neural learning
signals and vice versa.

Key features:
- Protocol for translating symbolic errors/rewards to neural learning signals
- Mapping success/failure of symbolic components to numerical reward signals
- Minimal "reward adapter" for symbolic-neural interface
- Integration with the Unified Reward System
"""

import torch
import numpy as np
import logging
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import local modules with error handling for both package and direct imports
try:
    # When imported as package
    from models.snn.unified_reward_system import (
        UnifiedRewardSystem,
        RewardComponent,
        create_reward_system
    )
    from models.symbolic.vector_symbolic import VectorSymbolicEngine, VectorSymbolicAdapter
    from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge
except ImportError:
    # When run directly
    try:
        from unified_reward_system import (
            UnifiedRewardSystem,
            RewardComponent,
            create_reward_system
        )
        from models.symbolic.vector_symbolic import VectorSymbolicEngine, VectorSymbolicAdapter
        from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge
    except ImportError:
        # Stubs for testing
        class UnifiedRewardSystem:
            pass
        class RewardComponent:
            pass
        def create_reward_system():
            return None
        class VectorSymbolicEngine:
            pass
        class VectorSymbolicAdapter:
            pass
        class EnhancedNeuroSymbolicBridge:
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reward_propagation_bridge")


class SymbolicRewardType(Enum):
    """Types of symbolic rewards/errors that can be translated to neural signals."""
    LOGICAL_CONSISTENCY = "logical_consistency"   # Consistency in logical inference
    THEOREM_PROVING = "theorem_proving"           # Success in proving theorems
    CONSTRAINT_SATISFACTION = "constraint"        # Satisfaction of constraints
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"   # Success in retrieving knowledge
    RULE_APPLICATION = "rule_application"         # Correct application of rules
    SEMANTIC_COHERENCE = "semantic_coherence"     # Coherence of symbolic representations
    CAUSAL_REASONING = "causal_reasoning"         # Success in causal reasoning
    CONTRADICTION_DETECTION = "contradiction"     # Detection of contradictions


class NeuralRewardType(Enum):
    """Types of neural rewards that can be translated to symbolic signals."""
    INFORMATION_RETENTION = "information_retention" # Information preservation in neural processing
    PREDICTION_ACCURACY = "prediction_accuracy"     # Accuracy of neural predictions
    RECONSTRUCTION_ERROR = "reconstruction_error"   # Error in reconstructing inputs
    NOVELTY_DETECTION = "novelty_detection"         # Detection of novel patterns
    UNCERTAINTY_REDUCTION = "uncertainty_reduction" # Reduction in uncertainty
    SURPRISE_MINIMIZATION = "surprise_minimization" # Minimization of surprise
    FEATURE_LEARNING = "feature_learning"           # Success in learning useful features


class RewardPropagationBridge:
    """
    Bridge for propagating reward signals between symbolic and neural paradigms.
    
    This bridge translates between symbolic reasoning success/failure and 
    numerical reward signals that can be used for neural learning. It also
    translates neural performance metrics into feedback for symbolic components.
    """
    
    def __init__(
        self,
        unified_reward_system: Optional[UnifiedRewardSystem] = None,
        vector_symbolic_adapter: Optional[VectorSymbolicAdapter] = None,
        neural_symbolic_bridge: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the reward propagation bridge.
        
        Args:
            unified_reward_system: The unified reward system
            vector_symbolic_adapter: Adapter for vector symbolic operations
            neural_symbolic_bridge: Bridge between neural and symbolic components
            config: Configuration dictionary
        """
        # Store components
        self.unified_reward_system = unified_reward_system or create_reward_system()
        self.vector_symbolic_adapter = vector_symbolic_adapter
        self.neural_symbolic_bridge = neural_symbolic_bridge
        
        # Apply configuration
        self.config = config or {}
        self.symbolic_weight = self.config.get("symbolic_weight", 0.7)
        self.neural_weight = self.config.get("neural_weight", 0.7)
        self.default_reward = self.config.get("default_reward", 0.0)
        self.feedback_threshold = self.config.get("feedback_threshold", 0.6)
        
        # Initialize reward mappings
        self._init_symbolic_reward_mappings()
        self._init_neural_reward_mappings()
        
        # Initialize statistics
        self.symbolic_success_count = 0
        self.symbolic_failure_count = 0
        self.neural_reward_history = []
        self.symbolic_reward_history = []
        
        # Tracking for recurrent processing
        self.cross_paradigm_iterations = 0
        self.max_iterations = self.config.get("max_iterations", 3)
        
        logger.info(f"Initialized RewardPropagationBridge with symbolic_weight={self.symbolic_weight}, neural_weight={self.neural_weight}")
    
    def _init_symbolic_reward_mappings(self):
        """Initialize mappings from symbolic outcomes to numerical rewards."""
        self.symbolic_reward_mappings = {
            SymbolicRewardType.LOGICAL_CONSISTENCY: {
                "success_reward": 1.0,
                "failure_penalty": -0.7,
                "neutral": 0.0,
                "weight": 0.9
            },
            SymbolicRewardType.THEOREM_PROVING: {
                "success_reward": 1.0,
                "failure_penalty": -0.5,
                "neutral": 0.0,
                "weight": 0.8
            },
            SymbolicRewardType.CONSTRAINT_SATISFACTION: {
                "success_reward": 0.9,
                "failure_penalty": -0.8,
                "neutral": 0.0,
                "weight": 0.7
            },
            SymbolicRewardType.KNOWLEDGE_RETRIEVAL: {
                "success_reward": 0.8,
                "failure_penalty": -0.3,
                "neutral": 0.0,
                "weight": 0.6
            },
            SymbolicRewardType.RULE_APPLICATION: {
                "success_reward": 0.8,
                "failure_penalty": -0.5,
                "neutral": 0.0,
                "weight": 0.7
            },
            SymbolicRewardType.SEMANTIC_COHERENCE: {
                "success_reward": 0.7,
                "failure_penalty": -0.4,
                "neutral": 0.0,
                "weight": 0.6
            },
            SymbolicRewardType.CAUSAL_REASONING: {
                "success_reward": 0.9,
                "failure_penalty": -0.6,
                "neutral": 0.0,
                "weight": 0.8
            },
            SymbolicRewardType.CONTRADICTION_DETECTION: {
                "success_reward": 0.9,
                "failure_penalty": -0.2,
                "neutral": 0.0,
                "weight": 0.5
            }
        }
    
    def _init_neural_reward_mappings(self):
        """Initialize mappings from neural outcomes to symbolic feedback."""
        self.neural_reward_mappings = {
            NeuralRewardType.INFORMATION_RETENTION: {
                "high_threshold": 0.9,
                "low_threshold": 0.3,
                "weight": 0.8
            },
            NeuralRewardType.PREDICTION_ACCURACY: {
                "high_threshold": 0.8,
                "low_threshold": 0.4,
                "weight": 0.9
            },
            NeuralRewardType.RECONSTRUCTION_ERROR: {
                "high_threshold": 0.7,  # For error, "high" is bad
                "low_threshold": 0.2,
                "weight": 0.7,
                "inverse": True  # Lower is better for error metrics
            },
            NeuralRewardType.NOVELTY_DETECTION: {
                "high_threshold": 0.8,
                "low_threshold": 0.4,
                "weight": 0.6
            },
            NeuralRewardType.UNCERTAINTY_REDUCTION: {
                "high_threshold": 0.8,
                "low_threshold": 0.3,
                "weight": 0.7
            },
            NeuralRewardType.SURPRISE_MINIMIZATION: {
                "high_threshold": 0.7,
                "low_threshold": 0.3,
                "weight": 0.6,
                "inverse": True  # Lower surprise is better
            },
            NeuralRewardType.FEATURE_LEARNING: {
                "high_threshold": 0.8,
                "low_threshold": 0.4,
                "weight": 0.7
            }
        }
    
    def symbolic_to_neural_reward(
        self,
        symbolic_outcome: Dict[str, Any],
        reward_type: Union[str, SymbolicRewardType] = SymbolicRewardType.KNOWLEDGE_RETRIEVAL
    ) -> float:
        """
        Convert symbolic reasoning outcome to a numerical reward for neural learning.
        
        Args:
            symbolic_outcome: Dictionary with symbolic reasoning outcome
            reward_type: Type of symbolic reward to convert
            
        Returns:
            Numerical reward value (-1 to 1)
        """
        # Convert string to enum if needed
        if isinstance(reward_type, str):
            try:
                reward_type = SymbolicRewardType(reward_type)
            except ValueError:
                logger.warning(f"Unknown reward type: {reward_type}, using KNOWLEDGE_RETRIEVAL")
                reward_type = SymbolicRewardType.KNOWLEDGE_RETRIEVAL
        
        # Get mapping for this reward type
        reward_mapping = self.symbolic_reward_mappings.get(
            reward_type, 
            self.symbolic_reward_mappings[SymbolicRewardType.KNOWLEDGE_RETRIEVAL]
        )
        
        # Determine if the outcome was successful
        success = symbolic_outcome.get("success", False)
        
        # Get confidence if available
        confidence = symbolic_outcome.get("confidence", 1.0)
        
        # Calculate reward
        if success:
            reward = reward_mapping["success_reward"] * confidence
            self.symbolic_success_count += 1
        else:
            reward = reward_mapping["failure_penalty"] * confidence
            self.symbolic_failure_count += 1
        
        # Apply weight from mapping
        reward *= reward_mapping["weight"]
        
        # Apply global symbolic weight
        reward *= self.symbolic_weight
        
        # Track history
        self.symbolic_reward_history.append({
            "type": reward_type.value,
            "success": success,
            "confidence": confidence,
            "reward": reward
        })
        
        logger.debug(f"Symbolic to neural reward: {reward_type.value} -> {reward:.4f}")
        return reward
    
    def neural_to_symbolic_feedback(
        self,
        neural_metrics: Dict[str, float],
        reward_type: Union[str, NeuralRewardType] = NeuralRewardType.INFORMATION_RETENTION
    ) -> Dict[str, Any]:
        """
        Convert neural performance metrics to feedback for symbolic components.
        
        Args:
            neural_metrics: Dictionary with neural performance metrics
            reward_type: Type of neural reward to convert
            
        Returns:
            Dictionary with feedback for symbolic components
        """
        # Convert string to enum if needed
        if isinstance(reward_type, str):
            try:
                reward_type = NeuralRewardType(reward_type)
            except ValueError:
                logger.warning(f"Unknown reward type: {reward_type}, using INFORMATION_RETENTION")
                reward_type = NeuralRewardType.INFORMATION_RETENTION
        
        # Get mapping for this reward type
        reward_mapping = self.neural_reward_mappings.get(
            reward_type, 
            self.neural_reward_mappings[NeuralRewardType.INFORMATION_RETENTION]
        )
        
        # Get metric value
        metric_name = reward_type.value
        metric_value = neural_metrics.get(metric_name, 0.5)
        
        # Check if this metric is inverse (lower is better)
        is_inverse = reward_mapping.get("inverse", False)
        
        # Apply inverse if needed
        if is_inverse:
            metric_value = 1.0 - metric_value
        
        # Determine performance level
        high_threshold = reward_mapping["high_threshold"]
        low_threshold = reward_mapping["low_threshold"]
        
        if metric_value >= high_threshold:
            performance = "high"
            confidence_adjustment = 0.1  # Increase confidence in symbolic reasoning
            rule_weight_adjustment = 0.05  # Increase weight of related rules
        elif metric_value <= low_threshold:
            performance = "low"
            confidence_adjustment = -0.1  # Decrease confidence in symbolic reasoning
            rule_weight_adjustment = -0.05  # Decrease weight of related rules
        else:
            performance = "medium"
            confidence_adjustment = 0.0
            rule_weight_adjustment = 0.0
        
        # Apply weight from mapping
        confidence_adjustment *= reward_mapping["weight"]
        rule_weight_adjustment *= reward_mapping["weight"]
        
        # Apply global neural weight
        confidence_adjustment *= self.neural_weight
        rule_weight_adjustment *= self.neural_weight
        
        # Track history
        self.neural_reward_history.append({
            "type": reward_type.value,
            "value": metric_value,
            "performance": performance,
            "confidence_adjustment": confidence_adjustment,
            "rule_weight_adjustment": rule_weight_adjustment
        })
        
        # Create feedback dictionary
        feedback = {
            "performance": performance,
            "confidence_adjustment": confidence_adjustment,
            "rule_weight_adjustment": rule_weight_adjustment,
            "metric_value": metric_value,
            "reward_type": reward_type.value
        }
        
        logger.debug(f"Neural to symbolic feedback: {reward_type.value} -> {performance}")
        return feedback
    
    def process_cross_paradigm_feedback(
        self,
        symbolic_outcomes: Dict[str, Any],
        neural_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Process bidirectional feedback across symbolic and neural paradigms.
        
        This implements iterative feedback between symbolic and neural components,
        allowing them to refine each other's outputs through multiple passes.
        
        Args:
            symbolic_outcomes: Dictionary with symbolic reasoning outcomes
            neural_metrics: Dictionary with neural performance metrics
            
        Returns:
            Dictionary with integrated feedback
        """
        # Reset iteration counter
        self.cross_paradigm_iterations = 0
        
        # Initial feedback
        integrated_feedback = {
            "symbolic_rewards": {},
            "neural_feedback": {},
            "improvement_suggestions": [],
            "confidence": 0.5
        }
        
        # Process each symbolic outcome type
        for outcome_type in symbolic_outcomes:
            if outcome_type in [rt.value for rt in SymbolicRewardType]:
                # Convert symbolic outcome to neural reward
                reward = self.symbolic_to_neural_reward(
                    symbolic_outcomes[outcome_type],
                    reward_type=outcome_type
                )
                
                integrated_feedback["symbolic_rewards"][outcome_type] = reward
        
        # Process each neural metric type
        for metric_type in neural_metrics:
            if metric_type in [rt.value for rt in NeuralRewardType]:
                # Convert neural metric to symbolic feedback
                feedback = self.neural_to_symbolic_feedback(
                    {metric_type: neural_metrics[metric_type]},
                    reward_type=metric_type
                )
                
                integrated_feedback["neural_feedback"][metric_type] = feedback
        
        # Iterative refinement
        while self.cross_paradigm_iterations < self.max_iterations:
            # Use feedback to improve outcomes
            improved_symbolic = self._apply_neural_feedback_to_symbolic(
                symbolic_outcomes, 
                integrated_feedback["neural_feedback"]
            )
            
            improved_neural = self._apply_symbolic_rewards_to_neural(
                neural_metrics,
                integrated_feedback["symbolic_rewards"]
            )
            
            # Check if we've converged
            symbolic_change = self._calculate_change(symbolic_outcomes, improved_symbolic)
            neural_change = self._calculate_change(neural_metrics, improved_neural)
            
            if symbolic_change < 0.05 and neural_change < 0.05:
                # Converged
                logger.debug(f"Cross-paradigm feedback converged after {self.cross_paradigm_iterations} iterations")
                break
            
            # Update for next iteration
            symbolic_outcomes = improved_symbolic
            neural_metrics = improved_neural
            self.cross_paradigm_iterations += 1
        
        # Calculate overall confidence
        symbolic_confidence = self._calculate_symbolic_confidence(symbolic_outcomes)
        neural_confidence = self._calculate_neural_confidence(neural_metrics)
        
        integrated_feedback["confidence"] = (
            symbolic_confidence * self.symbolic_weight +
            neural_confidence * self.neural_weight
        ) / (self.symbolic_weight + self.neural_weight)
        
        # Generate improvement suggestions
        integrated_feedback["improvement_suggestions"] = self._generate_improvement_suggestions(
            symbolic_outcomes,
            neural_metrics
        )
        
        # Final integrated state
        integrated_feedback["final_symbolic_state"] = symbolic_outcomes
        integrated_feedback["final_neural_state"] = neural_metrics
        integrated_feedback["iterations"] = self.cross_paradigm_iterations
        
        return integrated_feedback
    
    def _apply_neural_feedback_to_symbolic(
        self,
        symbolic_outcomes: Dict[str, Any],
        neural_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply neural feedback to improve symbolic outcomes."""
        improved_outcomes = symbolic_outcomes.copy()
        
        # Aggregate confidence adjustments
        total_adjustment = 0.0
        for feedback_type, feedback in neural_feedback.items():
            total_adjustment += feedback.get("confidence_adjustment", 0.0)
        
        # Apply to each outcome
        for outcome_type in improved_outcomes:
            outcome = improved_outcomes[outcome_type]
            
            # Update confidence if available
            if "confidence" in outcome:
                outcome["confidence"] = min(1.0, max(0.1, 
                    outcome["confidence"] + total_adjustment))
            
            # Potentially flip success for borderline cases
            if "success" in outcome and "confidence" in outcome:
                if outcome["success"] and outcome["confidence"] < 0.3:
                    outcome["success"] = False
                elif not outcome["success"] and outcome["confidence"] > 0.7:
                    outcome["success"] = True
        
        return improved_outcomes
    
    def _apply_symbolic_rewards_to_neural(
        self,
        neural_metrics: Dict[str, float],
        symbolic_rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply symbolic rewards to adjust neural metrics."""
        improved_metrics = neural_metrics.copy()
        
        # Calculate average symbolic reward
        if symbolic_rewards:
            avg_reward = sum(symbolic_rewards.values()) / len(symbolic_rewards)
        else:
            avg_reward = 0.0
        
        # Apply to each metric
        for metric_name in improved_metrics:
            # Check if this is a metric that should be adjusted
            is_adjustable = metric_name in [rt.value for rt in NeuralRewardType]
            
            if is_adjustable:
                # Get mapping to determine if this is an inverse metric
                try:
                    reward_type = NeuralRewardType(metric_name)
                    mapping = self.neural_reward_mappings.get(reward_type)
                    is_inverse = mapping.get("inverse", False) if mapping else False
                except ValueError:
                    is_inverse = False
                
                # Adjust based on symbolic reward
                adjustment = avg_reward * 0.1
                
                # For inverse metrics, adjust in opposite direction
                if is_inverse:
                    adjustment = -adjustment
                
                # Apply adjustment
                improved_metrics[metric_name] = min(1.0, max(0.0,
                    improved_metrics[metric_name] + adjustment))
        
        return improved_metrics
    
    def _calculate_change(self, old_dict: Dict, new_dict: Dict) -> float:
        """Calculate the average change between two dictionaries."""
        total_change = 0.0
        count = 0
        
        # For numeric values in dictionaries
        for key in old_dict:
            if key in new_dict:
                if isinstance(old_dict[key], (int, float)) and isinstance(new_dict[key], (int, float)):
                    total_change += abs(old_dict[key] - new_dict[key])
                    count += 1
                elif isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    # For nested dictionaries, recurse
                    if "confidence" in old_dict[key] and "confidence" in new_dict[key]:
                        total_change += abs(old_dict[key]["confidence"] - new_dict[key]["confidence"])
                        count += 1
        
        return total_change / max(1, count)
    
    def _calculate_symbolic_confidence(self, symbolic_outcomes: Dict[str, Any]) -> float:
        """Calculate overall confidence in symbolic outcomes."""
        confidences = []
        
        for outcome_type, outcome in symbolic_outcomes.items():
            if "confidence" in outcome:
                # Weight by success
                if outcome.get("success", False):
                    confidences.append(outcome["confidence"])
                else:
                    confidences.append(1.0 - outcome["confidence"])
        
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default confidence
    
    def _calculate_neural_confidence(self, neural_metrics: Dict[str, float]) -> float:
        """Calculate overall confidence in neural metrics."""
        # Get values only for known neural reward types
        values = []
        
        for metric_name, value in neural_metrics.items():
            if metric_name in [rt.value for rt in NeuralRewardType]:
                try:
                    reward_type = NeuralRewardType(metric_name)
                    mapping = self.neural_reward_mappings.get(reward_type)
                    
                    if mapping:
                        # For inverse metrics, higher value = lower confidence
                        is_inverse = mapping.get("inverse", False)
                        normalized_value = value if not is_inverse else 1.0 - value
                        
                        # Weight by importance
                        weight = mapping.get("weight", 1.0)
                        values.append((normalized_value, weight))
                except ValueError:
                    continue
        
        if values:
            weighted_sum = sum(value * weight for value, weight in values)
            total_weight = sum(weight for _, weight in values)
            return weighted_sum / total_weight
        else:
            return 0.5  # Default confidence
    
    def _generate_improvement_suggestions(
        self,
        symbolic_outcomes: Dict[str, Any],
        neural_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate suggestions for improving cross-paradigm integration."""
        suggestions = []
        
        # Check symbolic confidence
        symbolic_confidence = self._calculate_symbolic_confidence(symbolic_outcomes)
        if symbolic_confidence < 0.4:
            suggestions.append("Improve symbolic reasoning confidence through additional rules or facts")
        
        # Check neural confidence
        neural_confidence = self._calculate_neural_confidence(neural_metrics)
        if neural_confidence < 0.4:
            suggestions.append("Improve neural processing through additional training or finetuning")
        
        # Check specific metrics
        for metric_name, value in neural_metrics.items():
            if metric_name == "information_retention" and value < 0.5:
                suggestions.append("Focus on improving information retention in neural components")
            elif metric_name == "reconstruction_error" and value > 0.5:
                suggestions.append("Reduce reconstruction error in neural encoding/decoding")
        
        # Check specific outcomes
        for outcome_type, outcome in symbolic_outcomes.items():
            if outcome_type == "logical_consistency" and not outcome.get("success", True):
                suggestions.append("Address logical inconsistencies in symbolic reasoning")
            elif outcome_type == "contradiction_detection" and outcome.get("success", False):
                suggestions.append("Resolve detected contradictions in knowledge base")
        
        return suggestions
    
    def register_with_unified_reward_system(self) -> bool:
        """
        Register this bridge as a component in the unified reward system.
        
        Returns:
            Boolean indicating success
        """
        if self.unified_reward_system is None:
            logger.warning("No unified reward system available for registration")
            return False
        
        # Create a reward component for symbolic-neural integration
        class SymbolicNeuralIntegrationComponent(RewardComponent):
            def __init__(self, bridge, weight=1.5):
                super().__init__(
                    name="symbolic_neural_integration",
                    weight=weight,
                    scaling_factor=1.0,
                    target_models=None  # Applies to all models
                )
                self.bridge = bridge
            
            def calculate(self, system_state: Dict[str, Any]) -> float:
                # Extract symbolic outcomes and neural metrics from system state
                symbolic_outcomes = system_state.get("symbolic_outcomes", {})
                neural_metrics = system_state.get("neural_metrics", {})
                
                # Process feedback across paradigms
                feedback = self.bridge.process_cross_paradigm_feedback(
                    symbolic_outcomes,
                    neural_metrics
                )
                
                # Return overall confidence as reward
                return (feedback["confidence"] * 2) - 1  # Scale to [-1, 1]
        
        # Create and register the component
        component = SymbolicNeuralIntegrationComponent(self, weight=2.0)
        self.unified_reward_system.register_component(component)
        
        logger.info("Registered symbolic-neural integration component with unified reward system")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about reward propagation.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate success rate
        total_symbolic = self.symbolic_success_count + self.symbolic_failure_count
        symbolic_success_rate = (
            self.symbolic_success_count / total_symbolic 
            if total_symbolic > 0 else 0.0
        )
        
        # Calculate average rewards
        avg_symbolic_reward = (
            sum(entry["reward"] for entry in self.symbolic_reward_history) / 
            len(self.symbolic_reward_history) if self.symbolic_reward_history else 0.0
        )
        
        # Calculate neural performance
        neural_performance = {}
        for reward_type in NeuralRewardType:
            entries = [e for e in self.neural_reward_history if e["type"] == reward_type.value]
            if entries:
                avg_value = sum(e["value"] for e in entries) / len(entries)
                neural_performance[reward_type.value] = avg_value
        
        return {
            "symbolic_success_count": self.symbolic_success_count,
            "symbolic_failure_count": self.symbolic_failure_count,
            "symbolic_success_rate": symbolic_success_rate,
            "average_symbolic_reward": avg_symbolic_reward,
            "neural_performance": neural_performance,
            "cross_paradigm_iterations": self.cross_paradigm_iterations,
            "symbolic_reward_history_length": len(self.symbolic_reward_history),
            "neural_reward_history_length": len(self.neural_reward_history)
        }


def create_reward_propagation_bridge(
    unified_reward_system: Optional[UnifiedRewardSystem] = None,
    vector_symbolic_adapter: Optional[VectorSymbolicAdapter] = None,
    neural_symbolic_bridge: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> RewardPropagationBridge:
    """
    Create a reward propagation bridge with the specified configuration.
    
    Args:
        unified_reward_system: The unified reward system
        vector_symbolic_adapter: Adapter for vector symbolic operations
        neural_symbolic_bridge: Bridge between neural and symbolic components
        config: Configuration dictionary
        
    Returns:
        Configured RewardPropagationBridge instance
    """
    bridge = RewardPropagationBridge(
        unified_reward_system=unified_reward_system,
        vector_symbolic_adapter=vector_symbolic_adapter,
        neural_symbolic_bridge=neural_symbolic_bridge,
        config=config
    )
    
    # Register with unified reward system if available
    if unified_reward_system is not None:
        bridge.register_with_unified_reward_system()
    
    return bridge


# Example/test function
def test_reward_propagation():
    """Test the reward propagation bridge with sample data."""
    # Create a bridge
    bridge = create_reward_propagation_bridge()
    
    # Sample symbolic outcomes
    symbolic_outcomes = {
        "logical_consistency": {
            "success": True,
            "confidence": 0.8,
            "details": "No logical contradictions found"
        },
        "knowledge_retrieval": {
            "success": True,
            "confidence": 0.9,
            "details": "Successfully retrieved relevant knowledge"
        },
        "constraint_satisfaction": {
            "success": False,
            "confidence": 0.7,
            "details": "Some constraints were violated"
        }
    }
    
    # Sample neural metrics
    neural_metrics = {
        "information_retention": 0.85,
        "reconstruction_error": 0.25,
        "prediction_accuracy": 0.7,
        "uncertainty_reduction": 0.6
    }
    
    # Test symbolic to neural conversion
    for outcome_type, outcome in symbolic_outcomes.items():
        reward = bridge.symbolic_to_neural_reward(outcome, reward_type=outcome_type)
        print(f"Symbolic outcome '{outcome_type}' -> Neural reward: {reward:.4f}")
    
    # Test neural to symbolic conversion
    for metric_type, value in neural_metrics.items():
        feedback = bridge.neural_to_symbolic_feedback(
            {metric_type: value},
            reward_type=metric_type
        )
        print(f"Neural metric '{metric_type}' -> Symbolic feedback: {feedback['performance']}, "
              f"confidence adj: {feedback['confidence_adjustment']:.4f}")
    
    # Test cross-paradigm process
    integrated_feedback = bridge.process_cross_paradigm_feedback(
        symbolic_outcomes,
        neural_metrics
    )
    
    print(f"Integrated confidence: {integrated_feedback['confidence']:.4f}")
    print(f"Cross-paradigm iterations: {integrated_feedback['iterations']}")
    
    if integrated_feedback["improvement_suggestions"]:
        print("Improvement suggestions:")
        for suggestion in integrated_feedback["improvement_suggestions"]:
            print(f"- {suggestion}")
    
    # Print statistics
    stats = bridge.get_statistics()
    print(f"Symbolic success rate: {stats['symbolic_success_rate']:.2f}")
    
    return bridge


if __name__ == "__main__":
    test_reward_propagation() 