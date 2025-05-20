#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Reward System for SNN Models

This module implements a hierarchical reward system for the SNN model ecosystem,
providing a unified objective function framework that coordinates learning across
all component models while supporting various training approaches.

Key features:
- Global system-level objectives
- Component-specific reward signals
- Hierarchical reward propagation
- Support for different training approaches (pretraining, hybrid, absolute zero)
- Integration with both supervised and reinforcement learning
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedRewardSystem")

class RewardComponent:
    """
    Base class for reward components that contribute to the overall system objective.
    
    Each component represents a specific aspect of performance that should be 
    optimized across the system (e.g., information retention, logical consistency).
    """
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        scaling_factor: float = 1.0,
        target_models: Optional[List[str]] = None
    ):
        """
        Initialize a reward component.
        
        Args:
            name: Unique name for this reward component
            weight: Relative importance in the global objective (0-10)
            scaling_factor: Scaling factor to normalize component values
            target_models: List of model types this component applies to, or None for all
        """
        self.name = name
        self.weight = weight
        self.scaling_factor = scaling_factor
        self.target_models = target_models or []
        self.history = []
        self.baseline = 0.0
        self.last_value = 0.0
        
    def calculate(self, system_state: Dict[str, Any]) -> float:
        """
        Calculate this component's contribution to the reward.
        
        Args:
            system_state: Dictionary containing the current state of the system
            
        Returns:
            Component reward value (normalized between -1 and 1)
        """
        # This is a base implementation that should be overridden
        # by specific component implementations
        return 0.0
    
    def update_baseline(self, decay_factor: float = 0.95):
        """
        Update the baseline value based on history.
        
        Args:
            decay_factor: Weight of historical values vs. recent ones
        """
        if not self.history:
            self.baseline = 0.0
            return
            
        # Update baseline with exponential moving average
        if self.baseline == 0.0:  # First update
            self.baseline = self.history[-1]
        else:
            self.baseline = decay_factor * self.baseline + (1 - decay_factor) * self.history[-1]
    
    def get_relative_improvement(self) -> float:
        """
        Calculate improvement relative to baseline.
        
        Returns:
            Relative improvement (-1 to 1 scale)
        """
        if self.baseline == 0.0:
            return 0.0
            
        # Calculate relative improvement
        improvement = (self.last_value - self.baseline) / max(abs(self.baseline), 1e-6)
        
        # Clip to reasonable range
        return max(-1.0, min(1.0, improvement))


class InformationRetentionComponent(RewardComponent):
    """
    Reward component that measures information retention across processing steps.
    
    This component calculates how well information is preserved when passing
    through different models and transformations.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        target_models: Optional[List[str]] = None
    ):
        """Initialize the information retention component."""
        super().__init__(
            name="information_retention",
            weight=weight,
            scaling_factor=1.0,
            target_models=target_models or ["memory", "perceptual", "decision"]
        )
    
    def calculate(self, system_state: Dict[str, Any]) -> float:
        """
        Calculate information retention reward component.
        
        Args:
            system_state: Dictionary with input/output pairs for evaluation
            
        Returns:
            Information retention score (-1 to 1)
        """
        # Extract relevant information from system state
        input_data = system_state.get("input_data")
        output_data = system_state.get("output_data")
        
        if input_data is None or output_data is None:
            return 0.0
        
        # Convert tensors to appropriate format
        if isinstance(input_data, torch.Tensor) and isinstance(output_data, torch.Tensor):
            # Calculate cosine similarity if dimensions match
            if input_data.shape == output_data.shape:
                similarity = torch.nn.functional.cosine_similarity(
                    input_data.reshape(1, -1),
                    output_data.reshape(1, -1)
                ).item()
                
                # Scale from [-1, 1] to [0, 1] for retention
                retention = (similarity + 1) / 2
                
                # Update history and last value
                self.history.append(retention)
                self.last_value = retention
                
                # Return normalized retention score
                return (retention * 2) - 1  # Scale back to [-1, 1]
        
        # For non-tensor data or dimension mismatch, use a custom similarity metric
        retention = self._calculate_custom_similarity(input_data, output_data)
        
        # Update history and last value
        self.history.append(retention)
        self.last_value = retention
        
        # Return normalized retention score
        return (retention * 2) - 1  # Scale to [-1, 1]
    
    def _calculate_custom_similarity(
        self,
        input_data: Any,
        output_data: Any
    ) -> float:
        """
        Calculate similarity for non-tensor data.
        
        Args:
            input_data: Input data
            output_data: Output data
            
        Returns:
            Similarity score (0 to 1)
        """
        # Handle text data
        if isinstance(input_data, str) and isinstance(output_data, str):
            # Simple word overlap measure
            input_words = set(input_data.lower().split())
            output_words = set(output_data.lower().split())
            
            if not input_words:
                return 0.0
                
            # Calculate Jaccard similarity
            intersection = len(input_words.intersection(output_words))
            union = len(input_words.union(output_words))
            
            return intersection / union if union > 0 else 0.0
        
        # Default case
        return 0.5  # Neutral value


class LogicalConsistencyComponent(RewardComponent):
    """
    Reward component for measuring logical consistency of outputs.
    
    This checks for internal contradictions and adherence to logical rules
    in the outputs of reasoning and decision models.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        target_models: Optional[List[str]] = None
    ):
        """Initialize the logical consistency component."""
        super().__init__(
            name="logical_consistency",
            weight=weight,
            scaling_factor=1.0,
            target_models=target_models or ["reasoning", "decision"]
        )
        
        # Dictionary of contradiction patterns to check for
        self.contradiction_patterns = {
            "binary_opposites": [
                ("true", "false"),
                ("yes", "no"),
                ("increase", "decrease"),
                ("positive", "negative"),
                ("high", "low")
            ]
        }
    
    def calculate(self, system_state: Dict[str, Any]) -> float:
        """
        Calculate logical consistency reward component.
        
        Args:
            system_state: Dictionary containing reasoning chains or decision outputs
            
        Returns:
            Logical consistency score (-1 to 1)
        """
        # Extract reasoning outputs
        reasoning_output = system_state.get("reasoning_output", {})
        
        if not reasoning_output:
            return 0.0
        
        # Check for logical structure if available
        logical_structure = reasoning_output.get("logical_structure", {})
        
        if logical_structure:
            consistency = self._evaluate_logical_structure(logical_structure)
        else:
            # Fall back to text-based analysis
            output_text = reasoning_output.get("text", "")
            consistency = self._check_text_consistency(output_text)
        
        # Update history and last value
        self.history.append(consistency)
        self.last_value = consistency
        
        return (consistency * 2) - 1  # Scale to [-1, 1]
    
    def _evaluate_logical_structure(self, logical_structure: Dict) -> float:
        """
        Evaluate logical structure for consistency.
        
        Args:
            logical_structure: Dictionary with logical structure information
            
        Returns:
            Consistency score (0 to 1)
        """
        # Check for contradictory conclusions
        conclusions = logical_structure.get("conclusions", [])
        premises = logical_structure.get("premises", [])
        
        # Simple check: contradictions between conclusions
        contradiction_penalty = self._check_contradictions(conclusions)
        
        # Check for premises supporting conclusions
        support_score = logical_structure.get("support_score", 0.5)
        
        # Combine metrics
        consistency = 0.5 + (support_score / 2) - contradiction_penalty
        
        return max(0.0, min(1.0, consistency))
    
    def _check_contradictions(self, statements: List[str]) -> float:
        """
        Check for contradictions in a list of statements.
        
        Args:
            statements: List of statement strings
            
        Returns:
            Contradiction penalty (0 to 0.5)
        """
        if not statements or len(statements) < 2:
            return 0.0
            
        contradictions = 0
        
        # Check each pair of statements
        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i+1:]:
                # Check against known contradiction patterns
                for opposite_pair in self.contradiction_patterns["binary_opposites"]:
                    if (opposite_pair[0] in stmt1.lower() and 
                        opposite_pair[1] in stmt2.lower()):
                        contradictions += 1
                    elif (opposite_pair[1] in stmt1.lower() and 
                          opposite_pair[0] in stmt2.lower()):
                        contradictions += 1
        
        # Scale contradictions to penalty
        max_possible = len(statements) * (len(statements) - 1) / 2
        penalty = min(0.5, contradictions / max(1, max_possible))
        
        return penalty
    
    def _check_text_consistency(self, text: str) -> float:
        """
        Check text for logical consistency.
        
        Args:
            text: Output text to analyze
            
        Returns:
            Consistency score (0 to 1)
        """
        if not text:
            return 0.5  # Neutral score for empty text
            
        # Simple heuristic consistency checking
        # Check for explicit contradictions
        contradiction_markers = [
            "but actually", "in contrast to what I said", 
            "contrary to", "this contradicts", "this conflicts with"
        ]
        
        contradiction_count = sum(text.lower().count(marker) for marker in contradiction_markers)
        
        # Check for logical flow markers
        logical_markers = [
            "therefore", "thus", "because", "since", 
            "as a result", "consequently", "it follows that"
        ]
        
        logical_marker_count = sum(text.lower().count(marker) for marker in logical_markers)
        
        # Calculate consistency score
        base_score = 0.5
        contradiction_penalty = min(0.5, contradiction_count * 0.1)
        logical_bonus = min(0.3, logical_marker_count * 0.05)
        
        consistency = base_score + logical_bonus - contradiction_penalty
        
        return max(0.0, min(1.0, consistency))


class NoveltyRelevanceComponent(RewardComponent):
    """
    Reward component for balancing novelty and relevance in outputs.
    
    This component rewards outputs that introduce novel information while
    remaining relevant to the input queries or context.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        novelty_weight: float = 0.4,
        relevance_weight: float = 0.6,
        target_models: Optional[List[str]] = None
    ):
        """
        Initialize the novelty and relevance component.
        
        Args:
            weight: Overall component weight
            novelty_weight: Weight of novelty in the balance (0-1)
            relevance_weight: Weight of relevance in the balance (0-1)
            target_models: List of model types this component applies to
        """
        super().__init__(
            name="novelty_relevance",
            weight=weight,
            scaling_factor=1.0,
            target_models=target_models or ["memory", "reasoning", "metacognitive"]
        )
        
        self.novelty_weight = novelty_weight
        self.relevance_weight = relevance_weight
        self.reference_embeddings = {}  # Store reference embeddings
    
    def calculate(self, system_state: Dict[str, Any]) -> float:
        """
        Calculate novelty and relevance reward component.
        
        Args:
            system_state: Dictionary with input, output, and context
            
        Returns:
            Novelty-relevance score (-1 to 1)
        """
        # Extract relevant information
        input_data = system_state.get("input_data")
        output_data = system_state.get("output_data")
        context = system_state.get("context", {})
        
        if input_data is None or output_data is None:
            return 0.0
        
        # Get vector representations if available
        input_vector = system_state.get("input_vector")
        output_vector = system_state.get("output_vector")
        
        # Get embeddings if vectors are not available
        if input_vector is None or output_vector is None:
            # If text data, use string similarity as fallback
            if isinstance(input_data, str) and isinstance(output_data, str):
                return self._calculate_from_text(input_data, output_data, context)
            return 0.0  # Can't calculate without vectors or text
        
        # Calculate relevance as cosine similarity
        relevance = max(0, torch.nn.functional.cosine_similarity(
            input_vector.reshape(1, -1),
            output_vector.reshape(1, -1)
        ).item())
        
        # Calculate novelty relative to previous outputs
        previous_outputs = context.get("previous_outputs", [])
        if previous_outputs and isinstance(previous_outputs[0], torch.Tensor):
            novelty_scores = []
            for prev_output in previous_outputs[-5:]:  # Consider last 5 outputs
                similarity = torch.nn.functional.cosine_similarity(
                    output_vector.reshape(1, -1),
                    prev_output.reshape(1, -1)
                ).item()
                # Novelty is inverse of similarity
                novelty_scores.append(1.0 - max(0, similarity))
            
            novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
        else:
            novelty = 0.5  # Default novelty if no previous outputs
        
        # Combine novelty and relevance
        combined_score = (self.novelty_weight * novelty) + (self.relevance_weight * relevance)
        
        # Update history and last value
        self.history.append(combined_score)
        self.last_value = combined_score
        
        return (combined_score * 2) - 1  # Scale to [-1, 1]
    
    def _calculate_from_text(
        self,
        input_text: str,
        output_text: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate novelty-relevance from text inputs and outputs.
        
        Args:
            input_text: Input text
            output_text: Output text
            context: Additional context
            
        Returns:
            Novelty-relevance score (-1 to 1)
        """
        # Calculate word overlap for relevance
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        if not input_words:
            return 0.0
            
        # Calculate relevance as Jaccard similarity
        intersection = len(input_words.intersection(output_words))
        union = len(input_words.union(output_words))
        relevance = intersection / union if union > 0 else 0.0
        
        # Calculate novelty relative to previous outputs
        previous_outputs = context.get("previous_outputs", [])
        if previous_outputs and isinstance(previous_outputs[0], str):
            novelty_scores = []
            for prev_output in previous_outputs[-5:]:  # Consider last 5 outputs
                prev_words = set(prev_output.lower().split())
                if not prev_words:
                    continue
                
                # Calculate Jaccard similarity with previous output
                intersection = len(output_words.intersection(prev_words))
                union = len(output_words.union(prev_words))
                similarity = intersection / union if union > 0 else 0.0
                
                # Novelty is inverse of similarity
                novelty_scores.append(1.0 - similarity)
            
            novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
        else:
            novelty = 0.5  # Default novelty
        
        # Combine novelty and relevance
        combined_score = (self.novelty_weight * novelty) + (self.relevance_weight * relevance)
        
        return (combined_score * 2) - 1  # Scale to [-1, 1]


class SystemIntegrationComponent(RewardComponent):
    """
    Reward component for measuring integration between different SNN models.
    
    This component evaluates how effectively different models work together,
    focusing on information transfer across model boundaries.
    """
    
    def __init__(
        self,
        weight: float = 1.5,  # Higher weight due to importance for system cohesion
        target_models: Optional[List[str]] = None
    ):
        """Initialize the system integration component."""
        super().__init__(
            name="system_integration",
            weight=weight,
            scaling_factor=1.0,
            target_models=target_models  # This applies to all models
        )
        
        # Integration metrics
        self.transfer_success_rate = 0.0
        self.interface_health = {}
    
    def calculate(self, system_state: Dict[str, Any]) -> float:
        """
        Calculate system integration reward component.
        
        Args:
            system_state: Dictionary with cross-model interactions
            
        Returns:
            System integration score (-1 to 1)
        """
        # Extract model interactions
        model_interactions = system_state.get("model_interactions", {})
        
        if not model_interactions:
            return 0.0
        
        # Calculate integration metrics
        information_transfer = []
        
        # Process each model interaction
        for source_model, targets in model_interactions.items():
            for target_model, interaction_data in targets.items():
                # Calculate transfer success
                transfer_score = interaction_data.get("transfer_success", 0.0)
                information_transfer.append(transfer_score)
                
                # Update interface health
                interface_key = f"{source_model}_{target_model}"
                
                if interface_key not in self.interface_health:
                    self.interface_health[interface_key] = []
                
                self.interface_health[interface_key].append(transfer_score)
                
                # Keep only recent history
                if len(self.interface_health[interface_key]) > 20:
                    self.interface_health[interface_key].pop(0)
        
        # Calculate overall transfer success rate
        if information_transfer:
            self.transfer_success_rate = sum(information_transfer) / len(information_transfer)
        
        # Calculate interface health variance
        interface_variance = 0.0
        if self.interface_health:
            # Calculate variance across interface health metrics
            mean_health = sum(sum(scores) / len(scores) for scores in self.interface_health.values()) / len(self.interface_health)
            
            variance_sum = sum(((sum(scores) / len(scores)) - mean_health) ** 2 
                              for scores in self.interface_health.values())
            
            interface_variance = variance_sum / len(self.interface_health)
        
        # Calculate integration score
        # High transfer success and low variance is ideal
        integration_score = self.transfer_success_rate * (1.0 - min(1.0, interface_variance * 2))
        
        # Update history and last value
        self.history.append(integration_score)
        self.last_value = integration_score
        
        return (integration_score * 2) - 1  # Scale to [-1, 1]
    
    def get_interface_health_report(self) -> Dict[str, float]:
        """
        Generate a report on interface health between models.
        
        Returns:
            Dictionary mapping interface names to health scores (0-1)
        """
        health_report = {}
        
        for interface, scores in self.interface_health.items():
            if scores:
                # Calculate recent health
                recent_health = sum(scores[-5:]) / min(5, len(scores))
                health_report[interface] = recent_health
        
        return health_report


class UnifiedRewardSystem:
    """
    Main class that manages the unified reward system for all SNN models.
    
    This class coordinates reward components, calculates global and local rewards,
    and provides interfaces for different training approaches.
    """
    
    def __init__(self):
        """Initialize the unified reward system."""
        # Register reward components
        self.components = {}
        self.register_default_components()
        
        # Model-specific objectives
        self.model_objectives = {}
        self.initialize_model_objectives()
        
        # Training configuration
        self.training_approach = "hybrid"  # Options: "pretraining", "hybrid", "absolute_zero"
        self.global_temperature = 1.0      # For temperature-based training schedules
        
        # Metrics tracking
        self.global_reward_history = []
        self.model_reward_history = defaultdict(list)
    
    def register_default_components(self):
        """Register default reward components."""
        # Core components
        self.register_component(InformationRetentionComponent(weight=1.0))
        self.register_component(LogicalConsistencyComponent(weight=1.2))
        self.register_component(NoveltyRelevanceComponent(weight=0.8))
        self.register_component(SystemIntegrationComponent(weight=1.5))
    
    def register_component(self, component: RewardComponent):
        """
        Register a reward component.
        
        Args:
            component: RewardComponent instance to register
        """
        self.components[component.name] = component
        logger.info(f"Registered reward component: {component.name}")
    
    def initialize_model_objectives(self):
        """
        Initialize model-specific objectives.
        
        This defines which components are most important for each model type
        and their relative weights.
        """
        # Memory SNN objectives
        self.model_objectives["memory"] = {
            "information_retention": 1.5,  # Critical for memory
            "novelty_relevance": 1.2,      # Important for useful memories
            "system_integration": 1.0      # Standard weight
        }
        
        # Reasoning SNN objectives
        self.model_objectives["reasoning"] = {
            "logical_consistency": 1.8,    # Critical for reasoning
            "information_retention": 0.8,  # Important but secondary
            "system_integration": 1.0      # Standard weight
        }
        
        # Perceptual SNN objectives
        self.model_objectives["perceptual"] = {
            "information_retention": 1.5,  # Critical for perceptual fidelity
            "system_integration": 1.2      # Important for feeding other models
        }
        
        # Decision SNN objectives
        self.model_objectives["decision"] = {
            "logical_consistency": 1.5,    # Critical for decision quality
            "system_integration": 1.2,     # Important for acting on other models
            "information_retention": 0.7   # Less critical
        }
        
        # Affective SNN objectives
        self.model_objectives["affective"] = {
            "novelty_relevance": 1.3,      # Important for affect signals
            "system_integration": 1.1      # Needs good integration
        }
        
        # Metacognitive SNN objectives
        self.model_objectives["metacognitive"] = {
            "system_integration": 1.8,     # Critical for monitoring other models
            "logical_consistency": 1.3     # Important for evaluation
        }
        
        # Statistical SNN objectives
        self.model_objectives["statistical"] = {
            "information_retention": 1.4,  # Important for statistical accuracy
            "logical_consistency": 1.2     # Important for valid inferences
        }
    
    def calculate_global_reward(
        self,
        system_state: Dict[str, Any]
    ) -> float:
        """
        Calculate the global system reward.
        
        Args:
            system_state: Current state of the SNN system
            
        Returns:
            Global reward value (-1 to 1 scale)
        """
        if not self.components:
            return 0.0
        
        # Calculate each component's contribution
        component_rewards = {}
        total_weight = 0.0
        
        for name, component in self.components.items():
            # Calculate raw component value
            component_value = component.calculate(system_state)
            component_rewards[name] = component_value
            
            # Add to weighted sum
            total_weight += component.weight
        
        # Calculate weighted average
        if total_weight == 0.0:
            global_reward = 0.0
        else:
            weighted_sum = sum(component.weight * component_rewards[name] 
                               for name, component in self.components.items())
            global_reward = weighted_sum / total_weight
        
        # Update history
        self.global_reward_history.append(global_reward)
        
        # Update component baselines
        for component in self.components.values():
            component.update_baseline()
        
        return global_reward
    
    def calculate_model_reward(
        self,
        model_type: str,
        system_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a specific model type.
        
        Args:
            model_type: Type of model ("memory", "reasoning", etc.)
            system_state: Current state of the SNN system
            
        Returns:
            Model-specific reward value (-1 to 1 scale)
        """
        # Get model-specific objectives
        if model_type not in self.model_objectives:
            # Default to basic objectives if model type unknown
            model_weights = {"information_retention": 1.0, "system_integration": 1.0}
        else:
            model_weights = self.model_objectives[model_type]
        
        # Calculate components that apply to this model
        component_values = {}
        total_weight = 0.0
        
        for component_name, weight in model_weights.items():
            if component_name in self.components:
                component = self.components[component_name]
                
                # Check if component applies to this model
                if not component.target_models or model_type in component.target_models:
                    # Calculate component value
                    component_values[component_name] = component.calculate(system_state)
                    total_weight += weight
        
        # Calculate weighted average
        if total_weight == 0.0:
            model_reward = 0.0
        else:
            weighted_sum = sum(model_weights[name] * value 
                               for name, value in component_values.items())
            model_reward = weighted_sum / total_weight
        
        # Update history
        self.model_reward_history[model_type].append(model_reward)
        
        return model_reward
    
    def propagate_reward(
        self,
        global_reward: float,
        model_rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Propagate rewards through the system hierarchy.
        
        This combines global and model-specific rewards based on the
        current training approach and temperature.
        
        Args:
            global_reward: Global system reward
            model_rewards: Dictionary mapping model types to rewards
            
        Returns:
            Dictionary with final propagated rewards for each model
        """
        # Default propagation weights based on training approach
        if self.training_approach == "pretraining":
            # Focus on model-specific rewards during pretraining
            global_weight = 0.2
            model_weight = 0.8
        elif self.training_approach == "absolute_zero":
            # Focus primarily on global reward for training from scratch
            global_weight = 0.8
            model_weight = 0.2
        else:  # hybrid
            # Balanced approach for hybrid training
            global_weight = 0.5
            model_weight = 0.5
        
        # Apply temperature scaling (higher temp = more focus on global reward)
        # Temperature should be between 0.1 and 10.0
        temp_factor = min(1.0, max(0.0, (self.global_temperature - 0.1) / 9.9))
        
        # Adjust weights based on temperature
        adjusted_global = global_weight + (temp_factor * (1.0 - global_weight))
        adjusted_model = 1.0 - adjusted_global
        
        # Calculate propagated rewards
        propagated_rewards = {}
        
        for model_type, model_reward in model_rewards.items():
            # Combine global and model-specific rewards
            propagated = (adjusted_global * global_reward) + (adjusted_model * model_reward)
            propagated_rewards[model_type] = propagated
        
        return propagated_rewards
    
    def set_training_approach(self, approach: str, temperature: float = 1.0):
        """
        Set the training approach.
        
        Args:
            approach: Training approach ("pretraining", "hybrid", "absolute_zero")
            temperature: Temperature parameter (0.1-10.0)
        """
        valid_approaches = ["pretraining", "hybrid", "absolute_zero"]
        
        if approach not in valid_approaches:
            logger.warning(f"Invalid training approach: {approach}. Using 'hybrid' instead.")
            approach = "hybrid"
        
        self.training_approach = approach
        self.global_temperature = max(0.1, min(10.0, temperature))
        
        logger.info(f"Set training approach to {approach} with temperature {temperature}")
    
    def get_component_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed analysis of reward components.
        
        Returns:
            Dictionary with component metrics
        """
        component_metrics = {}
        
        for name, component in self.components.items():
            metrics = {
                "current_value": component.last_value,
                "baseline": component.baseline,
                "improvement": component.get_relative_improvement(),
                "weight": component.weight
            }
            component_metrics[name] = metrics
        
        return component_metrics
    
    def get_model_performance_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Get analysis of model-specific performance.
        
        Returns:
            Dictionary with model performance metrics
        """
        model_metrics = {}
        
        for model_type, history in self.model_reward_history.items():
            if not history:
                continue
                
            metrics = {
                "current_reward": history[-1],
                "mean_reward": sum(history[-20:]) / min(20, len(history)),
                "trend": self._calculate_trend(history[-20:])
            }
            model_metrics[model_type] = metrics
        
        return model_metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Trend score (-1 to 1, where positive means improving)
        """
        if not values or len(values) < 2:
            return 0.0
            
        # Calculate simple linear regression slope
        n = len(values)
        indices = list(range(n))
        
        mean_x = sum(indices) / n
        mean_y = sum(values) / n
        
        numerator = sum((i - mean_x) * (y - mean_y) for i, y in zip(indices, values))
        denominator = sum((i - mean_x) ** 2 for i in indices)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize to -1 to 1 range
        normalized_slope = max(-1.0, min(1.0, slope * 5.0))
        
        return normalized_slope

# Helper function to create a reward system with default components
def create_reward_system() -> UnifiedRewardSystem:
    """
    Create a unified reward system with default components.
    
    Returns:
        Initialized UnifiedRewardSystem instance
    """
    return UnifiedRewardSystem()


# Example usage (for testing)
if __name__ == "__main__":
    # Create reward system
    reward_system = create_reward_system()
    
    # Simulate system state
    system_state = {
        "input_data": "What is the capital of France?",
        "output_data": "The capital of France is Paris.",
        "context": {
            "previous_outputs": [
                "The capital of Germany is Berlin.",
                "Paris is known for the Eiffel Tower."
            ]
        },
        "reasoning_output": {
            "logical_structure": {
                "premises": ["France is a country", "Countries have capitals"],
                "conclusions": ["Paris is the capital of France"],
                "support_score": 0.8
            }
        },
        "model_interactions": {
            "memory": {
                "reasoning": {"transfer_success": 0.85},
                "decision": {"transfer_success": 0.75}
            },
            "reasoning": {
                "decision": {"transfer_success": 0.9}
            }
        }
    }
    
    # Calculate global reward
    global_reward = reward_system.calculate_global_reward(system_state)
    print(f"Global reward: {global_reward:.4f}")
    
    # Calculate model-specific rewards
    model_rewards = {}
    for model_type in ["memory", "reasoning", "decision", "affective"]:
        model_reward = reward_system.calculate_model_reward(model_type, system_state)
        model_rewards[model_type] = model_reward
        print(f"{model_type.capitalize()} model reward: {model_reward:.4f}")
    
    # Propagate rewards
    propagated = reward_system.propagate_reward(global_reward, model_rewards)
    print("\nPropagated rewards:")
    for model_type, reward in propagated.items():
        print(f"{model_type.capitalize()}: {reward:.4f}")
    
    # Component analysis
    print("\nComponent analysis:")
    component_analysis = reward_system.get_component_analysis()
    for component, metrics in component_analysis.items():
        print(f"{component}: value={metrics['current_value']:.4f}, improvement={metrics['improvement']:.4f}")
    
    # Set different training approach
    print("\nChanging to absolute_zero training:")
    reward_system.set_training_approach("absolute_zero", temperature=3.0)
    propagated = reward_system.propagate_reward(global_reward, model_rewards)
    for model_type, reward in propagated.items():
        print(f"{model_type.capitalize()}: {reward:.4f}") 