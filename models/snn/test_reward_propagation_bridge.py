#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the Reward Propagation Bridge.

This module contains unit tests to verify the functionality of the
reward propagation bridge between symbolic and neural components.
"""

import unittest
import torch
import numpy as np
from typing import Dict, Any

# Try different import paths to handle both package and direct imports
try:
    # When imported as package
    from models.snn.reward_propagation_bridge import (
        RewardPropagationBridge,
        create_reward_propagation_bridge,
        SymbolicRewardType,
        NeuralRewardType
    )
    from models.snn.unified_reward_system import (
        UnifiedRewardSystem,
        create_reward_system
    )
except ImportError:
    # When run directly
    try:
        from reward_propagation_bridge import (
            RewardPropagationBridge,
            create_reward_propagation_bridge,
            SymbolicRewardType,
            NeuralRewardType
        )
        from unified_reward_system import (
            UnifiedRewardSystem,
            create_reward_system
        )
    except ImportError:
        print("WARNING: Could not import reward_propagation_bridge. Tests may fail.")


class TestRewardPropagationBridge(unittest.TestCase):
    """Test cases for the RewardPropagationBridge."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test bridge
        self.bridge = create_reward_propagation_bridge()
        
        # Sample symbolic outcomes for testing
        self.symbolic_outcomes = {
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
        
        # Sample neural metrics for testing
        self.neural_metrics = {
            "information_retention": 0.85,
            "reconstruction_error": 0.25,
            "prediction_accuracy": 0.7,
            "uncertainty_reduction": 0.6
        }
    
    def test_creation(self):
        """Test creation of the bridge."""
        self.assertIsInstance(self.bridge, RewardPropagationBridge)
        self.assertEqual(self.bridge.symbolic_weight, 0.7)
        self.assertEqual(self.bridge.neural_weight, 0.7)
    
    def test_symbolic_to_neural_reward(self):
        """Test conversion from symbolic outcomes to neural rewards."""
        # Test successful outcome
        success_outcome = {
            "success": True,
            "confidence": 0.9,
            "details": "Test successful outcome"
        }
        
        reward = self.bridge.symbolic_to_neural_reward(
            success_outcome,
            reward_type=SymbolicRewardType.LOGICAL_CONSISTENCY
        )
        
        # Check that reward is positive for success
        self.assertGreater(reward, 0)
        
        # Test failed outcome
        failure_outcome = {
            "success": False,
            "confidence": 0.8,
            "details": "Test failed outcome"
        }
        
        reward = self.bridge.symbolic_to_neural_reward(
            failure_outcome,
            reward_type=SymbolicRewardType.CONSTRAINT_SATISFACTION
        )
        
        # Check that reward is negative for failure
        self.assertLess(reward, 0)
        
        # Test with string reward type
        reward = self.bridge.symbolic_to_neural_reward(
            success_outcome,
            reward_type="knowledge_retrieval"
        )
        
        # Check that string reward type works
        self.assertGreater(reward, 0)
    
    def test_neural_to_symbolic_feedback(self):
        """Test conversion from neural metrics to symbolic feedback."""
        # Test high performance metric
        high_metric = {
            "information_retention": 0.95
        }
        
        feedback = self.bridge.neural_to_symbolic_feedback(
            high_metric,
            reward_type=NeuralRewardType.INFORMATION_RETENTION
        )
        
        # Check that feedback is positive for high performance
        self.assertEqual(feedback["performance"], "high")
        self.assertGreater(feedback["confidence_adjustment"], 0)
        
        # Test low performance metric
        low_metric = {
            "reconstruction_error": 0.8  # High error = low performance
        }
        
        feedback = self.bridge.neural_to_symbolic_feedback(
            low_metric,
            reward_type=NeuralRewardType.RECONSTRUCTION_ERROR
        )
        
        # Check that feedback accounts for inverse metrics (high error = low performance)
        self.assertEqual(feedback["performance"], "low")
        self.assertLess(feedback["confidence_adjustment"], 0)
        
        # Test medium performance metric
        medium_metric = {
            "prediction_accuracy": 0.6
        }
        
        feedback = self.bridge.neural_to_symbolic_feedback(
            medium_metric,
            reward_type="prediction_accuracy"
        )
        
        # Check that medium performance gives neutral feedback
        self.assertEqual(feedback["performance"], "medium")
        self.assertEqual(feedback["confidence_adjustment"], 0)
    
    def test_cross_paradigm_process(self):
        """Test the cross-paradigm integration process."""
        # Process feedback
        integrated_feedback = self.bridge.process_cross_paradigm_feedback(
            self.symbolic_outcomes,
            self.neural_metrics
        )
        
        # Check that integrated feedback has expected fields
        self.assertIn("confidence", integrated_feedback)
        self.assertIn("symbolic_rewards", integrated_feedback)
        self.assertIn("neural_feedback", integrated_feedback)
        self.assertIn("improvement_suggestions", integrated_feedback)
        self.assertIn("iterations", integrated_feedback)
        
        # Check that symbolic rewards were calculated
        self.assertGreater(len(integrated_feedback["symbolic_rewards"]), 0)
        
        # Check that neural feedback was calculated
        self.assertGreater(len(integrated_feedback["neural_feedback"]), 0)
        
        # Check that iterations were performed
        self.assertGreaterEqual(integrated_feedback["iterations"], 0)
        
        # Check that overall confidence is between 0 and 1
        self.assertGreaterEqual(integrated_feedback["confidence"], 0)
        self.assertLessEqual(integrated_feedback["confidence"], 1)
    
    def test_iterative_improvement(self):
        """Test that the iterative process improves outcomes."""
        # Process with only one iteration
        self.bridge.max_iterations = 1
        single_iteration_feedback = self.bridge.process_cross_paradigm_feedback(
            self.symbolic_outcomes.copy(),
            self.neural_metrics.copy()
        )
        
        # Process with multiple iterations
        self.bridge.max_iterations = 3
        multi_iteration_feedback = self.bridge.process_cross_paradigm_feedback(
            self.symbolic_outcomes.copy(),
            self.neural_metrics.copy()
        )
        
        # Test that outcomes were modified through iterations
        if multi_iteration_feedback["iterations"] > 1:
            self.assertNotEqual(
                single_iteration_feedback["final_symbolic_state"]["constraint_satisfaction"]["confidence"],
                multi_iteration_feedback["final_symbolic_state"]["constraint_satisfaction"]["confidence"]
            )
    
    def test_statistics(self):
        """Test that statistics are correctly tracked."""
        # Initial counts should be zero
        initial_stats = self.bridge.get_statistics()
        self.assertEqual(initial_stats["symbolic_success_count"], 0)
        self.assertEqual(initial_stats["symbolic_failure_count"], 0)
        
        # Process some outcomes to generate statistics
        _ = self.bridge.process_cross_paradigm_feedback(
            self.symbolic_outcomes,
            self.neural_metrics
        )
        
        # Check that statistics were updated
        updated_stats = self.bridge.get_statistics()
        self.assertGreater(updated_stats["symbolic_success_count"], 0)
        self.assertGreater(updated_stats["symbolic_failure_count"], 0)
        self.assertGreater(len(updated_stats["neural_performance"]), 0)
    
    def test_unified_reward_integration(self):
        """Test integration with unified reward system."""
        # Create a unified reward system
        reward_system = create_reward_system()
        
        # Create a bridge with the reward system
        bridge_with_system = create_reward_propagation_bridge(
            unified_reward_system=reward_system
        )
        
        # Verify that registration worked
        self.assertTrue(any(
            component.name == "symbolic_neural_integration"
            for component in reward_system.components
        ))
        
        # Test calculating reward with the integrated component
        system_state = {
            "symbolic_outcomes": self.symbolic_outcomes,
            "neural_metrics": self.neural_metrics
        }
        
        # Calculate global reward
        global_reward = reward_system.calculate_global_reward(system_state)
        
        # Reward should be between -1 and 1
        self.assertGreaterEqual(global_reward, -1.0)
        self.assertLessEqual(global_reward, 1.0)
    
    def test_custom_config(self):
        """Test bridge with custom configuration."""
        # Create bridge with custom config
        custom_config = {
            "symbolic_weight": 0.9,
            "neural_weight": 0.5,
            "max_iterations": 5,
            "feedback_threshold": 0.8
        }
        
        custom_bridge = create_reward_propagation_bridge(
            config=custom_config
        )
        
        # Check that config was applied
        self.assertEqual(custom_bridge.symbolic_weight, 0.9)
        self.assertEqual(custom_bridge.neural_weight, 0.5)
        self.assertEqual(custom_bridge.max_iterations, 5)
        self.assertEqual(custom_bridge.feedback_threshold, 0.8)
        
        # Process with custom bridge
        feedback = custom_bridge.process_cross_paradigm_feedback(
            self.symbolic_outcomes,
            self.neural_metrics
        )
        
        # Check that symbolic weight influences are stronger
        self.assertGreater(
            sum(feedback["symbolic_rewards"].values()),
            sum(val["confidence_adjustment"] for val in feedback["neural_feedback"].values())
        )
    
    def test_improvement_suggestions(self):
        """Test generation of improvement suggestions."""
        # Create metrics with issues to trigger suggestions
        problematic_metrics = {
            "information_retention": 0.3,  # Low retention
            "reconstruction_error": 0.9    # High error
        }
        
        problematic_outcomes = {
            "logical_consistency": {
                "success": False,
                "confidence": 0.6,
                "details": "Logical contradictions found"
            },
            "contradiction_detection": {
                "success": True,
                "confidence": 0.9,
                "details": "Contradictions detected"
            }
        }
        
        # Process with problematic data
        feedback = self.bridge.process_cross_paradigm_feedback(
            problematic_outcomes,
            problematic_metrics
        )
        
        # Check that suggestions were generated
        self.assertGreater(len(feedback["improvement_suggestions"]), 0)
        
        # Check for specific suggestions
        suggestion_text = " ".join(feedback["improvement_suggestions"])
        self.assertIn("information retention", suggestion_text.lower())
        self.assertIn("logical", suggestion_text.lower())


if __name__ == "__main__":
    unittest.main() 