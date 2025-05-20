#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Unified Reward System

This module contains comprehensive tests for the reward components and
unified reward system to ensure proper functionality of the reward mechanisms.
"""

import unittest
import torch
import numpy as np
from typing import Dict, Any

from unified_reward_system import (
    RewardComponent,
    InformationRetentionComponent,
    LogicalConsistencyComponent,
    NoveltyRelevanceComponent,
    SystemIntegrationComponent,
    UnifiedRewardSystem,
    create_reward_system
)


class TestRewardComponents(unittest.TestCase):
    """Test cases for individual reward components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock system state for testing
        self.system_state = {
            "input_data": "What is the capital of France?",
            "output_data": "The capital of France is Paris.",
            "input_vector": torch.tensor([0.1, 0.2, 0.3]),
            "output_vector": torch.tensor([0.15, 0.25, 0.35]),
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
                },
                "text": "Since France is a country and countries have capitals, Paris is the capital of France."
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
        
    def test_information_retention_component(self):
        """Test information retention component with various data types."""
        # Create component
        component = InformationRetentionComponent()
        
        # Test with tensor data
        reward = component.calculate(self.system_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test with text data
        test_state = {
            "input_data": "What is the capital of France?",
            "output_data": "Paris is the capital of France."
        }
        reward = component.calculate(test_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test with missing data
        test_state = {"input_data": "Test"}
        reward = component.calculate(test_state)
        self.assertEqual(reward, 0.0)
        
    def test_logical_consistency_component(self):
        """Test logical consistency component."""
        # Create component
        component = LogicalConsistencyComponent()
        
        # Test with logical structure
        reward = component.calculate(self.system_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test with contradictory conclusions
        test_state = {
            "reasoning_output": {
                "logical_structure": {
                    "premises": ["X is true", "X implies Y"],
                    "conclusions": ["Y is true", "Y is false"],
                    "support_score": 0.5
                }
            }
        }
        reward = component.calculate(test_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(reward < 0)  # Should penalize contradiction
        
        # Test with text only
        test_state = {
            "reasoning_output": {
                "text": "Since X is true and X implies Y, therefore Y is true."
            }
        }
        reward = component.calculate(test_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
    def test_novelty_relevance_component(self):
        """Test novelty and relevance component."""
        # Create component
        component = NoveltyRelevanceComponent()
        
        # Test with vectors
        reward = component.calculate(self.system_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test with text
        test_state = {
            "input_data": "What is the capital of France?",
            "output_data": "The capital of France is Paris.",
            "context": {
                "previous_outputs": [
                    "The capital of Germany is Berlin.",
                    "Paris has many famous landmarks."
                ]
            }
        }
        reward = component.calculate(test_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test with highly irrelevant output
        test_state = {
            "input_data": "What is the capital of France?",
            "output_data": "Bananas are yellow fruits that grow in tropical climates.",
            "context": {"previous_outputs": []}
        }
        reward = component.calculate(test_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(reward < 0)  # Should penalize irrelevance
        
    def test_system_integration_component(self):
        """Test system integration component."""
        # Create component
        component = SystemIntegrationComponent()
        
        # Test with interaction data
        reward = component.calculate(self.system_state)
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Test interface health report
        health_report = component.get_interface_health_report()
        self.assertIsInstance(health_report, dict)
        for interface, score in health_report.items():
            self.assertTrue(0.0 <= score <= 1.0)
        
        # Test with no interactions
        test_state = {"model_interactions": {}}
        reward = component.calculate(test_state)
        self.assertEqual(reward, 0.0)


class TestUnifiedRewardSystem(unittest.TestCase):
    """Test cases for the unified reward system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create reward system
        self.reward_system = create_reward_system()
        
        # Create mock system state
        self.system_state = {
            "input_data": "What is the capital of France?",
            "output_data": "The capital of France is Paris.",
            "input_vector": torch.tensor([0.1, 0.2, 0.3]),
            "output_vector": torch.tensor([0.15, 0.25, 0.35]),
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
    
    def test_register_component(self):
        """Test registering custom components."""
        # Create custom component
        class CustomComponent(RewardComponent):
            def calculate(self, system_state):
                return 0.5
        
        # Register component
        custom_component = CustomComponent(name="custom", weight=2.0)
        self.reward_system.register_component(custom_component)
        
        # Check if registered
        self.assertIn("custom", self.reward_system.components)
        self.assertEqual(self.reward_system.components["custom"].weight, 2.0)
    
    def test_global_reward_calculation(self):
        """Test calculation of global reward."""
        # Calculate global reward
        reward = self.reward_system.calculate_global_reward(self.system_state)
        
        # Check result
        self.assertIsInstance(reward, float)
        self.assertTrue(-1.0 <= reward <= 1.0)
        
        # Check history update
        self.assertEqual(len(self.reward_system.global_reward_history), 1)
        self.assertEqual(self.reward_system.global_reward_history[0], reward)
    
    def test_model_reward_calculation(self):
        """Test calculation of model-specific rewards."""
        # Test for each model type
        for model_type in ["memory", "reasoning", "perceptual", "decision"]:
            reward = self.reward_system.calculate_model_reward(model_type, self.system_state)
            
            # Check result
            self.assertIsInstance(reward, float)
            self.assertTrue(-1.0 <= reward <= 1.0)
            
            # Check history update
            self.assertGreaterEqual(len(self.reward_system.model_reward_history[model_type]), 1)
            
    def test_reward_propagation(self):
        """Test reward propagation through the system hierarchy."""
        # Calculate global and model rewards
        global_reward = self.reward_system.calculate_global_reward(self.system_state)
        model_rewards = {
            "memory": self.reward_system.calculate_model_reward("memory", self.system_state),
            "reasoning": self.reward_system.calculate_model_reward("reasoning", self.system_state),
            "decision": self.reward_system.calculate_model_reward("decision", self.system_state)
        }
        
        # Test propagation with different training approaches
        for approach in ["pretraining", "hybrid", "absolute_zero"]:
            # Set approach
            self.reward_system.set_training_approach(approach)
            
            # Propagate rewards
            propagated = self.reward_system.propagate_reward(global_reward, model_rewards)
            
            # Check results
            self.assertIsInstance(propagated, dict)
            for model_type, reward in propagated.items():
                self.assertIsInstance(reward, float)
                self.assertTrue(-1.0 <= reward <= 1.0)
    
    def test_training_approach_setting(self):
        """Test setting training approach."""
        # Test valid approaches
        for approach in ["pretraining", "hybrid", "absolute_zero"]:
            self.reward_system.set_training_approach(approach, temperature=2.0)
            self.assertEqual(self.reward_system.training_approach, approach)
            self.assertEqual(self.reward_system.global_temperature, 2.0)
        
        # Test invalid approach
        self.reward_system.set_training_approach("invalid_approach")
        self.assertEqual(self.reward_system.training_approach, "hybrid")  # Should default to hybrid
    
    def test_component_analysis(self):
        """Test component analysis functionality."""
        # Run a reward calculation to populate history
        self.reward_system.calculate_global_reward(self.system_state)
        
        # Get component analysis
        analysis = self.reward_system.get_component_analysis()
        
        # Check structure
        self.assertIsInstance(analysis, dict)
        for component_name, metrics in analysis.items():
            self.assertIn(component_name, self.reward_system.components)
            self.assertIn("current_value", metrics)
            self.assertIn("baseline", metrics)
            self.assertIn("improvement", metrics)
            self.assertIn("weight", metrics)
    
    def test_model_performance_analysis(self):
        """Test model performance analysis functionality."""
        # Calculate rewards for multiple models
        for model_type in ["memory", "reasoning", "decision"]:
            self.reward_system.calculate_model_reward(model_type, self.system_state)
        
        # Get performance analysis
        analysis = self.reward_system.get_model_performance_analysis()
        
        # Check structure
        self.assertIsInstance(analysis, dict)
        for model_type, metrics in analysis.items():
            self.assertIn("current_reward", metrics)
            self.assertIn("mean_reward", metrics)
            self.assertIn("trend", metrics)
    
    def test_trend_calculation(self):
        """Test trend calculation functionality."""
        # Test with increasing values
        increasing = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = self.reward_system._calculate_trend(increasing)
        self.assertGreater(trend, 0)
        
        # Test with decreasing values
        decreasing = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = self.reward_system._calculate_trend(decreasing)
        self.assertLess(trend, 0)
        
        # Test with stable values
        stable = [0.3, 0.3, 0.3, 0.3]
        trend = self.reward_system._calculate_trend(stable)
        self.assertAlmostEqual(trend, 0.0)


if __name__ == "__main__":
    unittest.main() 