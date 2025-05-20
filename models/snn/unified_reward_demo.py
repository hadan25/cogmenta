#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Reward System Demonstration

This script demonstrates the key features of the Unified Reward System,
showing how it calculates rewards for different SNN models and how
it can be used in various training scenarios.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time
import logging

from unified_reward_system import (
    RewardComponent,
    InformationRetentionComponent, 
    LogicalConsistencyComponent,
    NoveltyRelevanceComponent,
    SystemIntegrationComponent,
    UnifiedRewardSystem,
    create_reward_system
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RewardDemo")


def create_sample_system_state(quality_level: float = 0.8) -> Dict[str, Any]:
    """
    Create a sample system state for demonstration.
    
    Args:
        quality_level: Quality level (0-1) to simulate better/worse performance
        
    Returns:
        System state dictionary
    """
    # Scale quality between 0.2 and 0.9 to avoid extremes
    scaled_quality = 0.2 + (quality_level * 0.7)
    
    # Create base vectors with adjustable similarity
    base_input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    noise = torch.randn(5) * (1.0 - scaled_quality) * 0.5
    base_output = base_input + noise
    base_output = base_output / base_output.norm()
    
    # Sample input/output texts
    input_text = "What is the capital of France?"
    
    # Vary output quality based on quality_level
    if scaled_quality > 0.8:
        output_text = "The capital of France is Paris, which is located in the north-central part of the country."
    elif scaled_quality > 0.6:
        output_text = "Paris is the capital of France."
    elif scaled_quality > 0.4:
        output_text = "I believe it's Paris."
    else:
        output_text = "France has many cities."
    
    # Create system state
    system_state = {
        "input_data": input_text,
        "output_data": output_text,
        "input_vector": base_input,
        "output_vector": base_output,
        "context": {
            "previous_outputs": [
                "The capital of Germany is Berlin.",
                "Paris is known for the Eiffel Tower.",
                "France is a country in Western Europe."
            ]
        },
        "reasoning_output": {
            "logical_structure": {
                "premises": [
                    "France is a country", 
                    "Countries have capitals"
                ],
                "conclusions": [
                    "Paris is the capital of France"
                ],
                "support_score": scaled_quality
            },
            "text": "Since France is a country and countries have capitals, Paris is the capital of France."
        },
        "model_interactions": {
            "memory": {
                "reasoning": {"transfer_success": scaled_quality * 0.9},
                "decision": {"transfer_success": scaled_quality * 0.8}
            },
            "reasoning": {
                "decision": {"transfer_success": scaled_quality * 0.95}
            },
            "perceptual": {
                "memory": {"transfer_success": scaled_quality * 0.85}
            }
        }
    }
    
    return system_state


def demo_quality_sensitivity():
    """Demonstrate reward system sensitivity to output quality."""
    logger.info("Demonstrating reward sensitivity to output quality...")
    
    # Create reward system
    reward_system = create_reward_system()
    
    # Try different quality levels
    quality_levels = np.linspace(0.1, 1.0, 10)
    global_rewards = []
    model_rewards = {
        "memory": [],
        "reasoning": [],
        "decision": []
    }
    
    for quality in quality_levels:
        # Create system state with varying quality
        system_state = create_sample_system_state(quality_level=quality)
        
        # Calculate global reward
        global_reward = reward_system.calculate_global_reward(system_state)
        global_rewards.append(global_reward)
        
        # Calculate model-specific rewards
        for model_type in model_rewards.keys():
            reward = reward_system.calculate_model_reward(model_type, system_state)
            model_rewards[model_type].append(reward)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(quality_levels, global_rewards, 'bo-', linewidth=2, label='Global Reward')
    plt.xlabel('System Quality')
    plt.ylabel('Reward Value')
    plt.title('Global Reward vs. System Quality')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for model_type, rewards in model_rewards.items():
        plt.plot(quality_levels, rewards, 'o-', linewidth=2, label=f'{model_type.capitalize()} Model')
    
    plt.xlabel('System Quality')
    plt.ylabel('Reward Value')
    plt.title('Model-Specific Rewards vs. System Quality')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quality_sensitivity.png')
    logger.info("Saved quality sensitivity plot to 'quality_sensitivity.png'")


def demo_training_approaches():
    """Demonstrate different training approaches."""
    logger.info("Demonstrating different training approaches...")
    
    # Create reward system
    reward_system = create_reward_system()
    
    # Create moderate quality system state
    system_state = create_sample_system_state(quality_level=0.7)
    
    # Calculate global reward
    global_reward = reward_system.calculate_global_reward(system_state)
    logger.info(f"Global reward: {global_reward:.4f}")
    
    # Calculate model rewards
    model_rewards = {}
    for model_type in ["memory", "reasoning", "decision", "perceptual"]:
        reward = reward_system.calculate_model_reward(model_type, system_state)
        model_rewards[model_type] = reward
        logger.info(f"{model_type.capitalize()} model reward: {reward:.4f}")
    
    # Compare different training approaches
    approaches = ["pretraining", "hybrid", "absolute_zero"]
    temperatures = [0.5, 1.0, 3.0]
    
    results = {}
    
    for approach in approaches:
        approach_results = {}
        
        for temp in temperatures:
            # Set training approach
            reward_system.set_training_approach(approach, temperature=temp)
            
            # Propagate rewards
            propagated = reward_system.propagate_reward(global_reward, model_rewards)
            approach_results[temp] = propagated
        
        results[approach] = approach_results
    
    # Display results
    for approach, temp_results in results.items():
        logger.info(f"\nApproach: {approach}")
        
        for temp, propagated in temp_results.items():
            logger.info(f"  Temperature: {temp}")
            
            for model_type, reward in propagated.items():
                logger.info(f"    {model_type}: {reward:.4f}")
    
    # Plot results for a specific model
    model_to_plot = "reasoning"
    
    plt.figure(figsize=(10, 6))
    
    for approach in approaches:
        temps = []
        rewards = []
        
        for temp in temperatures:
            temps.append(temp)
            rewards.append(results[approach][temp][model_to_plot])
        
        plt.plot(temps, rewards, 'o-', linewidth=2, label=approach)
    
    plt.xlabel('Temperature')
    plt.ylabel(f'Propagated Reward for {model_to_plot.capitalize()} Model')
    plt.title('Effect of Training Approach and Temperature on Propagated Rewards')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('training_approaches.png')
    logger.info("Saved training approaches plot to 'training_approaches.png'")


def demo_custom_component():
    """Demonstrate creating and using a custom reward component."""
    logger.info("Demonstrating custom reward component...")
    
    # Create a custom reward component
    class EnergySavingComponent(RewardComponent):
        """
        Reward component that incentivizes energy-efficient processing.
        
        This component rewards models that perform well while using fewer
        computational resources.
        """
        
        def __init__(self, weight: float = 1.0):
            super().__init__(
                name="energy_saving",
                weight=weight,
                scaling_factor=1.0,
                target_models=["memory", "reasoning", "perceptual", "decision"]
            )
        
        def calculate(self, system_state: Dict[str, Any]) -> float:
            # Extract resource usage if available
            resource_usage = system_state.get("resource_usage", {})
            
            if not resource_usage:
                return 0.0
            
            # Calculate efficiency score based on computations and quality
            computations = resource_usage.get("computations", 1000)
            quality = resource_usage.get("quality", 0.5)
            
            # Normalize computations (lower is better)
            normalized_comp = max(0.0, min(1.0, 1.0 - (computations / 10000)))
            
            # Efficiency is quality divided by resource usage
            efficiency = quality * normalized_comp
            
            # Update history and last value
            self.history.append(efficiency)
            self.last_value = efficiency
            
            return (efficiency * 2) - 1  # Scale to [-1, 1]
    
    # Create reward system
    reward_system = create_reward_system()
    
    # Register custom component
    energy_component = EnergySavingComponent(weight=1.2)
    reward_system.register_component(energy_component)
    
    # Create system state with resource usage information
    system_state = create_sample_system_state(quality_level=0.8)
    
    # Add resource usage information
    system_state["resource_usage"] = {
        "computations": 5000,  # Number of operations
        "memory_used": 200,    # MB of memory
        "quality": 0.75        # Quality of result
    }
    
    # Calculate global reward with custom component
    global_reward = reward_system.calculate_global_reward(system_state)
    logger.info(f"Global reward with energy component: {global_reward:.4f}")
    
    # Get component analysis
    component_analysis = reward_system.get_component_analysis()
    
    # Display component contributions
    logger.info("\nComponent Analysis:")
    for component, metrics in component_analysis.items():
        logger.info(f"  {component}: value={metrics['current_value']:.4f}, weight={metrics['weight']}")


def demo_adaptive_training():
    """Demonstrate adaptive training based on reward feedback."""
    logger.info("Demonstrating adaptive training based on rewards...")
    
    # Create reward system
    reward_system = create_reward_system()
    
    # Simulate training progress over time
    num_steps = 20
    learning_rate = 0.1
    quality = 0.3  # Start with low quality
    
    global_rewards = []
    qualities = []
    
    for step in range(num_steps):
        # Create system state with current quality
        system_state = create_sample_system_state(quality_level=quality)
        
        # Calculate reward
        global_reward = reward_system.calculate_global_reward(system_state)
        global_rewards.append(global_reward)
        qualities.append(quality)
        
        logger.info(f"Step {step+1}: Quality = {quality:.4f}, Reward = {global_reward:.4f}")
        
        # Adjust quality based on reward (simulate learning)
        quality_delta = learning_rate * (global_reward + 1) / 2
        quality = min(0.95, quality + quality_delta)
        
        # Wait briefly to see progress
        time.sleep(0.2)
    
    # Plot learning progress
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_steps+1), qualities, 'go-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('System Quality')
    plt.title('System Quality vs. Training Step')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_steps+1), global_rewards, 'ro-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Global Reward')
    plt.title('Global Reward vs. Training Step')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_training.png')
    logger.info("Saved adaptive training plot to 'adaptive_training.png'")
    
    # Get component analysis
    component_analysis = reward_system.get_component_analysis()
    
    # Display component contributions at end of training
    logger.info("\nFinal Component Analysis:")
    for component, metrics in component_analysis.items():
        logger.info(f"  {component}: value={metrics['current_value']:.4f}, improvement={metrics['improvement']:.4f}")


if __name__ == "__main__":
    logger.info("Starting Unified Reward System demonstration...")
    
    # Run demonstrations
    demo_quality_sensitivity()
    demo_training_approaches()
    demo_custom_component()
    demo_adaptive_training()
    
    logger.info("Unified Reward System demonstration completed!") 