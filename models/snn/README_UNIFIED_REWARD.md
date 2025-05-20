# Unified Reward System for SNN Models

This document provides comprehensive information about the Unified Reward System, a hierarchical framework for coordinating learning across all SNN model components in the Cogmenta Core framework.

## Overview

The Unified Reward System provides a structured approach to defining optimization criteria for the entire SNN ecosystem. It enables coordinated learning across different model types by creating a hierarchy of reward signals that propagate from global system objectives down to individual model-specific goals.

Key features:
- Global system-level objectives
- Component-specific reward signals
- Hierarchical reward propagation
- Support for different training approaches (pretraining, hybrid, absolute zero)
- Integration with both supervised and reinforcement learning

## Architecture

The Unified Reward System follows a modular architecture:

1. **Reward Components**: Individual reward calculators focused on specific aspects of performance
2. **Unified Reward System**: Central coordinator that manages components and propagates rewards
3. **Model Objectives**: Customized reward weights for each model type
4. **Training Approach Manager**: Controls how global vs. model-specific rewards are balanced

### Core Reward Components

The system includes several built-in reward components:

| Component | Purpose | Primary Target Models |
|-----------|---------|----------------------|
| **InformationRetentionComponent** | Measures how well information is preserved across transformations | memory, perceptual, decision |
| **LogicalConsistencyComponent** | Evaluates logical coherence and lack of contradictions | reasoning, decision |
| **NoveltyRelevanceComponent** | Balances novelty of outputs with relevance to inputs | memory, reasoning, metacognitive |
| **SystemIntegrationComponent** | Measures effectiveness of cross-model information transfer | all models |

## Usage

### Basic Usage

```python
from unified_reward_system import create_reward_system

# Create reward system with default components
reward_system = create_reward_system()

# Prepare system state (this would come from your SNN models)
system_state = {
    "input_data": input_data,
    "output_data": output_data,
    "reasoning_output": reasoning_results,
    "model_interactions": interaction_metrics
}

# Calculate global reward
global_reward = reward_system.calculate_global_reward(system_state)

# Calculate model-specific rewards
memory_reward = reward_system.calculate_model_reward("memory", system_state)
reasoning_reward = reward_system.calculate_model_reward("reasoning", system_state)

# Propagate rewards through hierarchy
model_rewards = {
    "memory": memory_reward,
    "reasoning": reasoning_reward,
    # Add other models
}
propagated_rewards = reward_system.propagate_reward(global_reward, model_rewards)
```

### Creating Custom Reward Components

You can extend the system with custom reward components:

```python
from unified_reward_system import RewardComponent

class CustomRewardComponent(RewardComponent):
    def __init__(self, weight=1.0):
        super().__init__(
            name="custom_component",
            weight=weight,
            scaling_factor=1.0,
            target_models=["memory", "reasoning"]
        )
    
    def calculate(self, system_state):
        # Your custom reward calculation logic
        # Should return a value between -1 and 1
        return score

# Register custom component
reward_system.register_component(CustomRewardComponent(weight=1.2))
```

### Training Approaches

The system supports three training approaches that determine how rewards are balanced:

1. **Pretraining Mode**: Focuses on model-specific rewards (80%) with limited global influence (20%)
   ```python
   reward_system.set_training_approach("pretraining", temperature=1.0)
   ```

2. **Hybrid Mode**: Balanced approach with equal weight to global and model-specific rewards
   ```python
   reward_system.set_training_approach("hybrid", temperature=1.0)
   ```

3. **Absolute Zero Mode**: Focuses primarily on global reward (80%) over model-specific rewards (20%)
   ```python
   reward_system.set_training_approach("absolute_zero", temperature=1.0)
   ```

The temperature parameter (0.1-10.0) further adjusts the balance, with higher values increasing the emphasis on global rewards.

## Analysis Tools

The system provides several tools for analyzing reward performance:

```python
# Get detailed component analysis
component_analysis = reward_system.get_component_analysis()

# Get model performance analysis
model_analysis = reward_system.get_model_performance_analysis()

# Get interface health between models
integration_component = reward_system.components["system_integration"]
interface_health = integration_component.get_interface_health_report()
```

## Integration with SNN Models

To integrate the Unified Reward System with your SNN models:

1. Create a shared system state dictionary during model evaluation
2. Add model-specific inputs, outputs, and intermediate results to the state
3. Track interaction metrics between models
4. Call the appropriate reward calculation methods
5. Use the propagated rewards to guide model updates

Example system state structure:
```python
system_state = {
    # Basic input/output
    "input_data": "What is the capital of France?",
    "output_data": "The capital of France is Paris.",
    
    # Vector representations (if available)
    "input_vector": torch.tensor([...]),
    "output_vector": torch.tensor([...]),
    
    # Context for novelty calculation
    "context": {
        "previous_outputs": ["Previous response 1", "Previous response 2"]
    },
    
    # Reasoning component outputs
    "reasoning_output": {
        "logical_structure": {
            "premises": ["France is a country", "Countries have capitals"],
            "conclusions": ["Paris is the capital of France"],
            "support_score": 0.8
        }
    },
    
    # Cross-model interaction metrics
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
```

## Testing

A comprehensive test suite is provided to verify the functionality of the Unified Reward System:

```python
# Run all tests
python -m unittest test_unified_reward_system.py

# Run specific test class
python -m unittest test_unified_reward_system.TestRewardComponents
```

## Demo

A demonstration script is provided to showcase the Unified Reward System's capabilities:

```python
# Run the full demonstration
python unified_reward_demo.py
```

The demo shows:
1. Reward sensitivity to output quality
2. Effects of different training approaches
3. Creating and using custom reward components
4. Adaptive training based on reward feedback

## Conclusion

The Unified Reward System plays a critical role in the Cogmenta Core framework by providing a cohesive learning signal that coordinates across all SNN models. By balancing global system objectives with model-specific goals, it enables effective training of specialized models that work together as an integrated system. 