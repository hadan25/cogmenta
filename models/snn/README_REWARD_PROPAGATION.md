# Reward Propagation Across Paradigms

This document describes the implementation of the Reward Propagation Bridge in the Spiking Neural Network (SNN) framework of the Cogmenta Core system.

## Overview

The Reward Propagation Bridge creates a bidirectional interface between symbolic and neural components, allowing rewards and error signals to propagate across these different paradigms. It translates between symbolic reasoning outcomes (success/failure) and numerical reward signals that can be used for neural learning, as well as converting neural performance metrics into feedback for symbolic components.

## Key Features

- **Bidirectional feedback mechanism** allowing neural and symbolic components to influence each other
- **Explicit reward mappings** for various symbolic reasoning outcomes and neural metrics
- **Iterative refinement process** that enables components to improve each other's outputs
- **Integration with the Unified Reward System** to influence global optimization
- **Customizable weighting** of symbolic vs. neural influences
- **Adaptive confidence adjustment** based on cross-paradigm performance
- **Improvement suggestions** for enhancing integration

## Implementation Components

The implementation consists of two main files:

1. **`reward_propagation_bridge.py`** - Core implementation of the bidirectional bridge with reward mapping functionality
2. **`README_REWARD_PROPAGATION.md`** - Documentation (this file)

## Reward Types

### Symbolic Reward Types

The bridge defines several types of symbolic rewards that can be translated to neural learning signals:

| Reward Type | Description |
|-------------|-------------|
| LOGICAL_CONSISTENCY | Consistency in logical inference |
| THEOREM_PROVING | Success in proving theorems |
| CONSTRAINT_SATISFACTION | Satisfaction of constraints |
| KNOWLEDGE_RETRIEVAL | Success in retrieving knowledge |
| RULE_APPLICATION | Correct application of rules |
| SEMANTIC_COHERENCE | Coherence of symbolic representations |
| CAUSAL_REASONING | Success in causal reasoning |
| CONTRADICTION_DETECTION | Detection of contradictions |

### Neural Reward Types

Neural metrics that can be translated to symbolic feedback:

| Reward Type | Description |
|-------------|-------------|
| INFORMATION_RETENTION | Information preservation in neural processing |
| PREDICTION_ACCURACY | Accuracy of neural predictions |
| RECONSTRUCTION_ERROR | Error in reconstructing inputs (inverse) |
| NOVELTY_DETECTION | Detection of novel patterns |
| UNCERTAINTY_REDUCTION | Reduction in uncertainty |
| SURPRISE_MINIMIZATION | Minimization of surprise (inverse) |
| FEATURE_LEARNING | Success in learning useful features |

## How It Works

### Symbolic to Neural Translation

The bridge translates symbolic reasoning outcomes into numerical rewards for neural learning:

```
Symbolic outcome (success/failure + confidence) → Numerical reward (-1 to 1)
```

Example mapping:
- Successful logical consistency (confidence 0.9) → +0.8 reward
- Failed constraint satisfaction (confidence 0.7) → -0.5 reward

### Neural to Symbolic Translation

Neural performance metrics are converted to feedback for symbolic components:

```
Neural metric (0 to 1) → Symbolic feedback (confidence/rule weight adjustments)
```

Example mapping:
- High information retention (0.9) → +0.1 confidence adjustment
- Low reconstruction error (0.2) → +0.05 rule weight adjustment

### Cross-Paradigm Integration

The bridge implements an iterative process that allows symbolic and neural components to refine each other's outputs:

1. Convert symbolic outcomes to neural rewards
2. Convert neural metrics to symbolic feedback
3. Apply neural feedback to improve symbolic outcomes
4. Apply symbolic rewards to adjust neural metrics
5. Check for convergence and repeat if necessary

This process continues until changes become minimal or a maximum iteration count is reached.

## Integration with Unified Reward System

The bridge registers a custom `SymbolicNeuralIntegrationComponent` with the Unified Reward System, allowing the cross-paradigm feedback mechanism to influence the global optimization of the SNN framework.

## Usage Examples

### Basic Usage

```python
from models.snn.reward_propagation_bridge import create_reward_propagation_bridge
from models.snn.unified_reward_system import create_reward_system

# Create a unified reward system
reward_system = create_reward_system()

# Create a reward propagation bridge
bridge = create_reward_propagation_bridge(unified_reward_system=reward_system)

# Sample symbolic outcomes and neural metrics
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
    }
}

neural_metrics = {
    "information_retention": 0.85,
    "reconstruction_error": 0.25,
    "prediction_accuracy": 0.7
}

# Process cross-paradigm feedback
feedback = bridge.process_cross_paradigm_feedback(
    symbolic_outcomes,
    neural_metrics
)

# Use feedback to improve model training
print(f"Integrated confidence: {feedback['confidence']:.4f}")
if feedback["improvement_suggestions"]:
    print("Improvement suggestions:")
    for suggestion in feedback["improvement_suggestions"]:
        print(f"- {suggestion}")
```

### Customizing Reward Mappings

You can customize the bridge configuration to adjust the relative importance of different reward types:

```python
config = {
    "symbolic_weight": 0.8,  # Increase symbolic influence
    "neural_weight": 0.6,    # Decrease neural influence
    "max_iterations": 5,     # Increase maximum integration iterations
    "feedback_threshold": 0.7  # Higher threshold for feedback
}

bridge = create_reward_propagation_bridge(
    unified_reward_system=reward_system,
    config=config
)
```

### Integration with Neural-Symbolic Pipeline

The bridge is designed to be integrated into the neural-symbolic pipeline:

```python
# In a training loop
for epoch in range(epochs):
    # Train neural models and collect metrics
    neural_metrics = train_neural_models(data)
    
    # Perform symbolic reasoning and collect outcomes
    symbolic_outcomes = perform_symbolic_reasoning(data)
    
    # Process cross-paradigm feedback
    feedback = bridge.process_cross_paradigm_feedback(
        symbolic_outcomes,
        neural_metrics
    )
    
    # Use feedback to adjust training parameters
    symbolic_confidence = feedback["confidence"]
    improved_symbolic = feedback["final_symbolic_state"]
    improved_neural = feedback["final_neural_state"]
    
    # Apply improvements to the next training iteration
    adjust_training_parameters(
        symbolic_confidence=symbolic_confidence,
        improved_symbolic=improved_symbolic,
        improved_neural=improved_neural
    )
```

## Diagrams

### Reward Propagation Flow

```
┌─────────────────┐        ┌───────────────────┐
│                 │        │                   │
│    Symbolic     │        │     Neural        │
│    Components   │◄───────┤     Components    │
│                 │        │                   │
└────────┬────────┘        └─────────┬─────────┘
         │                           │
         ▼                           ▼
┌────────────────┐         ┌──────────────────┐
│   Symbolic     │         │     Neural       │
│   Outcomes     │         │     Metrics      │
└────────┬───────┘         └────────┬─────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────┐
│                                              │
│           Reward Propagation Bridge          │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  ● Symbolic to Neural Translation            │
│  ● Neural to Symbolic Translation            │
│  ● Cross-Paradigm Integration                │
│  ● Iterative Refinement                      │
│                                              │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                                              │
│             Unified Reward System            │
│                                              │
└──────────────────────────────────────────────┘
```

## Benefits of the Approach

1. **Improved Neural Learning**: Neural components benefit from symbolic reasoning outcomes, gaining more structured feedback beyond simple error signals.

2. **Enhanced Symbolic Reasoning**: Symbolic components can adapt their confidence and rule weights based on neural performance metrics.

3. **Cross-Paradigm Synergy**: The iterative refinement process allows the strengths of each paradigm to compensate for the weaknesses of the other.

4. **Unified Optimization**: Integration with the Unified Reward System ensures that cross-paradigm interactions contribute to global optimization.

5. **Explainable Adjustments**: The bridge provides explicit mappings and adjustments, making it possible to understand and debug cross-paradigm interactions.

## Conclusion

The Reward Propagation Bridge is a critical component that enables effective integration between symbolic and neural paradigms in the SNN framework. By providing a bidirectional translation mechanism for rewards and feedback, it allows these different approaches to reasoning and learning to complement and enhance each other, leading to a more robust and capable integrated system. 