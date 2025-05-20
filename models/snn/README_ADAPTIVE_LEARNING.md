# Explicit Learning Protocol for Adaptive SNN Components

This document describes the Explicit Learning Protocol implementation for adaptive components in the Spiking Neural Network (SNN) models of the Cogmenta Core framework.

## Overview

The Explicit Learning Protocol provides a structured approach to training adaptive neural components in the SNN framework. It defines clear loss functions, optimization strategies, and mechanisms for controlling adaptiveness during different phases of training. This protocol ensures that adaptive layers are properly included in optimization and can be selectively frozen or unfrozen based on performance metrics.

## Key Features

- **Well-defined loss functions** for adaptive layers (MSE, cosine similarity, information retention)
- **Explicit learning phases** (pretraining, joint training, fine-tuning, continuous adaptation)
- **Automatic parameter update control** based on performance metrics
- **Phase-specific optimization strategies** with appropriate learning rates
- **Freezing/unfreezing mechanism** for adaptive components
- **Performance tracking** and training statistics

## Implementation Components

The learning protocol implementation consists of two key components:

1. **AdaptiveLearningProtocol Class** (`adaptive_learning_protocol.py`): Core implementation of the learning protocol with methods for loss calculation, parameter update decisions, and learning phase management.

2. **Integration with AdaptiveSpikeProcessor** (`adaptive_spike_processor.py`): Integration of the learning protocol with the adaptive processor to control training behavior and optimize adaptive components.

## Learning Phases

The protocol supports the following learning phases:

| Phase | Description | Update Behavior |
|-------|-------------|-----------------|
| PRETRAINING | Initial training phase | Always update parameters |
| JOINT | Training with other components | Update only if performance below threshold |
| FINE_TUNING | Targeted optimization for specific tasks | Always update parameters |
| CONTINUOUS | Ongoing adaptation during operation | Probabilistic updates based on performance |
| DISABLED | Adaptiveness turned off | Never update parameters |

## Loss Functions

The protocol provides several loss functions for training adaptive components:

### MSE Loss
Mean squared error between original and reconstructed vectors:
```python
def _mse_loss(self, original, reconstructed):
    return torch.nn.functional.mse_loss(reconstructed, original)
```

### Cosine Similarity Loss
Negative cosine similarity (higher similarity = lower loss):
```python
def _cosine_loss(self, original, reconstructed):
    cos_sim = torch.nn.functional.cosine_similarity(
        reconstructed, original
    ).mean()
    return 1.0 - cos_sim
```

### Information Retention Loss
Loss function that penalizes poor information retention:
```python
def _information_retention_loss(self, original, reconstructed, retention_target=0.99):
    cos_loss = self._cosine_loss(original, reconstructed)
    retention = 1.0 - cos_loss.item()
    retention_gap = max(0, retention_target - retention)
    retention_penalty = torch.tensor(retention_gap ** 2, device=original.device)
    return cos_loss + retention_penalty
```

### Combined Loss
Weighted combination of multiple loss functions:
```python
def _combined_loss(self, original, reconstructed):
    mse = self._mse_loss(original, reconstructed)
    cosine = self._cosine_loss(original, reconstructed)
    info_loss = self._information_retention_loss(original, reconstructed)
    
    return (
        self.loss_weights.get("mse", 1.0) * mse +
        self.loss_weights.get("cosine", 0.5) * cosine +
        self.loss_weights.get("info", 1.0) * info_loss
    )
```

## Parameter Update Control

The protocol includes a mechanism to control when parameters are updated based on performance metrics and the current learning phase:

```python
def should_update_parameters(self, performance_metric: float) -> bool:
    # Always update during pretraining or fine-tuning
    if self.learning_phase in [LearningPhase.PRETRAINING, LearningPhase.FINE_TUNING]:
        return True
    
    # In joint training, update only if performance is below threshold
    elif self.learning_phase == LearningPhase.JOINT:
        return performance_metric < self.adaptation_threshold
    
    # In continuous mode, use adaptive strategy based on performance
    elif self.learning_phase == LearningPhase.CONTINUOUS:
        # Update more frequently when performance is lower
        update_probability = max(0, 1.0 - performance_metric)
        return np.random.random() < update_probability
    
    # Don't update if disabled
    else:
        return False
```

## Usage

### Basic Usage

To use the learning protocol with an adaptive spike processor:

```python
from models.snn.adaptive_learning_protocol import create_learning_protocol
from models.snn.adaptive_spike_processor import create_adaptive_processor

# Create a learning protocol
protocol = create_learning_protocol(
    learning_rate=0.001,
    loss_function="combined",
    initial_phase="pretraining",
    adaptation_threshold=0.95
)

# Create an adaptive processor with the protocol
processor = create_adaptive_processor(
    vector_dim=300,
    neuron_count=500,
    learning_protocol=protocol
)

# Train the processor
for i in range(100):
    # Generate or load training data
    data = torch.rand(10, 300)
    
    # Perform training step
    metrics = processor.train_step(data)
    
    # Optionally change learning phase during training
    if i == 50:
        protocol.set_learning_phase("joint")
```

### Changing Learning Phase

To change the learning phase during training:

```python
# Change to joint training phase
protocol.set_learning_phase("joint")

# Or use the enum directly
from models.snn.adaptive_learning_protocol import LearningPhase
protocol.set_learning_phase(LearningPhase.FINE_TUNING)
```

### Customizing Loss Function

To customize the loss function:

```python
# Create protocol with custom loss weights
protocol = create_learning_protocol(
    loss_function="combined",
    loss_weights={
        "mse": 0.5,       # Reduce MSE weight
        "cosine": 1.0,    # Increase cosine similarity weight
        "info": 2.0       # Double information retention weight
    }
)
```

## Monitoring Training

The protocol provides methods to monitor training progress:

```python
# Get training summary
summary = protocol.get_training_summary()
print(f"Iterations: {summary['iterations']}")
print(f"Current phase: {summary['current_phase']}")
print(f"Average loss: {summary['avg_loss']:.4f}")
print(f"Max performance: {summary['max_performance']:.4f}")
```

## Integration with Training Pipeline

The learning protocol integrates with the SNN training pipeline to control adaptiveness during different training stages:

1. **Pretraining Phase**: Uses high learning rate and always updates parameters to quickly adapt to basic patterns in data
2. **Joint Training Phase**: More conservative updates based on performance threshold to prevent overfitting to other components
3. **Fine-tuning Phase**: Focused optimization for specific tasks with lower learning rate
4. **Continuous Mode**: Ongoing adaptation during model deployment with very conservative updates

## Conclusion

The Explicit Learning Protocol for Adaptiveness provides a structured approach to training adaptive components in the SNN framework. By defining clear loss functions, optimization strategies, and update mechanisms, it ensures that adaptive layers contribute effectively to model performance while preventing overfitting or instability. This protocol is a key component of the adaptive encoding/decoding system that enables efficient information transfer through spike-based representations. 