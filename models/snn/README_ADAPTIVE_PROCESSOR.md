# Adaptive Spike Processor

The Adaptive Spike Processor extends the standard spike encoding/decoding functionality with learnable parameters and adaptive mechanisms to optimize spike-based information transfer.

## Overview

This module provides a trainable end-to-end system for converting vector representations to spike patterns and back, with improved information retention and noise robustness. The adaptive processor automatically learns to optimize the encoding and decoding processes through gradient-based learning.

## Key Features

- **Learnable Encoding/Decoding**: Neural network layers learn optimal transformations for both encoding and decoding
- **Automatic Adaptation**: Self-adjusts based on input data statistics
- **Multi-modal Support**: Can process text, image, and other data modalities
- **Improved Information Retention**: Preserves more information through the spike conversion process
- **Noise Robustness**: Maintains performance even with noisy input data
- **Performance Optimization**: Provides efficient batch processing and sparse tensor support
- **Adjustable Precision**: Configure the precision-speed tradeoff with precision levels (1-5)

## Usage

### Basic Usage

```python
from models.snn.adaptive_spike_processor import create_adaptive_processor

# Create an adaptive processor
processor = create_adaptive_processor(
    vector_dim=300,
    neuron_count=500,
    encoding_type="temporal",
    precision_level=3
)

# Encode a vector to spikes
vector = torch.rand(300)
spikes = processor.encode(vector)

# Decode spikes back to vector
reconstructed, retention = processor.reconstruct_vector(spikes, vector)
print(f"Information retention: {retention:.4f}")

# Train the processor to improve performance
for i in range(20):
    training_vectors = torch.rand(8, 300)  # Batch of vectors
    metrics = processor.train_step(training_vectors)
    print(f"Loss: {metrics['reconstruction_loss']:.6f}")

# Adapts based on observed statistics
processor.adapt_parameters()

# Save and load the processor
processor.save("./saved_processor")
processor.load("./saved_processor")
```

### Text Processing

```python
from models.snn.bidirectional_encoding import create_processor

# Create text processor
bidirectional_processor = create_processor(model_type="generic", vector_dim=300)

# Create adaptive processor with text capabilities
processor = create_adaptive_processor(
    vector_dim=300,
    neuron_count=500,
    bidirectional_processor=bidirectional_processor
)

# Process text to spikes
text = "Convert this text to spike patterns"
spikes = processor.process_text(text)

# Process with intermediate vector access
spikes, vector = processor.process_text(text, return_vector=True)
```

### Multi-modal Processing

```python
# Create processor with multiple modality support
processor = create_adaptive_processor(
    vector_dim=300,
    neuron_count=500,
    modalities=["text", "image"]
)

# Process image features (e.g., from a CNN)
image_features = torch.rand(512)  # Feature vector from image model
spikes = processor.process_modality(image_features, modality="image")
```

## Implementation Details

The adaptive processor consists of several key components:

1. **AdaptiveEncodingLayer**: Learns optimal transformations before spike encoding
2. **AdaptiveDecodingLayer**: Learns to restore information lost in the encoding/decoding process
3. **ModalityAdapter**: Converts between different data modalities and vector spaces
4. **AdaptiveSpikeProcessor**: Main class that coordinates the encoding/decoding process

The training process optimizes these components to minimize reconstruction loss and maximize information retention.

## Main Classes

- **AdaptiveSpikeProcessor**: End-to-end learnable spike processing system
- **AdaptiveEncodingLayer**: Learnable layer for pre-encoding transformations
- **AdaptiveDecodingLayer**: Learnable layer for post-decoding transformations
- **ModalityAdapter**: Handles different data modalities

## API Reference

### AdaptiveSpikeProcessor

Main class for adaptive spike processing:

- **encode(vector, timesteps)**: Encode vector to spike pattern
- **decode(spike_pattern, target_dim)**: Decode spike pattern to vector
- **reconstruct_vector(spike_pattern, original_vector)**: Reconstruct and measure retention
- **train_step(input_vector)**: Perform a training step
- **adapt_parameters()**: Adjust parameters based on observed statistics
- **process_text(text, timesteps, return_vector)**: Process text input
- **process_modality(input_data, modality, timesteps)**: Process multi-modal inputs
- **save(directory, prefix)**: Save processor state
- **load(directory, prefix)**: Load processor state

### Factory Function

- **create_adaptive_processor()**: Create an AdaptiveSpikeProcessor with standard parameters

## Demo Script

See `adaptive_processor_demo.py` for examples demonstrating:
- Basic usage
- Noise robustness comparison
- Text processing
- Performance benchmarking 