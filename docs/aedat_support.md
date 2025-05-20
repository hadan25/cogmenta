# AEDAT Support for Neuromorphic Camera Data

This document describes the support for AEDAT (Address-Event Data) format in the Cogmenta Core framework. AEDAT is a file format used by neuromorphic cameras such as Dynamic Vision Sensors (DVS), DAVIS cameras, and other event-based vision sensors.

## Overview

Event-based cameras capture visual information in a fundamentally different way than conventional frame-based cameras. Instead of capturing frames at fixed time intervals, they detect pixel-level brightness changes (events) asynchronously. This results in a stream of events, each containing:

- x, y: Spatial coordinates of the event
- t: Timestamp of the event
- p: Polarity (1 for increase in brightness, 0 for decrease)

The AEDAT format stores these events efficiently, making it ideal for neuromorphic computing applications.

## Supported AEDAT Versions

The framework supports the following AEDAT versions:

- **AEDAT3**: Used by DVS128 and DAVIS cameras (including CIFAR10-DVS dataset)
- **AEDAT4**: Used by newer neuromorphic cameras

## CIFAR10-DVS Dataset

The framework has been tested with the CIFAR10-DVS dataset, which contains neuromorphic camera recordings of the CIFAR-10 dataset. The dataset consists of 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each class contains 1000 AEDAT recordings, for a total of 10,000 samples.

## Implementation Details

### Event-to-Spike Conversion

The framework provides two main classes for handling event-based data:

1. `EventToSpikeConverter`: Base class for converting event data to spike representations
2. `AEDATEventToSpikeConverter`: Specialized converter for AEDAT format data

The conversion process involves:

1. Parsing the AEDAT file to extract events (x, y, t, p)
2. Mapping spatial coordinates (x, y) to neuron indices
3. Normalizing timestamps
4. Applying polarity-specific encoding (optional)

### Binary AEDAT3 Parsing

For CIFAR10-DVS dataset, which uses AEDAT3 format, we implemented a custom binary parser that:

1. Finds the start of the data section in the file
2. Parses each 8-byte event:
   - First 4 bytes: address (contains x, y, polarity)
   - Last 4 bytes: timestamp
3. Extracts x, y coordinates and polarity using bit masks
4. Handles large timestamps using unsigned integers

### Polarity-Based Encoding

The framework supports polarity-based encoding, where ON and OFF events are mapped to different neuron populations. This preserves the important distinction between brightness increases and decreases.

## Usage Examples

### Loading and Converting AEDAT Files

```python
from training.utils.data_to_spike import AEDATEventToSpikeConverter

# Create converter
converter = AEDATEventToSpikeConverter(
    neuron_count=1000,
    spatial_dimensions=(128, 128)  # CIFAR10-DVS uses 128x128 DVS
)

# Convert AEDAT file to spike representation
spike_data = converter.convert("path/to/file.aedat")
```

### Using the EventBasedDataset

```python
from training.utils.dataset_loader import EventBasedDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = EventBasedDataset("training/datasets/cifar10_dvs", max_events=5000)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in dataloader:
    samples = batch['sample']
    labels = batch['label']
    # Process batch...
```

### Feature Extraction

The framework includes utilities for extracting features from event-based data:

```python
# Extract features directly from events
features = extract_event_features(events)
```

Features include:
- Event count and distribution
- Temporal statistics
- Spatial distribution
- Polarity ratio

## Example Scripts

The framework includes several example scripts for working with AEDAT data:

- `examples/cifar10_dvs_example.py`: Demonstrates loading and visualizing CIFAR10-DVS data
- `examples/cifar10_dvs_classifier.py`: Trains classifiers on features extracted from CIFAR10-DVS

## Performance

Initial experiments with the CIFAR10-DVS dataset show promising results. Using features extracted from the event data, classifiers can achieve reasonable accuracy even with limited training samples. 