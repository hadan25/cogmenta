# Quantitative Fidelity Gate for SNN Models

This document describes the Quantitative Fidelity Gate system implemented for the Spiking Neural Network (SNN) models in the Cogmenta Core framework.

## Overview

The Quantitative Fidelity Gate is a quality control checkpoint that ensures information preservation across encode-decode-re-encode cycles for all supported modalities. It serves as a critical validation step before proceeding to model training, ensuring that only models meeting strict fidelity thresholds are allowed to continue in the pipeline.

## Key Features

- **Explicit measurement** of information preservation across encode-decode-re-encode cycles
- **Hard numerical targets** for reconstruction error, cosine similarity, and semantic preservation
- **Comprehensive test suite** for round-trip fidelity across all modalities (text, vector, etc.)
- **Strict pass/fail thresholds** with pipeline enforcement
- **Detailed reporting** of fidelity metrics for model analysis
- **Integration with training pipeline** to automate quality control

## Implementation Components

The fidelity gate system consists of several key components:

1. **FidelityGate Class** (`fidelity_gate.py`): Core implementation of the fidelity gate with methods for model evaluation, testing, and reporting.

2. **FidelityMetrics Class** (`fidelity_gate.py`): Container for collecting and analyzing fidelity metrics across different modalities and models.

3. **Test Suite** (`test_fidelity_gate.py`): Comprehensive test suite for verifying the fidelity gate functionality.

4. **CLI Tool** (`fidelity_gate_cli.py`): Command-line interface for running the fidelity gate as part of a pipeline.

5. **Pipeline Integration** (`snn_pipeline.py`): Integration with the SNN training pipeline to enforce fidelity requirements.

## Metrics and Thresholds

The fidelity gate evaluates models using several key metrics:

### Vector Modality

| Metric | Description | Default Threshold |
|--------|-------------|------------------|
| Cosine Similarity | Similarity between original and reconstructed vectors | ≥ 0.999 |
| MSE | Mean squared error between original and reconstructed vectors | ≤ 0.001 |
| Retention Score | Information retention score from the adaptive processor | ≥ 0.99 |

### Text Modality

| Metric | Description | Default Threshold |
|--------|-------------|------------------|
| Cosine Similarity | Similarity between original and reconstructed vectors | ≥ 0.95 |
| Semantic Similarity | Semantic similarity between original and reconstructed text | ≥ 0.90 |
| Retention Score | Information retention score from the adaptive processor | ≥ 0.90 |

### Other Modalities

The system is designed to be extensible to other modalities (image, audio, etc.) as they are implemented.

## Usage

### Basic Usage

To run the fidelity gate on a model:

```python
from models.snn.fidelity_gate import create_fidelity_gate
from models.snn.adaptive_spike_processor import create_adaptive_processor
from models.snn.bidirectional_encoding import create_processor

# Create processors
bidirectional_processor = create_processor(model_type="generic", vector_dim=300)
processor = create_adaptive_processor(
    vector_dim=300,
    neuron_count=500,
    encoding_type="temporal",
    precision_level=3,
    timesteps=10,
    bidirectional_processor=bidirectional_processor
)

# Create fidelity gate
gate = create_fidelity_gate()

# Evaluate the processor
passed, metrics = gate.evaluate_model(
    processor, 
    bidirectional_processor=bidirectional_processor,
    model_name="my_model"
)

# Print the results
print(metrics.to_report())
print(f"Model passed fidelity gate: {passed}")
```

### Command-Line Usage

To run the fidelity gate from the command line:

```bash
# Evaluate a single model
python fidelity_gate_cli.py --model-config model_config.json --output-dir ./reports

# Evaluate all models in a directory
python fidelity_gate_cli.py --model-dir ./models --output-dir ./reports

# Using custom thresholds
python fidelity_gate_cli.py --model-config model_config.json --thresholds custom_thresholds.json
```

### Pipeline Integration

To run the complete SNN pipeline with fidelity gate integration:

```bash
# Run the full pipeline
python snn_pipeline.py --config pipeline_config.json

# Skip the fidelity gate (for development/testing)
python snn_pipeline.py --config pipeline_config.json --skip-fidelity
```

## Configuration Options

### Fidelity Gate Configuration

Fidelity gate configuration in `pipeline_config.json`:

```json
"fidelity_gate": {
  "modalities": ["text", "vector"],
  "fail_action": "exit",
  "thresholds": {
    "text": {
      "cosine_similarity": 0.95,
      "semantic_similarity": 0.90,
      "retention_score": 0.90
    },
    "vector": {
      "cosine_similarity": 0.999,
      "mse": 0.001,
      "retention_score": 0.99
    }
  }
}
```

### Failure Actions

The system supports the following actions when a model fails to meet the thresholds:

- **exit**: Exit the pipeline immediately (default)
- **warn**: Continue the pipeline but issue a warning
- **continue**: Continue silently

## Round-Trip Fidelity Testing

The fidelity gate tests round-trip fidelity through multiple steps:

1. **Encode**: Convert input data to spike patterns
2. **Decode**: Convert spike patterns back to vectors
3. **Re-encode**: Encode the reconstructed vectors again
4. **Compare**: Measure similarity between original, reconstruction, and re-encoded reconstruction

This process ensures that information is preserved through multiple encoding-decoding cycles, which is critical for maintaining data integrity in a complex SNN system.

## Integration with Training

When integrated with the training pipeline, the fidelity gate ensures that:

1. Only models meeting fidelity thresholds proceed to training
2. Re-evaluation occurs after training to verify that fidelity is maintained or improved
3. Training approaches are adapted based on fidelity results

## Extending the Fidelity Gate

### Adding New Modalities

To add support for a new modality:

1. Update the `_setup_test_data` method in `FidelityGate` class to include test data for the new modality
2. Implement a corresponding test method (e.g., `_test_image_fidelity`)
3. Update the metrics and thresholds for the new modality

### Adding New Metrics

To add a new metric:

1. Update the relevant test method to calculate and store the new metric
2. Update the default thresholds in the `FidelityGate` constructor
3. Consider updating the reporting format in `FidelityMetrics`

## Troubleshooting

Common issues and solutions:

### Model Failing Fidelity Gate

If a model fails the fidelity gate:

1. Check the fidelity report to identify which metrics are failing
2. Increase the precision level of the encoder/decoder
3. Increase the neuron count to capture more information
4. Adjust the adaptive processor parameters
5. For debugging, try temporarily lowering the thresholds

### Performance Issues

If the fidelity gate is taking too long to run:

1. Reduce the number of test samples
2. Use a smaller vector dimension for testing
3. Run only specific modalities with the `--modalities` option

## Conclusion

The Quantitative Fidelity Gate is a critical component in ensuring the quality and reliability of SNN models. By enforcing strict standards for information preservation, it helps prevent the propagation of information loss through the system and ensures that only high-quality models proceed to training. 