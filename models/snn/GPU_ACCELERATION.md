# SNN GPU Acceleration

This document describes the GPU acceleration capabilities added to the SNN models in the codebase. Using PyTorch's tensor operations on GPUs, we can significantly speed up both training and inference for all SNN model types.

## Implementation Overview

We've implemented GPU acceleration by:

1. Converting NumPy arrays to PyTorch tensors with device placement
2. Moving tensor operations to the specified device (CPU or GPU)
3. Adding batch processing capabilities for efficient use of GPU resources
4. Implementing memory optimization techniques to handle large models

All core SNN classes now support a `device` parameter that determines where the computations will run.

## Key Components

### 1. EnhancedSpikingCore

The base SNN class now includes:
- Device specification in initialization
- Tensor operations on the specified device
- Memory-efficient operations for large networks

### 2. SNNVectorEngine

The vectorization engine now supports:
- Device-specific tensor operations
- GPU acceleration for embedding lookup and manipulation
- Memory-efficient sparse representation

### 3. Utility Functions

New utility functions in `snn_utils.py`:
- `setup_gpu_acceleration()`: Configure a model for GPU acceleration
- `move_tensors_to_device()`: Move all tensors in an object to the specified device
- `optimize_memory_usage()`: Apply memory optimization techniques
- `batch_process()`: Process inputs in batches for memory efficiency
- `get_gpu_memory_usage()`: Monitor GPU memory consumption

## How to Use

### Basic Usage

```python
from models.snn.statistical_snn import StatisticalSNN
from models.snn.snn_utils import setup_gpu_acceleration

# Create model
model = StatisticalSNN(input_size=100, hidden_size=200, output_size=50)

# Move to GPU
model = setup_gpu_acceleration(model, device='cuda')

# Use the model
result = model.process_input(input_data)
```

### Specify Device at Creation

```python
# Create model directly on GPU
model = StatisticalSNN(
    input_size=100, 
    hidden_size=200, 
    output_size=50,
    device='cuda'
)

# Use the model
result = model.process_input(input_data)
```

### Vector Engine on GPU

```python
from models.snn.snn_vector_engine import SNNVectorEngine

# Create vector engine on GPU
vector_engine = SNNVectorEngine(
    embedding_dim=300,
    vocab_size=10000,
    device='cuda'
)

# Get embeddings
embedding = vector_engine.get_embedding("example text")
```

### Batch Processing

```python
from models.snn.snn_utils import batch_process

# Define processing function
def process_fn(model, input_item):
    return model.process_input(input_item)

# Process batch of inputs
batch_inputs = [input1, input2, input3, ...]
results = batch_process(model, batch_inputs, process_fn, batch_size=32)
```

## Performance Considerations

### Memory Management

Large SNN models can consume significant GPU memory. Use these techniques to optimize memory usage:

1. **Batch Processing**: Process inputs in batches instead of all at once
2. **Memory Cleanup**: Call `torch.cuda.empty_cache()` after large operations
3. **Optimize Batch Size**: Adjust batch size based on model complexity and GPU memory

```python
# Enable memory optimization
model = setup_gpu_acceleration(model, device='cuda', optimize_memory=True, batch_size=32)
```

### Mixed Precision

For further performance improvements, consider using mixed precision (FP16):

```python
# Import torch.cuda.amp for mixed precision
from torch.cuda.amp import autocast

# Use mixed precision for inference
with autocast():
    result = model.process_input(input_data)
```

### When to Use GPU Acceleration

GPU acceleration is most beneficial for:

1. **Large Models**: SNNs with >5,000 neurons
2. **Batch Processing**: When processing multiple inputs simultaneously
3. **Complex Operations**: For models with complex vector operations
4. **Training**: Especially beneficial for training with backpropagation

For small models or single inferences, CPU may still be faster due to the overhead of transferring data to the GPU.

## Demo Script

A demonstration script is provided to benchmark the performance gains:

```bash
python models/snn/demo_gpu_acceleration.py --model all
```

This will run benchmarks on all SNN model types and compare CPU vs. GPU performance.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: 
   - Reduce batch size
   - Call `torch.cuda.empty_cache()` between operations
   - Use `optimize_memory_usage(model)` function

2. **Unexpected Results**:
   - Check tensor device placement with `tensor.device`
   - Ensure all tensors are on the same device

3. **Slow Performance**:
   - Check if small tensors are being moved between CPU and GPU frequently
   - Monitor GPU utilization with `get_gpu_memory_usage()`

## Future Enhancements

Planned enhancements for GPU acceleration:

1. **Custom CUDA Kernels**: For specialized SNN operations
2. **Multi-GPU Support**: Distributing large models across multiple GPUs
3. **Quantization**: Reduced precision for faster inference
4. **JIT Compilation**: For optimized execution paths

## Credits

GPU acceleration implementation by the Cogmenta Core team. 