#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for SNN GPU Acceleration

This script demonstrates how to use GPU acceleration with the different SNN models,
including performance benchmarking and memory usage optimization.
"""

import os
import sys
import time
import numpy as np
import torch
from argparse import ArgumentParser
import logging

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import SNN utilities and models
from models.snn.snn_utils import setup_gpu_acceleration, optimize_memory_usage, get_gpu_memory_usage, batch_process
from models.snn.snn_vector_engine import SNNVectorEngine
from models.snn.statistical_snn import StatisticalSNN
from models.snn.memory_snn import MemorySNN
from models.snn.metacognitive_snn import MetacognitiveSNN
from models.snn.decision_snn import DecisionSNN
from models.snn.reasoning_snn import ReasoningSNN
from models.snn.affective_snn import AffectiveSNN
from models.snn.perceptual_snn import PerceptualSNN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SNN-GPU-Demo")

def benchmark_model(model, input_data, iterations=10, batch_size=None):
    """
    Benchmark model performance with and without GPU acceleration.
    
    Args:
        model: The SNN model to benchmark
        input_data: Input data for the model
        iterations: Number of iterations for benchmarking
        batch_size: Optional batch size for batch processing
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Benchmark CPU performance
    logger.info("Benchmarking on CPU...")
    model_cpu = setup_gpu_acceleration(model, device='cpu')
    
    # Warmup
    for _ in range(3):
        if hasattr(model_cpu, 'process'):
            model_cpu.process(input_data)
        elif hasattr(model_cpu, 'process_input'):
            model_cpu.process_input(input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        if hasattr(model_cpu, 'process'):
            model_cpu.process(input_data)
        elif hasattr(model_cpu, 'process_input'):
            model_cpu.process_input(input_data)
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    results['cpu_time_per_iteration'] = cpu_time / iterations
    logger.info(f"CPU time: {cpu_time:.4f}s ({results['cpu_time_per_iteration']:.4f}s per iteration)")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        # Benchmark GPU performance
        logger.info("Benchmarking on GPU...")
        model_gpu = setup_gpu_acceleration(model, device='cuda', optimize_memory=True, batch_size=batch_size)
        
        # Warmup
        for _ in range(3):
            if hasattr(model_gpu, 'process'):
                model_gpu.process(input_data)
            elif hasattr(model_gpu, 'process_input'):
                model_gpu.process_input(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            if hasattr(model_gpu, 'process'):
                model_gpu.process(input_data)
            elif hasattr(model_gpu, 'process_input'):
                model_gpu.process_input(input_data)
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['gpu_time_per_iteration'] = gpu_time / iterations
        results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        logger.info(f"GPU time: {gpu_time:.4f}s ({results['gpu_time_per_iteration']:.4f}s per iteration)")
        logger.info(f"GPU speedup: {results['speedup']:.2f}x")
        
        # Get GPU memory usage
        memory_usage = get_gpu_memory_usage()
        results['gpu_memory_usage'] = memory_usage
        logger.info(f"GPU memory usage: {memory_usage}")
    else:
        logger.warning("GPU not available, skipping GPU benchmarks")
        results['gpu_available'] = False
    
    return results

def demo_statistical_snn():
    """Demonstrate GPU acceleration with Statistical SNN"""
    logger.info("\n=== Statistical SNN GPU Acceleration Demo ===")
    
    # Create model
    model = StatisticalSNN(input_size=100, hidden_size=200, output_size=50)
    
    # Create sample input
    input_data = np.random.random(100)
    
    # Benchmark
    results = benchmark_model(model, input_data, iterations=20)
    
    # Demo batch processing
    if torch.cuda.is_available():
        logger.info("\nDemonstrating batch processing...")
        model_gpu = setup_gpu_acceleration(model, device='cuda', optimize_memory=True, batch_size=32)
        
        # Create batch of inputs
        batch_inputs = [np.random.random(100) for _ in range(100)]
        
        # Define processing function
        def process_fn(model, input_item):
            return model.process_input(input_item)
        
        # Process batch
        start_time = time.time()
        batch_results = batch_process(model_gpu, batch_inputs, process_fn)
        batch_time = time.time() - start_time
        
        logger.info(f"Batch processing time for 100 inputs: {batch_time:.4f}s ({batch_time/100:.4f}s per input)")
    
    return results

def demo_memory_snn():
    """Demonstrate GPU acceleration with Memory SNN"""
    logger.info("\n=== Memory SNN GPU Acceleration Demo ===")
    
    # Create model
    model = MemorySNN()
    
    # Create sample memory item
    memory_item = {
        "content": "Sample memory content",
        "timestamp": time.time(),
        "importance": 0.8,
        "tags": ["sample", "demo"]
    }
    
    # Store some items
    for i in range(10):
        item = memory_item.copy()
        item["content"] = f"Memory item {i}"
        model.store(item)
    
    # Create sample query
    query = {"tags": ["sample"]}
    
    # Benchmark retrieval
    logger.info("Benchmarking memory retrieval...")
    results = benchmark_model(model, query, iterations=20)
    
    return results

def demo_vector_engine():
    """Demonstrate GPU acceleration with SNNVectorEngine"""
    logger.info("\n=== SNNVectorEngine GPU Acceleration Demo ===")
    
    # Create vector engine
    vector_engine_cpu = SNNVectorEngine(embedding_dim=300, vocab_size=10000, device='cpu')
    
    # Add some word embeddings
    for i in range(1000):
        word = f"word_{i}"
        vector_engine_cpu.word_to_idx[word] = i + 5  # Start after special tokens
        vector_engine_cpu.idx_to_word[i + 5] = word
        vector_engine_cpu.word_embeddings[word] = vector_engine_cpu._create_random_embedding()
    
    # Benchmark CPU performance
    logger.info("Benchmarking on CPU...")
    sample_words = [f"word_{i}" for i in range(0, 1000, 50)]  # Sample every 50th word
    
    start_time = time.time()
    for _ in range(20):
        for word in sample_words:
            vector_engine_cpu.get_embedding(word)
    cpu_time = time.time() - start_time
    logger.info(f"CPU time for 20 iterations of 20 words: {cpu_time:.4f}s")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        # Create GPU vector engine
        vector_engine_gpu = SNNVectorEngine(embedding_dim=300, vocab_size=10000, device='cuda')
        
        # Add same word embeddings
        for i in range(1000):
            word = f"word_{i}"
            vector_engine_gpu.word_to_idx[word] = i + 5
            vector_engine_gpu.idx_to_word[i + 5] = word
            vector_engine_gpu.word_embeddings[word] = vector_engine_gpu._create_random_embedding()
        
        # Benchmark GPU performance
        logger.info("Benchmarking on GPU...")
        start_time = time.time()
        for _ in range(20):
            for word in sample_words:
                vector_engine_gpu.get_embedding(word)
        gpu_time = time.time() - start_time
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        logger.info(f"GPU time for 20 iterations of 20 words: {gpu_time:.4f}s")
        logger.info(f"GPU speedup: {speedup:.2f}x")
        
        # Get GPU memory usage
        memory_usage = get_gpu_memory_usage()
        logger.info(f"GPU memory usage: {memory_usage}")
        
        # Return results
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'gpu_memory_usage': memory_usage
        }
    else:
        logger.warning("GPU not available, skipping GPU benchmarks")
        return {'cpu_time': cpu_time, 'gpu_available': False}

def run_comprehensive_benchmark(batch_size=64):
    """Run benchmarks on all SNN model types"""
    logger.info("\n=== Comprehensive SNN Benchmarks ===")
    
    results = {}
    
    # Statistical SNN
    results['statistical'] = demo_statistical_snn()
    
    # Memory SNN
    results['memory'] = demo_memory_snn()
    
    # Vector Engine
    results['vector_engine'] = demo_vector_engine()
    
    # Summary of speedups
    if torch.cuda.is_available():
        logger.info("\n=== GPU Acceleration Summary ===")
        for model_type, result in results.items():
            if 'speedup' in result:
                logger.info(f"{model_type.capitalize()}: {result['speedup']:.2f}x speedup")
    
    return results

def main():
    """Main entry point"""
    parser = ArgumentParser(description="SNN GPU Acceleration Demo")
    parser.add_argument("--model", choices=['statistical', 'memory', 'vector_engine', 'all'], 
                      default='all', help="Model to benchmark")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations for benchmarking")
    
    args = parser.parse_args()
    
    # Print system info
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available")
    
    # Run selected benchmark
    if args.model == 'statistical':
        demo_statistical_snn()
    elif args.model == 'memory':
        demo_memory_snn()
    elif args.model == 'vector_engine':
        demo_vector_engine()
    else:
        run_comprehensive_benchmark(batch_size=args.batch_size)
    
    logger.info("Demo completed!")

if __name__ == "__main__":
    main() 