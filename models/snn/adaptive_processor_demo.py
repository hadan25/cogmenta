#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of the Adaptive Spike Processor.

This script shows how to use the adaptive spike processor for
different tasks and modalities, highlighting its advantages over
standard encoding/decoding.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from models.snn.adaptive_spike_processor import AdaptiveSpikeProcessor, create_adaptive_processor
from models.snn.spike_encoder import SpikeEncoder, create_encoder
from models.snn.spike_decoder import SpikeDecoder, create_decoder
from models.snn.bidirectional_encoding import BidirectionalProcessor, create_processor

def demo_basic_usage():
    """Demonstrate basic usage of the adaptive spike processor"""
    print("\n=== Basic Usage Demo ===")
    
    # Create an adaptive processor
    processor = create_adaptive_processor(
        vector_dim=300,
        neuron_count=500,
        encoding_type="temporal",
        timesteps=15
    )
    
    # Create a sample vector
    test_vector = torch.rand(300)
    print(f"Created test vector with shape: {test_vector.shape}")
    
    # Initial encoding/decoding test
    print("\nTesting initial reconstruction (before training)...")
    spikes = processor.encode(test_vector)
    reconstructed, retention = processor.reconstruct_vector(spikes, test_vector)
    
    print(f"Spike pattern shape: {spikes.shape}")
    print(f"Reconstructed vector shape: {reconstructed.shape}")
    print(f"Initial information retention: {retention:.4f}")
    
    # Train the processor
    print("\nTraining adaptive processor...")
    for i in range(20):
        # Generate random training vectors
        training_vectors = torch.rand(8, 300)  # Batch of 8 vectors
        
        # Perform training step
        metrics = processor.train_step(training_vectors)
        
        if (i + 1) % 5 == 0:
            print(f"Step {i+1}: Loss = {metrics['reconstruction_loss']:.6f}, "
                  f"Similarity = {metrics['cosine_similarity']:.4f}")
    
    # Test again after training
    print("\nTesting reconstruction after training...")
    spikes = processor.encode(test_vector)
    reconstructed, retention = processor.reconstruct_vector(spikes, test_vector)
    print(f"Information retention after training: {retention:.4f}")

def demo_noise_robustness():
    """Demonstrate robustness to noise"""
    print("\n=== Noise Robustness Demo ===")
    
    # Create standard encoder/decoder
    encoder = create_encoder(encoding_type="temporal", neuron_count=500)
    decoder = create_decoder(decoding_type="temporal", neuron_count=500)
    
    # Create adaptive processor
    processor = create_adaptive_processor(
        vector_dim=300,
        neuron_count=500,
        encoding_type="temporal"
    )
    
    # Train the adaptive processor
    print("Training adaptive processor...")
    for i in range(20):
        processor.train_step(torch.rand(8, 300))
    
    # Create test vector
    test_vector = torch.rand(300)
    
    # Test with different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    standard_retention = []
    adaptive_retention = []
    
    print("\nTesting noise robustness:")
    for noise in noise_levels:
        # Add noise to vector
        noisy_vector = test_vector + torch.randn_like(test_vector) * noise
        noisy_vector = torch.clamp(noisy_vector, 0, 1)
        
        # Standard processing
        spikes_std = encoder.encode_vector(noisy_vector, 15)
        decoded_std = decoder.decode_spikes(spikes_std, 300)
        
        # Calculate similarity
        cos_std = torch.nn.functional.cosine_similarity(
            test_vector.view(1, -1), decoded_std.view(1, -1)
        ).item()
        standard_retention.append(cos_std)
        
        # Adaptive processing
        spikes_adp = processor.encode(noisy_vector)
        decoded_adp, _ = processor.reconstruct_vector(spikes_adp, test_vector)
        
        # Calculate similarity
        cos_adp = torch.nn.functional.cosine_similarity(
            test_vector.view(1, -1), decoded_adp.view(1, -1)
        ).item()
        adaptive_retention.append(cos_adp)
        
        print(f"Noise level {noise:.1f} - Standard: {cos_std:.4f}, Adaptive: {cos_adp:.4f}")
    
    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, standard_retention, 'o-', label="Standard")
        plt.plot(noise_levels, adaptive_retention, 's-', label="Adaptive")
        plt.xlabel("Noise Level")
        plt.ylabel("Information Retention")
        plt.title("Noise Robustness Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("noise_robustness.png")
        print("\nPlot saved to noise_robustness.png")
    except:
        print("Could not create plot (matplotlib may not be available)")

def demo_text_processing():
    """Demonstrate text processing capabilities"""
    print("\n=== Text Processing Demo ===")
    
    # Create bidirectional processor for text
    bidirectional_processor = create_processor(
        model_type="generic",
        vector_dim=300
    )
    
    # Create adaptive processor with bidirectional processor
    processor = create_adaptive_processor(
        vector_dim=300,
        neuron_count=500,
        bidirectional_processor=bidirectional_processor
    )
    
    # Sample text
    test_text = "The adaptive spike processor demonstrates learning capabilities for improved information transfer."
    print(f"Test text: \"{test_text}\"")
    
    # Process text
    spikes, vector = processor.process_text(test_text, return_vector=True)
    
    print(f"Text converted to vector shape: {vector.shape}")
    print(f"Text converted to spike pattern shape: {spikes.shape}")
    
    # Train the processor with fresh training data instead of reusing vector
    print("\nTraining adaptive processor...")
    for i in range(10):
        # Generate random training vectors of the same shape as the text vectors
        random_vectors = torch.rand(vector.shape)
        processor.train_step(random_vectors)
    
    # Re-process the text
    spikes_after, _ = processor.process_text(test_text, return_vector=True)
    
    # Compare sparsity before and after training
    if hasattr(spikes, "is_sparse") and spikes.is_sparse:
        before_nnz = spikes._nnz()
        after_nnz = spikes_after._nnz()
    else:
        before_nnz = torch.sum(spikes > 0).item()
        after_nnz = torch.sum(spikes_after > 0).item()
    
    print(f"Spike sparsity before training: {before_nnz / spikes.numel():.4f}")
    print(f"Spike sparsity after training: {after_nnz / spikes_after.numel():.4f}")

def demo_performance_benchmark():
    """Benchmark performance of adaptive vs standard encoding/decoding"""
    print("\n=== Performance Benchmark ===")
    
    # Parameters
    vector_dim = 300
    neuron_count = 1000
    batch_size = 32
    timesteps = 20
    
    # Create standard encoder/decoder
    encoder = create_encoder(
        encoding_type="temporal",
        neuron_count=neuron_count,
        precision_level=3
    )
    
    decoder = create_decoder(
        decoding_type="temporal",
        neuron_count=neuron_count,
        precision_level=3
    )
    
    # Create adaptive processor
    processor = create_adaptive_processor(
        vector_dim=vector_dim,
        neuron_count=neuron_count,
        encoding_type="temporal",
        precision_level=3
    )
    
    # Train adaptive processor
    print("Training adaptive processor...")
    for i in range(10):
        processor.train_step(torch.rand(batch_size, vector_dim))
    
    # Create test batch
    test_batch = torch.rand(batch_size, vector_dim)
    
    # Benchmark standard encoding/decoding
    print("\nBenchmarking standard encoding/decoding...")
    start_time = time.time()
    
    # Forward pass
    spike_patterns = []
    for i in range(batch_size):
        spike_patterns.append(encoder.encode_vector(test_batch[i], timesteps))
    
    # Decode
    decoded_vectors = []
    for i in range(batch_size):
        decoded_vectors.append(decoder.decode_spikes(spike_patterns[i], vector_dim))
    
    decoded_batch_std = torch.stack(decoded_vectors)
    
    std_time = time.time() - start_time
    print(f"Standard processing time: {std_time:.4f} seconds")
    
    # Calculate metrics
    mse_std = torch.mean((test_batch - decoded_batch_std) ** 2).item()
    similarity_std = torch.nn.functional.cosine_similarity(
        test_batch.reshape(batch_size, -1),
        decoded_batch_std.reshape(batch_size, -1)
    ).mean().item()
    
    print(f"Standard MSE: {mse_std:.6f}")
    print(f"Standard cosine similarity: {similarity_std:.4f}")
    
    # Benchmark adaptive encoding/decoding
    print("\nBenchmarking adaptive encoding/decoding...")
    start_time = time.time()
    
    # Forward pass
    encoded_batch = processor.encode(test_batch, timesteps)
    decoded_batch_adp = processor.decode(encoded_batch)
    
    adp_time = time.time() - start_time
    print(f"Adaptive processing time: {adp_time:.4f} seconds")
    
    # Calculate metrics
    mse_adp = torch.mean((test_batch - decoded_batch_adp) ** 2).item()
    similarity_adp = torch.nn.functional.cosine_similarity(
        test_batch.reshape(batch_size, -1),
        decoded_batch_adp.reshape(batch_size, -1)
    ).mean().item()
    
    print(f"Adaptive MSE: {mse_adp:.6f}")
    print(f"Adaptive cosine similarity: {similarity_adp:.4f}")
    
    # Speedup
    speedup = std_time / adp_time
    print(f"\nAdaptive processing is {speedup:.2f}x faster than standard processing")
    
    # Quality improvement
    if similarity_adp > similarity_std:
        improvement = (similarity_adp - similarity_std) / similarity_std * 100
        print(f"Adaptive processing improves information retention by {improvement:.2f}%")

def main():
    """Run all demos"""
    print("=== Adaptive Spike Processor Demo ===")
    
    try:
        # Basic usage
        demo_basic_usage()
        
        # Noise robustness
        demo_noise_robustness()
        
        # Text processing
        demo_text_processing()
        
        # Performance benchmark
        demo_performance_benchmark()
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 