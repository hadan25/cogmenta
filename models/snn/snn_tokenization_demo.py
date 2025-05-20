#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of integrated tokenization capabilities in SNN models.

This script demonstrates how to use the integrated BidirectionalProcessor 
in the EnhancedSpikingCore class for text processing in SNN models.
"""

import os
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Import SNN components
from models.snn.enhanced_snn import EnhancedSpikingCore
from models.snn.bidirectional_encoding import BidirectionalProcessor, create_processor
from models.snn.advanced_tokenizer import AdvancedTokenizer, create_tokenizer
from models.snn.spike_encoder import SpikeEncoder, create_encoder
from models.snn.spike_decoder import SpikeDecoder, create_decoder

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SNN_Tokenization_Demo")

def basic_snn_tokenization_demo():
    """Demonstrate the basic integrated tokenization in SNN models"""
    
    logger.info("Creating SNN model with integrated tokenization")
    
    # Create SNN with integrated BidirectionalProcessor
    snn = EnhancedSpikingCore(
        neuron_count=1000,
        topology_type="flexible",
        model_type="memory",
        vector_dim=300
    )
    
    # Example text input
    text_input = "This is a demonstration of the integrated tokenization capabilities in SNN models."
    
    logger.info(f"Processing text input: {text_input}")
    
    # Process text using the integrated processor
    spike_pattern, activations = snn.process_text_input(text_input)
    
    logger.info(f"Spike pattern shape: {spike_pattern.shape}")
    logger.info(f"Number of active neurons: {torch.count_nonzero(activations).item()}")
    
    # Generate text output from spike pattern
    output_text = snn.generate_text_output(spike_pattern)
    
    logger.info(f"Generated text output: {output_text}")
    
    return snn

def sequence_processing_demo(snn=None):
    """Demonstrate processing text as a sequence of tokens"""
    
    if snn is None:
        snn = EnhancedSpikingCore(
            neuron_count=1000,
            topology_type="flexible",
            model_type="memory",
            vector_dim=300
        )
    
    # Example text input
    text_input = "Processing text as sequences allows for better representation of temporal patterns in language."
    
    logger.info(f"Processing text sequence: {text_input}")
    
    # Process text as a sequence
    spike_sequence, activation_sequence = snn.process_text_sequence(text_input)
    
    logger.info(f"Spike sequence shape: {spike_sequence.shape}")
    logger.info(f"Number of tokens in sequence: {len(activation_sequence)}")
    
    # Generate text from the sequence
    output_text = snn.generate_text_from_sequence(spike_sequence)
    
    logger.info(f"Generated text from sequence: {output_text}")
    
    return spike_sequence, activation_sequence

def training_demo(snn=None):
    """Demonstrate training the SNN with text inputs"""
    
    if snn is None:
        snn = EnhancedSpikingCore(
            neuron_count=1000,
            topology_type="flexible",
            model_type="memory",
            vector_dim=300
        )
    
    # Training data
    training_inputs = [
        "Neural networks can process information in a parallel manner.",
        "Spiking neural networks use discrete events called spikes.",
        "Tokenization is the process of converting text to numerical representations.",
        "Learning occurs through synaptic plasticity in biological neurons.",
        "Memory formation involves strengthening connections between neurons."
    ]
    
    expected_outputs = [
        "Neural networks process information in parallel.",
        "SNNs use discrete spike events.",
        "Tokenization converts text to numbers.",
        "Learning happens through synaptic plasticity.",
        "Memory forms by strengthening neural connections."
    ]
    
    logger.info(f"Training the SNN with {len(training_inputs)} examples")
    
    # Train the SNN
    metrics = snn.train_with_text(
        text_inputs=training_inputs,
        expected_outputs=expected_outputs,
        epochs=5
    )
    
    # Plot training metrics
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['loss_history'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['accuracy_history'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    logger.info("Training completed and metrics plotted")
    
    # Test the trained model
    test_input = "Neural networks learn from examples."
    logger.info(f"Testing with new input: {test_input}")
    
    spike_pattern, activations = snn.process_text_input(test_input)
    output_text = snn.generate_text_output(activations)
    
    logger.info(f"Generated output: {output_text}")
    
    return metrics

def compare_model_types():
    """Compare different SNN model types for text processing"""
    
    model_types = ["generic", "memory", "perceptual", "reasoning"]
    test_inputs = [
        "Memory systems store and retrieve information over time.",
        "Visual perception processes information about colors, shapes, and movement.",
        "Logical reasoning involves drawing conclusions from premises."
    ]
    
    results = {}
    
    for model_type in model_types:
        logger.info(f"Testing with model type: {model_type}")
        
        # Create SNN with the specific model type
        snn = EnhancedSpikingCore(
            neuron_count=1000,
            topology_type="flexible",
            model_type=model_type,
            vector_dim=300
        )
        
        model_results = []
        
        for test_input in test_inputs:
            # Process text
            spike_pattern, activations = snn.process_text_input(test_input)
            
            # Generate output
            output_text = snn.generate_text_output(activations)
            
            # Calculate activation pattern metrics
            active_count = torch.count_nonzero(activations).item()
            activation_density = active_count / snn.neuron_count
            
            model_results.append({
                'input': test_input,
                'output': output_text,
                'active_neurons': active_count,
                'activation_density': activation_density
            })
        
        results[model_type] = model_results
    
    # Print comparison
    for model_type, model_results in results.items():
        logger.info(f"\nResults for {model_type.upper()} model:")
        for i, result in enumerate(model_results):
            logger.info(f"  Input {i+1}: {result['input'][:30]}...")
            logger.info(f"  Active neurons: {result['active_neurons']} ({result['activation_density']:.2%})")
            logger.info(f"  Output: {result['output'][:30]}...")
            logger.info("")
    
    return results

def custom_integrated_processor_demo():
    """Demonstrate creating a custom integrated processing pipeline"""
    
    # Create custom tokenizer with specialized vocabulary
    tokenizer = create_tokenizer(method="bpe", vocab_size=5000)
    
    # Create custom encoder/decoder
    encoder = create_encoder(encoding_type="temporal", neuron_count=1000)
    decoder = create_decoder(decoding_type="temporal", neuron_count=1000)
    
    # Create custom processor with the components
    processor = BidirectionalProcessor(
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        model_type="memory",
        vector_dim=300
    )
    
    # Create SNN with custom processor
    snn = EnhancedSpikingCore(
        neuron_count=1000,
        topology_type="flexible",
        bidirectional_processor=processor
    )
    
    # Test the custom processing pipeline
    test_input = "This text is processed by a custom tokenization pipeline."
    
    # Process text
    spike_pattern, activations = snn.process_text_input(test_input)
    output_text = snn.generate_text_output(activations)
    
    logger.info(f"Input: {test_input}")
    logger.info(f"Output: {output_text}")
    
    return snn

def run_all_demos():
    """Run all demonstration functions"""
    logger.info("RUNNING ALL TOKENIZATION DEMOS")
    
    # Basic demo
    logger.info("\n" + "="*50)
    logger.info("BASIC TOKENIZATION DEMO")
    logger.info("="*50)
    snn = basic_snn_tokenization_demo()
    
    # Sequence processing
    logger.info("\n" + "="*50)
    logger.info("SEQUENCE PROCESSING DEMO")
    logger.info("="*50)
    sequence_processing_demo(snn)
    
    # Training demo
    logger.info("\n" + "="*50)
    logger.info("TRAINING DEMO")
    logger.info("="*50)
    training_demo(snn)
    
    # Compare model types
    logger.info("\n" + "="*50)
    logger.info("MODEL TYPE COMPARISON")
    logger.info("="*50)
    compare_model_types()
    
    # Custom processor
    logger.info("\n" + "="*50)
    logger.info("CUSTOM PROCESSOR DEMO")
    logger.info("="*50)
    custom_integrated_processor_demo()
    
    logger.info("\n" + "="*50)
    logger.info("ALL DEMOS COMPLETED")
    logger.info("="*50)

if __name__ == "__main__":
    run_all_demos() 