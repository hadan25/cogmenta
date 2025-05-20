#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of HybridVectorSpace for hallucination prevention in SNN models.

This script demonstrates how to use the HybridVectorSpace with SNN models
to detect and prevent hallucinations in generated text.
"""

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Import SNN components
from models.snn.enhanced_snn import EnhancedSpikingCore
from models.snn.memory_snn import MemorySNN
from models.snn.bidirectional_encoding import create_processor
from models.snn.hybrid_vector_space import HybridVectorSpace

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HybridVectorDemo")

def train_hybrid_space(corpus):
    """Train the hybrid vector space on a corpus."""
    logger.info(f"Training hybrid vector space on corpus of {len(corpus)} documents")
    
    # Create hybrid vector space
    hybrid_space = HybridVectorSpace(symbolic_dim=300, statistical_dim=300)
    
    # Train on corpus
    hybrid_space.learn(corpus)
    
    return hybrid_space

def create_integrated_snn_model(hybrid_space=None):
    """Create an SNN model with integrated hybrid vector space."""
    # Create bidirectional processor
    processor = create_processor(model_type="memory", vector_dim=300)
    
    # Create memory SNN with bidirectional processor
    snn = MemorySNN(
        neuron_count=1000,
        model_type="memory",
        vector_dim=300,
        bidirectional_processor=processor
    )
    
    # Integrate hybrid vector space if provided
    if hybrid_space:
        # Add hybrid space as attribute
        snn.hybrid_space = hybrid_space
        
        # Patch text generation method to include verification
        original_generate_text = snn.generate_text_output
        
        def generate_text_with_verification(spike_pattern, max_length=100, remove_special_tokens=True, 
                                           context=None, verification_threshold=0.6):
            """Generate text with hallucination verification."""
            # First, generate text normally
            generated_text = original_generate_text(spike_pattern, max_length, remove_special_tokens)
            
            # Verify using hybrid space
            confidence, is_hallucination = hybrid_space.verify_output(generated_text, context)
            
            logger.info(f"Generated text: '{generated_text}'")
            logger.info(f"Confidence: {confidence:.2f}, Hallucination: {is_hallucination}")
            
            # If hallucination detected, try to correct
            if is_hallucination:
                logger.info("Hallucination detected! Attempting to correct...")
                corrected_text, new_confidence = hybrid_space.correct_hallucination(
                    generated_text, confidence
                )
                
                # If correction improved confidence enough, use it
                if new_confidence > verification_threshold:
                    logger.info(f"Corrected to: '{corrected_text}' (new confidence: {new_confidence:.2f})")
                    return corrected_text
                else:
                    # Return original with warning
                    logger.warning(f"Could not correct hallucination (confidence: {confidence:.2f})")
                    return f"[Low confidence: {confidence:.2f}] {generated_text}"
            
            return generated_text
        
        # Replace the method
        snn.generate_text_output = generate_text_with_verification
    
    return snn

def demonstrate_hallucination_prevention():
    """Demonstrate hallucination prevention with hybrid vector space."""
    # Sample corpus for training
    corpus = [
        "spiking neural networks process information using discrete spikes",
        "neurons in the brain communicate through action potentials",
        "artificial intelligence models learn patterns from data",
        "machine learning algorithms require training examples",
        "neural networks have multiple layers of interconnected neurons",
        "deep learning models can extract high-level features automatically",
        "recurrent neural networks have connections that form cycles",
        "transformers use attention mechanisms to process sequences",
        "reinforcement learning agents learn from rewards and punishments",
        "supervised learning uses labeled examples for training"
    ]
    
    # Train hybrid vector space
    hybrid_space = train_hybrid_space(corpus)
    
    # Create SNN with hybrid space
    snn = create_integrated_snn_model(hybrid_space)
    
    # Store some memories
    snn.store_text_memory("Neurons communicate through action potentials called spikes.")
    snn.store_text_memory("Deep learning is a subset of machine learning based on neural networks.")
    snn.store_text_memory("Recurrent neural networks can process sequences of variable length.")
    
    # Test with valid text (should not be flagged as hallucination)
    logger.info("\n" + "="*50)
    logger.info("TESTING WITH VALID TEXT")
    logger.info("="*50)
    
    valid_text = "Neural networks process information using interconnected neurons."
    spike_pattern, _ = snn.process_text_input(valid_text)
    output_text = snn.generate_text_output(spike_pattern)
    
    # Test with potentially hallucinated text
    logger.info("\n" + "="*50)
    logger.info("TESTING WITH POTENTIAL HALLUCINATION")
    logger.info("="*50)
    
    hallucinated_text = "Neural networks eat data and dream in binary code while sleeping."
    spike_pattern, _ = snn.process_text_input(hallucinated_text)
    output_text = snn.generate_text_output(spike_pattern)
    
    # Test retrieval of memories
    logger.info("\n" + "="*50)
    logger.info("TESTING MEMORY RETRIEVAL")
    logger.info("="*50)
    
    retrieved_text = snn.retrieve_text_memory("How do neurons communicate?")
    logger.info(f"Retrieved: {retrieved_text}")
    
    # Compare performance metrics
    logger.info("\n" + "="*50)
    logger.info("COMPARING WITH AND WITHOUT VERIFICATION")
    logger.info("="*50)
    
    test_cases = [
        "Neurons in the brain process information using spikes.",  # Valid
        "Neural networks think about philosophy when not training.",  # Invalid
        "Machine learning models extract patterns from training data.",  # Valid
        "Deep learning models feel sad when they make prediction errors."  # Invalid
    ]
    
    # Turn off verification temporarily
    original_verification = snn.generate_text_output
    snn.generate_text_output = lambda *args, **kwargs: original_generate_text(*args, **kwargs)
    
    # Test without verification
    logger.info("Testing without verification:")
    for test in test_cases:
        spike_pattern, _ = snn.process_text_input(test)
        output = original_generate_text(spike_pattern)
        logger.info(f"Input: '{test}'")
        logger.info(f"Output: '{output}'")
    
    # Restore verification
    snn.generate_text_output = original_verification
    
    # Test with verification
    logger.info("\nTesting with verification:")
    for test in test_cases:
        spike_pattern, _ = snn.process_text_input(test)
        output = snn.generate_text_output(spike_pattern)
        logger.info(f"Input: '{test}'")
        logger.info(f"Output: '{output}'")

def run_all_demos():
    """Run all demonstration functions."""
    logger.info("RUNNING HYBRID VECTOR SPACE DEMOS")
    demonstrate_hallucination_prevention()

if __name__ == "__main__":
    run_all_demos() 