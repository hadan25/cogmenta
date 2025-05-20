#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of text memory capabilities in Memory SNN.

This script demonstrates how to use the integrated BidirectionalProcessor 
with MemorySNN for storing and retrieving memories as text.
"""

import os
import time
import logging
import torch

# Import SNN components
from models.snn.memory_snn import MemorySNN
from models.snn.bidirectional_encoding import create_processor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TextMemoryDemo")

def basic_text_memory_demo():
    """Demonstrate basic text memory storage and retrieval"""
    
    logger.info("Creating MemorySNN with integrated text processing")
    
    # Create MemorySNN with integrated BidirectionalProcessor
    memory_snn = MemorySNN(
        neuron_count=2000,  # More neurons for better memory capacity
        memory_capacity=10,
        model_type="memory",  # Use memory-specific model type
        vector_dim=300
    )
    
    # Example memories to store
    memories = [
        "Neural networks are computational models inspired by the human brain.",
        "Memory formation in the brain involves strengthening synaptic connections.",
        "Hebbian learning is often summarized as 'neurons that fire together, wire together'.",
        "Spiking neural networks use discrete events to transmit information.",
        "The hippocampus plays a crucial role in consolidating information from short-term to long-term memory."
    ]
    
    # Store memories
    memory_keys = []
    logger.info("Storing text memories:")
    
    for i, memory_text in enumerate(memories):
        logger.info(f"  Memory {i+1}: {memory_text[:50]}...")
        key = memory_snn.store_text_memory(memory_text)
        memory_keys.append(key)
        logger.info(f"  Stored with key: {key}")
    
    # Retrieve memories using keys
    logger.info("\nRetrieving memories by key:")
    
    for i, key in enumerate(memory_keys):
        logger.info(f"  Retrieving memory with key: {key}")
        retrieved_text = memory_snn.retrieve_text_memory(memory_key=key)
        logger.info(f"  Retrieved: {retrieved_text[:50]}...")
        
        # Compare with original
        similarity = "similar" if memories[i][:20] in retrieved_text else "different"
        logger.info(f"  Comparison with original: {similarity}")
    
    # Query memories using text
    logger.info("\nQuerying memories with similar text:")
    
    queries = [
        "How do neural networks work?",
        "Tell me about memory formation in the brain",
        "What is Hebbian learning?",
        "How do spiking neural networks communicate?",
        "What brain region is responsible for memory consolidation?"
    ]
    
    for query in queries:
        logger.info(f"  Query: {query}")
        result = memory_snn.retrieve_text_memory(query_text=query)
        logger.info(f"  Result: {result[:50]}...")
    
    # Print memory summary
    logger.info("\nMemory Summary:")
    summary = memory_snn.summarize_text_memories()
    logger.info(summary)
    
    return memory_snn

def associative_memory_demo(memory_snn=None):
    """Demonstrate associative memory capabilities"""
    
    if memory_snn is None:
        memory_snn = MemorySNN(
            neuron_count=2000,
            memory_capacity=10,
            model_type="memory",
            vector_dim=300
        )
    
    # Create associated memory pairs
    associated_pairs = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet'."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second.")
    ]
    
    # Store question-answer pairs
    logger.info("\nStoring associated question-answer pairs:")
    
    for question, answer in associated_pairs:
        # Store both question and answer with the same memory key
        memory_key = f"qa_{hash(question) % 10000}"
        
        logger.info(f"  Question: {question}")
        logger.info(f"  Answer: {answer}")
        
        # Store combined memory
        combined_text = f"{question} {answer}"
        memory_snn.store_text_memory(combined_text, memory_key=memory_key)
        logger.info(f"  Stored with key: {memory_key}")
    
    # Retrieve answers using questions
    logger.info("\nRetrieving answers by questions:")
    
    for question, expected_answer in associated_pairs:
        logger.info(f"  Query: {question}")
        result = memory_snn.retrieve_text_memory(query_text=question)
        logger.info(f"  Result: {result}")
        
        # Check if answer is in the result
        if expected_answer[:20] in result:
            logger.info("  ✓ Answer found in result")
        else:
            logger.info("  × Answer not clearly found in result")
    
    return memory_snn

def contextual_memory_demo(memory_snn=None):
    """Demonstrate contextual memory capabilities"""
    
    if memory_snn is None:
        memory_snn = MemorySNN(
            neuron_count=2000,
            memory_capacity=20,
            model_type="memory",
            vector_dim=300
        )
    
    # Create memories with context
    contexts = ["science", "history", "literature", "technology"]
    contextual_memories = {
        "science": [
            "Atoms are the basic building blocks of matter.",
            "The theory of relativity was proposed by Albert Einstein.",
            "DNA carries genetic information in living organisms."
        ],
        "history": [
            "The Roman Empire fell in 476 CE.",
            "The American Revolution began in 1775.",
            "World War II ended in 1945."
        ],
        "literature": [
            "Shakespeare wrote 37 plays.",
            "The Great Gatsby was written by F. Scott Fitzgerald.",
            "Jane Austen is known for novels like Pride and Prejudice."
        ],
        "technology": [
            "The first personal computer was developed in the 1970s.",
            "The Internet evolved from ARPANET.",
            "Artificial intelligence aims to mimic human intelligence."
        ]
    }
    
    # Store memories with context in the key
    logger.info("\nStoring contextual memories:")
    
    for context, memories in contextual_memories.items():
        logger.info(f"  Context: {context}")
        
        for memory_text in memories:
            memory_key = f"{context}_{hash(memory_text) % 10000}"
            memory_snn.store_text_memory(memory_text, memory_key=memory_key)
            logger.info(f"    Stored: {memory_text}")
    
    # Query by context
    logger.info("\nQuerying memories by context:")
    
    for context in contexts:
        query = f"Tell me about {context}"
        logger.info(f"  Query: {query}")
        
        # Find memories with this context in the key
        memory_keys = [key for key in getattr(memory_snn, 'text_memory_patterns', {}).keys() 
                      if key.startswith(context)]
        
        logger.info(f"  Found {len(memory_keys)} memories in context '{context}':")
        
        for key in memory_keys:
            memory_text = memory_snn.retrieve_text_memory(memory_key=key)
            logger.info(f"    {memory_text}")
    
    return memory_snn

def run_all_demos():
    """Run all text memory demonstration functions"""
    logger.info("RUNNING TEXT MEMORY DEMOS")
    
    # Basic demo
    logger.info("\n" + "="*50)
    logger.info("BASIC TEXT MEMORY DEMO")
    logger.info("="*50)
    memory_snn = basic_text_memory_demo()
    
    # Associative memory
    logger.info("\n" + "="*50)
    logger.info("ASSOCIATIVE MEMORY DEMO")
    logger.info("="*50)
    memory_snn = associative_memory_demo(memory_snn)
    
    # Contextual memory
    logger.info("\n" + "="*50)
    logger.info("CONTEXTUAL MEMORY DEMO")
    logger.info("="*50)
    contextual_memory_demo(memory_snn)
    
    logger.info("\n" + "="*50)
    logger.info("ALL DEMOS COMPLETED")
    logger.info("="*50)

if __name__ == "__main__":
    run_all_demos() 