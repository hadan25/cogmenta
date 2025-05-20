#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Vector Engine and Cross-Model Integration

This script tests and demonstrates the capabilities of the SNNVectorEngine,
particularly its ability to provide unified vectorization across different SNN models
and enable cross-model concept sharing and communication.
"""

import os
import sys
import logging
import numpy as np
import time
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_vector_engine")

# Import vector engine components
try:
    from models.snn.snn_vector_engine import SNNVectorEngine
    from models.snn.vector_engine_integration import VectorEngineIntegration
    from models.snn.vector_engine_integration import get_vector_integration_instance
except ImportError as e:
    logger.error(f"Error importing vector engine components: {e}")
    sys.exit(1)

# Try to import SNN models (for integration testing)
try:
    from models.snn.memory_snn import MemorySNN
    from models.snn.statistical_snn import StatisticalSNN
    MODELS_AVAILABLE = True
except ImportError:
    logger.warning("SNN models not available, will test vector engine only")
    MODELS_AVAILABLE = False

def test_basic_vectorization():
    """Test basic vectorization capabilities"""
    logger.info("===== Testing basic vectorization =====")
    
    # Create vector engine
    vector_engine = SNNVectorEngine(embedding_dim=100, vocab_size=1000)
    
    # Test tokenization
    test_text = "testing the vector engine capabilities"
    tokens = vector_engine.tokenize(test_text)
    logger.info(f"Tokenized text: {test_text} -> {tokens}")
    
    # Test embedding generation
    embedding = vector_engine.get_embedding(test_text)
    logger.info(f"Generated embedding shape: {embedding.shape}")
    
    # Test different input formats
    embedding_from_tokens = vector_engine.get_embedding(tokens)
    similarity = np.dot(embedding, embedding_from_tokens)
    logger.info(f"Similarity between text and token embeddings: {similarity:.4f}")
    
    return True

def test_concept_embeddings():
    """Test concept embedding capabilities"""
    logger.info("===== Testing concept embeddings =====")
    
    # Create vector engine
    vector_engine = SNNVectorEngine(embedding_dim=100, vocab_size=1000)
    
    # Learn concept embeddings
    concepts = {
        "dog": "a domesticated canine animal that is often kept as a pet",
        "cat": "a small feline animal commonly kept as a household pet",
        "computer": "an electronic device for storing and processing data",
        "table": "a piece of furniture with a flat top supported by legs",
        "happiness": "a state of well-being and contentment"
    }
    
    # Learn concepts from descriptions
    for concept, description in concepts.items():
        vector_engine.learn_concept_embedding(
            concept_name=concept,
            text_description=description
        )
        logger.info(f"Learned concept: {concept}")
    
    # Test finding similar concepts
    test_queries = [
        "dog",
        "electronic computing device",
        "feeling of joy and satisfaction"
    ]
    
    for query in test_queries:
        similar = vector_engine.find_similar_concepts(query, top_k=2)
        logger.info(f"Query: '{query}' -> Similar concepts: {similar}")
    
    return True

def test_model_specific_projections():
    """Test model-specific projection capabilities"""
    logger.info("===== Testing model-specific projections =====")
    
    # Create vector engine with model-specific projections
    vector_engine = SNNVectorEngine(
        embedding_dim=100, 
        vocab_size=1000,
        model_specific_projections=True
    )
    
    # Test text input with different model projections
    test_text = "this is a test of model-specific projections"
    
    # Get embeddings for different models
    memory_embedding = vector_engine.get_embedding(test_text, model_type="memory")
    stat_embedding = vector_engine.get_embedding(test_text, model_type="statistical")
    
    # Calculate similarity between model-specific embeddings
    similarity = np.dot(memory_embedding, stat_embedding)
    logger.info(f"Similarity between memory and statistical embeddings: {similarity:.4f}")
    
    # Test cross-model translation
    translated = vector_engine.cross_model_translate(
        memory_embedding, "memory", "statistical"
    )
    
    # Calculate similarity after translation
    similarity_after = np.dot(translated, stat_embedding)
    logger.info(f"Similarity after translation: {similarity_after:.4f}")
    
    return similarity_after > similarity

def test_model_integration():
    """Test integration with actual SNN models"""
    if not MODELS_AVAILABLE:
        logger.warning("Skipping model integration test as models are not available")
        return True
        
    logger.info("===== Testing model integration =====")
    
    # Create vector engine
    vector_engine = SNNVectorEngine(embedding_dim=300, vocab_size=5000)
    
    # Create integration manager
    integration = VectorEngineIntegration(vector_engine)
    
    # Create test models
    memory_snn = MemorySNN(neuron_count=500)
    statistical_snn = StatisticalSNN(neuron_count=500, embedding_dim=300)
    
    # Connect models to vector engine
    memory_adapter = integration.connect_model(memory_snn, "memory")
    stat_adapter = integration.connect_model(statistical_snn, "statistical")
    
    # Test encoding through memory model
    test_memory = "this is a memory to encode and retrieve later"
    encoded_features = memory_snn.tokenizer.encode_memory_content(test_memory)
    logger.info(f"Memory encoding shape: {encoded_features.shape}")
    
    # Test concept learning through statistical model
    concept_name = "test_concept"
    embedding = statistical_snn.learn_concept_embedding(
        concept_name, 
        text_description="this is a test concept for cross-model integration"
    )
    logger.info(f"Learned concept {concept_name} in statistical model")
    
    # Test cross-model concept access
    # Memory model should now be able to access concepts from statistical model
    shared_concepts = integration.get_shared_concepts()
    logger.info(f"Shared concepts across models: {list(shared_concepts.keys())}")
    
    return concept_name in shared_concepts

def test_save_load():
    """Test saving and loading vector engine state"""
    logger.info("===== Testing save/load capabilities =====")
    
    # Create vector engine
    vector_engine = SNNVectorEngine(embedding_dim=100, vocab_size=1000)
    
    # Learn some concepts
    vector_engine.learn_concept_embedding("test1", text_description="first test concept")
    vector_engine.learn_concept_embedding("test2", text_description="second test concept")
    
    # Tokenize some text to build vocabulary
    vector_engine.tokenize("building some vocabulary for the vector engine test")
    
    # Save state
    save_path = "temp_vector_engine_state.json"
    success = vector_engine.save_state(save_path)
    logger.info(f"Save state result: {success}")
    
    # Create a new engine
    new_engine = SNNVectorEngine(embedding_dim=100, vocab_size=1000)
    
    # Load state
    success = new_engine.load_state(save_path)
    logger.info(f"Load state result: {success}")
    
    # Check if concepts were preserved
    concepts1 = set(vector_engine.concept_embeddings.keys())
    concepts2 = set(new_engine.concept_embeddings.keys())
    logger.info(f"Original concepts: {concepts1}")
    logger.info(f"Loaded concepts: {concepts2}")
    
    # Clean up
    try:
        os.remove(save_path)
    except:
        pass
    
    return concepts1 == concepts2

def test_adaptation():
    """Test adaptation of embeddings to specific tasks"""
    logger.info("===== Testing adaptation capabilities =====")
    
    # Create vector engine
    vector_engine = SNNVectorEngine(embedding_dim=100, vocab_size=1000)
    
    # Create some task examples (input -> output pairs)
    task_examples = [
        ("dog", "animal"),
        ("cat", "animal"),
        ("oak", "tree"),
        ("pine", "tree"),
        ("happy", "emotion"),
        ("sad", "emotion")
    ]
    
    # Get initial embeddings
    dog_embedding_before = vector_engine.get_embedding("dog")
    animal_embedding = vector_engine.get_embedding("animal")
    tree_embedding = vector_engine.get_embedding("tree")
    
    # Calculate initial similarities
    sim_dog_animal_before = np.dot(dog_embedding_before, animal_embedding)
    sim_dog_tree_before = np.dot(dog_embedding_before, tree_embedding)
    
    logger.info(f"Before adaptation - dog↔animal: {sim_dog_animal_before:.4f}, dog↔tree: {sim_dog_tree_before:.4f}")
    
    # Adapt embeddings to task
    vector_engine.adapt_embeddings_to_task(task_examples, learn_rate=0.1)
    logger.info("Adapted embeddings to categories task")
    
    # Get embeddings after adaptation
    dog_embedding_after = vector_engine.get_embedding("dog")
    
    # Calculate similarities after adaptation
    sim_dog_animal_after = np.dot(dog_embedding_after, animal_embedding)
    sim_dog_tree_after = np.dot(dog_embedding_after, tree_embedding)
    
    logger.info(f"After adaptation - dog↔animal: {sim_dog_animal_after:.4f}, dog↔tree: {sim_dog_tree_after:.4f}")
    
    # Check if adaptation worked (dog should be more similar to animal)
    improvement = sim_dog_animal_after > sim_dog_animal_before
    logger.info(f"Adaptation improved animal similarity: {improvement}")
    
    return improvement

def test_advanced_integration():
    """Test advanced integration with multiple models and cross-communication"""
    if not MODELS_AVAILABLE:
        logger.warning("Skipping advanced integration test as models are not available")
        return True
        
    logger.info("===== Testing advanced integration =====")
    
    # Get global integration instance
    integration = get_vector_integration_instance()
    
    # Create test models
    models = {
        "memory": MemorySNN(neuron_count=300),
        "statistical": StatisticalSNN(neuron_count=300, embedding_dim=300)
    }
    
    # Upgrade existing models to use vector engine
    adapters = integration.upgrade_existing_models(models)
    
    # Define a test scenario that requires cross-model communication
    test_concepts = [
        ("apple", "a sweet red or green fruit with a core"),
        ("banana", "a long curved yellow fruit"),
        ("orange", "a round orange citrus fruit")
    ]
    
    # Learn concepts in statistical model
    for concept, description in test_concepts:
        models["statistical"].learn_concept_embedding(
            concept, text_description=description
        )
        logger.info(f"Learned concept {concept} in statistical model")
    
    # Test retrieving memory using concept from statistical model
    try:
        fruit_memory = "I remember eating a delicious apple yesterday"
        
        # Encode memory
        encoded = models["memory"].tokenizer.encode_memory_content(fruit_memory)
        logger.info(f"Encoded memory with shape: {encoded.shape}")
        
        # Find similar concepts from statistical model
        similar = models["statistical"].find_similar_concepts("delicious fruit", top_k=1)
        logger.info(f"Retrieved similar concept from statistical model: {similar}")
        
        # Success if we found a match with any of our test concepts
        success = any(concept[0] in [c for c, _ in similar] for concept in test_concepts)
        logger.info(f"Cross-model integration success: {success}")
        
        return success
    except Exception as e:
        logger.error(f"Error in advanced integration test: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    tests = [
        test_basic_vectorization,
        test_concept_embeddings,
        test_model_specific_projections,
        test_save_load,
        test_adaptation,
        test_model_integration,
        test_advanced_integration
    ]
    
    results = {}
    
    for test in tests:
        test_name = test.__name__
        logger.info(f"\nRunning test: {test_name}")
        
        try:
            start_time = time.time()
            result = test()
            duration = time.time() - start_time
            
            results[test_name] = {
                "success": result,
                "duration": round(duration, 2)
            }
            
            logger.info(f"Test {test_name} {'PASSED' if result else 'FAILED'} in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Error in test {test_name}: {e}")
            results[test_name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    logger.info("\n===== Test Results Summary =====")
    passed = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"Passed: {passed}/{len(tests)}")
    
    for name, result in results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        duration = result.get("duration", "N/A")
        error = result.get("error", "")
        
        if error:
            logger.info(f"{status} - {name} - {error}")
        else:
            logger.info(f"{status} - {name} - {duration}s")
    
    return passed == len(tests)

if __name__ == "__main__":
    logger.info("Starting vector engine tests")
    success = run_all_tests()
    sys.exit(0 if success else 1) 