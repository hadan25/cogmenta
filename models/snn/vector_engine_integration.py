#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SNN Vector Engine Integration

This module provides integration between the unified SNNVectorEngine and various SNN models.
It handles connecting the vector engine to models, upgrading existing models, and providing
compatibility layers for backwards compatibility.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorEngineIntegration")

class VectorEngineIntegration:
    """
    Integration class for connecting SNNVectorEngine with SNN models.
    
    This class:
    1. Provides adapter methods for each SNN model type
    2. Handles model-specific embedding projections
    3. Maintains backwards compatibility with existing code
    4. Facilitates cross-model communication through shared embeddings
    """
    
    def __init__(self, vector_engine=None, shared_instance=True):
        """
        Initialize the vector engine integration.
        
        Args:
            vector_engine: Instance of SNNVectorEngine (optional)
            shared_instance: Whether to use a shared instance across models
        """
        self.shared_instance = shared_instance
        
        # Import here to avoid circular imports
        try:
            from models.snn.snn_vector_engine import SNNVectorEngine
            self.SNNVectorEngine = SNNVectorEngine
            
            # Create or use provided vector engine
            if vector_engine is None:
                self.vector_engine = SNNVectorEngine()
                logger.info("Created new SNNVectorEngine instance")
            else:
                self.vector_engine = vector_engine
                logger.info("Using provided SNNVectorEngine instance")
                
        except ImportError:
            logger.warning("Could not import SNNVectorEngine, integration will be limited")
            self.SNNVectorEngine = None
            self.vector_engine = None
        
        # Track models using this integration
        self.connected_models = {}
        
        # Create model-specific adapter instances
        self.adapters = {}
    
    def connect_model(self, model, model_type):
        """
        Connect a model to the vector engine.
        
        Args:
            model: SNN model instance
            model_type: Type of model ('memory', 'decision', etc.)
            
        Returns:
            Adapter for the model
        """
        # Check if vector engine is available
        if self.vector_engine is None:
            logger.warning(f"No vector engine available, skipping connection for {model_type}")
            return None
        
        # Create appropriate adapter for model type
        if model_type == "memory":
            adapter = MemorySNNAdapter(self.vector_engine, model)
        elif model_type == "decision":
            adapter = DecisionSNNAdapter(self.vector_engine, model)
        elif model_type == "metacognitive":
            adapter = MetacognitiveSNNAdapter(self.vector_engine, model)
        elif model_type == "statistical":
            adapter = StatisticalSNNAdapter(self.vector_engine, model)
        elif model_type == "perceptual":
            adapter = PerceptualSNNAdapter(self.vector_engine, model)
        elif model_type == "reasoning":
            adapter = ReasoningSNNAdapter(self.vector_engine, model)
        elif model_type == "affective":
            adapter = AffectiveSNNAdapter(self.vector_engine, model)
        else:
            adapter = GenericSNNAdapter(self.vector_engine, model)
            
        # Store adapter
        self.adapters[model_type] = adapter
        self.connected_models[model_type] = model
        
        logger.info(f"Connected {model_type} model to vector engine")
        return adapter
    
    def get_adapter(self, model_type):
        """Get adapter for a specific model type"""
        return self.adapters.get(model_type)
    
    def translate_between_models(self, source_embedding, source_model, target_model):
        """
        Translate embeddings between different model spaces.
        
        Args:
            source_embedding: Embedding from source model
            source_model: Source model type
            target_model: Target model type
            
        Returns:
            Embedding in target model's space
        """
        if self.vector_engine is None:
            logger.warning("No vector engine available, cannot translate between models")
            return source_embedding
            
        return self.vector_engine.cross_model_translate(
            source_embedding, source_model, target_model
        )
    
    def get_shared_concepts(self):
        """Get concepts shared across all models"""
        if self.vector_engine is None:
            return {}
            
        return self.vector_engine.concept_embeddings
    
    def upgrade_existing_models(self, models_dict):
        """
        Upgrade existing models to use the vector engine.
        
        Args:
            models_dict: Dictionary of model_type: model
            
        Returns:
            Dictionary of model_type: adapter
        """
        adapters = {}
        for model_type, model in models_dict.items():
            adapter = self.connect_model(model, model_type)
            if adapter:
                adapters[model_type] = adapter
                
        return adapters

class BaseAdapter:
    """Base adapter for connecting SNN models to vector engine"""
    
    def __init__(self, vector_engine, model):
        """
        Initialize the adapter.
        
        Args:
            vector_engine: SNNVectorEngine instance
            model: SNN model instance
        """
        self.vector_engine = vector_engine
        self.model = model
        self.model_type = "base"
        
    def tokenize(self, text):
        """Tokenize text using vector engine"""
        return self.vector_engine.tokenize(text)
        
    def get_embedding(self, input_data):
        """Get embedding using vector engine with model-specific projection"""
        return self.vector_engine.get_embedding(input_data, model_type=self.model_type)
        
    def apply_to_model(self):
        """Apply integration to model - override in subclasses"""
        pass
        
class MemorySNNAdapter(BaseAdapter):
    """Adapter for Memory SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "memory"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to memory model"""
        # Save original methods for fallback
        self._original_tokenize = getattr(self.model.tokenizer, "tokenize", None)
        self._original_encode_content = getattr(self.model.tokenizer, "encode_memory_content", None)
        
        # Replace tokenizer methods with vector engine versions
        try:
            self.model.tokenizer.tokenize = self.tokenize_for_memory
            self.model.tokenizer.encode_memory_content = self.encode_memory_content
            self.model.tokenizer.vector_engine = self.vector_engine
            logger.info("Successfully upgraded MemorySNN tokenizer")
        except Exception as e:
            logger.error(f"Error upgrading MemorySNN tokenizer: {e}")
            
    def tokenize_for_memory(self, text):
        """Tokenize for memory model, maintaining compatibility"""
        # Use vector engine tokenization
        token_ids = self.vector_engine.tokenize(text)
        
        # If original method is available and tokenization failed, use it as fallback
        if not token_ids and self._original_tokenize:
            logger.debug("Falling back to original tokenize method")
            return self._original_tokenize(self.model.tokenizer, text)
            
        return token_ids
        
    def encode_memory_content(self, content, max_length=50):
        """Encode memory content into a feature vector"""
        if not content:
            return np.zeros(max_length)
            
        # Use vector engine to get embedding
        embedding = self.get_embedding(content)
        
        # Resize to expected length
        if len(embedding) > max_length:
            return embedding[:max_length]
        elif len(embedding) < max_length:
            return np.pad(embedding, (0, max_length - len(embedding)))
        return embedding

class StatisticalSNNAdapter(BaseAdapter):
    """Adapter for Statistical SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "statistical"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to statistical model"""
        # Save original methods for fallback
        self._original_learn_concept = getattr(self.model, "learn_concept_embedding", None)
        self._original_find_similar = getattr(self.model, "find_similar_concepts", None)
        
        # Replace with vector engine versions
        try:
            # Only override if the vector engine is enabled
            self.model.vector_engine = self.vector_engine
            self.model.learn_concept_embedding = self.learn_concept_embedding
            self.model.find_similar_concepts = self.find_similar_concepts
            logger.info("Successfully upgraded StatisticalSNN embeddings")
        except Exception as e:
            logger.error(f"Error upgrading StatisticalSNN embeddings: {e}")
            
    def learn_concept_embedding(self, concept_name, features=None, related_concepts=None):
        """Learn concept embedding using vector engine"""
        # Use vector engine to learn concept embedding
        embedding = self.vector_engine.learn_concept_embedding(
            concept_name=concept_name,
            features=features,
            related_concepts=related_concepts
        )
        
        # Also save in the model's own concept embeddings for compatibility
        if hasattr(self.model, 'concept_embeddings'):
            self.model.concept_embeddings[concept_name] = embedding
            
        return embedding
        
    def find_similar_concepts(self, query_concept=None, query_embedding=None, top_k=5):
        """Find similar concepts using vector engine"""
        if query_concept:
            # Use vector engine to find similar concepts
            return self.vector_engine.find_similar_concepts(query_concept, top_k)
        elif query_embedding is not None:
            # Use vector engine to find similar concepts
            return self.vector_engine.find_similar_concepts(query_embedding, top_k)
        else:
            # Fall back to original method if needed
            if self._original_find_similar:
                return self._original_find_similar(self.model, query_concept, query_embedding, top_k)
            return []

class DecisionSNNAdapter(BaseAdapter):
    """Adapter for Decision SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "decision"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to decision model"""
        # Implement decision-specific adaptations
        # This will depend on the specific implementation of DecisionSNN
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to DecisionSNN")
        except Exception as e:
            logger.error(f"Error connecting vector engine to DecisionSNN: {e}")

class MetacognitiveSNNAdapter(BaseAdapter):
    """Adapter for Metacognitive SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "metacognitive"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to metacognitive model"""
        # Implement metacognitive-specific adaptations
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to MetacognitiveSNN")
        except Exception as e:
            logger.error(f"Error connecting vector engine to MetacognitiveSNN: {e}")

class PerceptualSNNAdapter(BaseAdapter):
    """Adapter for Perceptual SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "perceptual"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to perceptual model"""
        # Implement perceptual-specific adaptations
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to PerceptualSNN")
        except Exception as e:
            logger.error(f"Error connecting vector engine to PerceptualSNN: {e}")

class ReasoningSNNAdapter(BaseAdapter):
    """Adapter for Reasoning SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "reasoning"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to reasoning model"""
        # Implement reasoning-specific adaptations
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to ReasoningSNN")
        except Exception as e:
            logger.error(f"Error connecting vector engine to ReasoningSNN: {e}")

class AffectiveSNNAdapter(BaseAdapter):
    """Adapter for Affective SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "affective"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to affective model"""
        # Implement affective-specific adaptations
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to AffectiveSNN")
        except Exception as e:
            logger.error(f"Error connecting vector engine to AffectiveSNN: {e}")

class GenericSNNAdapter(BaseAdapter):
    """Generic adapter for other SNN models"""
    
    def __init__(self, vector_engine, model):
        super().__init__(vector_engine, model)
        self.model_type = "generic"
        self.apply_to_model()
        
    def apply_to_model(self):
        """Apply integration to generic model"""
        try:
            # Add vector engine to model
            self.model.vector_engine = self.vector_engine
            logger.info("Connected vector engine to generic SNN model")
        except Exception as e:
            logger.error(f"Error connecting vector engine to generic SNN model: {e}")

# Singleton global instance for shared usage
global_vector_integration = None

def get_vector_integration_instance():
    """Get the global vector integration instance"""
    global global_vector_integration
    if global_vector_integration is None:
        try:
            from models.snn.snn_vector_engine import SNNVectorEngine
            vector_engine = SNNVectorEngine()
            global_vector_integration = VectorEngineIntegration(vector_engine, shared_instance=True)
        except ImportError:
            # Create empty instance if vector engine not available
            global_vector_integration = VectorEngineIntegration(None, shared_instance=True)
            
    return global_vector_integration 