#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SNNVectorEngine - Unified Vectorization System for SNN Models

This module provides a unified vectorization system for all SNN models,
supporting multiple encoding strategies, efficient sparse representations,
and optional integration with external embedding systems.
"""

import numpy as np
import os
import json
import pickle
import logging
import re
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SNNVectorEngine")

class SNNVectorEngine:
    """
    Unified vectorization engine for SNN models.
    
    Features:
    - Multiple encoding strategies (one-hot, random projection, learned)
    - Support for sparse vector representations
    - Optional import/export of embeddings from external sources
    - Consistent interface across all SNN models
    - Shared vocabulary and embedding space for cross-model communication
    - GPU acceleration for tensor operations
    """
    
    def __init__(self, 
                 embedding_dim: int = 300,
                 vocab_size: int = 50000,
                 strategy: str = "adaptive",
                 sparsity: float = 0.1,
                 shared_embeddings: bool = True,
                 model_specific_projections: bool = True,
                 pretrained_path: Optional[str] = None,
                 device: Optional[str] = None,
                 bidirectional_processor: Optional[Any] = None):
        """
        Initialize the SNNVectorEngine.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            vocab_size: Maximum vocabulary size
            strategy: Vectorization strategy ('one-hot', 'random', 'adaptive', 'learned')
            sparsity: Target sparsity level for embeddings (0.0-1.0)
            shared_embeddings: Whether to share embeddings across models
            model_specific_projections: Whether to create model-specific projections
            pretrained_path: Path to pretrained embeddings (optional)
            device: Device to run computations on ('cpu', 'cuda', 'cuda:0', etc. or None for auto-detection)
            bidirectional_processor: Optional BidirectionalProcessor for standardized processing
        """
        # Set device for GPU acceleration
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.strategy = strategy
        self.sparsity = sparsity
        self.shared_embeddings = shared_embeddings
        self.model_specific_projections = model_specific_projections
        
        # Store bidirectional processor if provided
        self.bidirectional_processor = bidirectional_processor
        
        # Core vocabulary and embeddings (shared across models)
        self.word_to_idx = {}  # Word to index mapping
        self.idx_to_word = {}  # Index to word mapping
        self.word_embeddings = {}  # Word to embedding mapping (tensors)
        self.concept_embeddings = {}  # Concept to embedding mapping (tensors)
        
        # Model-specific projection matrices (for customizing embeddings per model)
        self.model_projections = {}
        
        # Vocabulary statistics for adaptive vectorization
        self.word_frequencies = defaultdict(int)
        self.word_contexts = defaultdict(set)
        self.cooccurrence_matrix = {}
        
        # Initialize vocabulary with special tokens
        self._init_special_tokens()
        
        # Load pretrained embeddings if specified
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
            logger.info(f"Loaded pretrained embeddings from {pretrained_path}")
        else:
            logger.info("No pretrained embeddings loaded, using random initialization")
            
        # Initialize model projections if needed
        if model_specific_projections:
            self._init_model_projections()
        
        # Initialize compatibility with BidirectionalProcessor
        self._init_bidirectional_compatibility()
    
    def _init_bidirectional_compatibility(self):
        """Initialize compatibility with BidirectionalProcessor"""
        # This method ensures that SNNVectorEngine can work with 
        # the standardized BidirectionalProcessor for all SNN models
        
        # Check if we need to load default tokenizer vocabulary
        if self.bidirectional_processor and hasattr(self.bidirectional_processor, 'tokenizer'):
            # Import vocabulary from bidirectional processor's tokenizer if available
            tokenizer = self.bidirectional_processor.tokenizer
            if hasattr(tokenizer, 'token_to_id') and hasattr(tokenizer, 'id_to_token'):
                # Sync vocabulary with tokenizer
                for token, token_id in tokenizer.token_to_id.items():
                    if token not in self.word_to_idx:
                        self.word_to_idx[token] = token_id
                        self.idx_to_word[token_id] = token
                
                logger.info(f"Synchronized vocabulary with BidirectionalProcessor ({len(self.word_to_idx)} tokens)")
                
            # Share embeddings if supported and available in BidirectionalProcessor
            if hasattr(self.bidirectional_processor, 'token_embeddings'):
                for token, embedding in self.bidirectional_processor.token_embeddings.items():
                    if token not in self.word_embeddings:
                        # Convert to our device
                        self.word_embeddings[token] = embedding.to(self.device)
                        
                logger.info(f"Imported {len(self.bidirectional_processor.token_embeddings)} embeddings from BidirectionalProcessor")
            
            # Import model-specific projections if available
            if hasattr(self.bidirectional_processor, 'to_model_projection') and hasattr(self.bidirectional_processor, 'from_model_projection'):
                model_type = getattr(self.bidirectional_processor, 'model_type', 'generic')
                if model_type != 'generic':
                    # Import the projection matrices
                    self.model_projections[model_type] = self.bidirectional_processor.to_model_projection.to(self.device)
                    logger.info(f"Imported projection matrix for {model_type} model from BidirectionalProcessor")
    
    def ensure_vector_compatibility(self, vector: torch.Tensor, model_type: Optional[str] = None) -> torch.Tensor:
        """
        Ensure a vector is compatible with the expected format (dimension, device, etc.)
        
        Args:
            vector: Input vector to check
            model_type: Optional model type for model-specific processing
            
        Returns:
            Standardized compatible vector
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        
        # Move to correct device if needed
        if torch.is_tensor(vector) and vector.device != self.device:
            vector = vector.to(self.device)
        
        # Ensure correct dimension
        if vector.shape[-1] != self.embedding_dim:
            # Need to resize
            if vector.shape[-1] < self.embedding_dim:
                # Pad with zeros
                padding = torch.zeros(*vector.shape[:-1], self.embedding_dim - vector.shape[-1], 
                                      dtype=vector.dtype, device=self.device)
                vector = torch.cat([vector, padding], dim=-1)
            else:
                # Truncate
                vector = vector[..., :self.embedding_dim]
        
        # Apply model-specific projection if requested
        if model_type and model_type in self.model_projections:
            vector = torch.matmul(vector, self.model_projections[model_type])
        
        # Normalize vector
        norm = torch.norm(vector, dim=-1, keepdim=True)
        if torch.any(norm > 0):
            vector = vector / (norm + 1e-8)  # Add small epsilon to avoid division by zero
            
        return vector
    
    def process_token_batch(self, tokens: List[str], model_type: Optional[str] = None) -> torch.Tensor:
        """
        Process a batch of tokens to vectors in a standardized way.
        
        Args:
            tokens: List of tokens to process
            model_type: Optional model type for model-specific processing
            
        Returns:
            Batch of token vectors
        """
        batch_size = len(tokens)
        vectors = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        
        for i, token in enumerate(tokens):
            # Get or create embedding for this token
            if token in self.word_embeddings:
                vectors[i] = self.word_embeddings[token]
            else:
                # Create a new embedding
                vectors[i] = self._create_random_embedding()
                # Store for future use
                self.word_embeddings[token] = vectors[i].clone()
        
        # Apply model-specific projection if requested
        if model_type and model_type in self.model_projections:
            vectors = torch.matmul(vectors, self.model_projections[model_type])
        
        return vectors
    
    def standardize_batch_processing(self, 
                                    input_data: Union[List[str], List[int], torch.Tensor, np.ndarray],
                                    model_type: Optional[str] = None) -> torch.Tensor:
        """
        Standardize batch processing for different input types.
        
        Args:
            input_data: Input data (text, tokens, vectors)
            model_type: Optional model type for model-specific processing
            
        Returns:
            Standardized batch of vectors
        """
        # Check input type and convert accordingly
        if isinstance(input_data, list):
            if all(isinstance(x, str) for x in input_data):
                # List of strings (words/tokens)
                return self.process_token_batch(input_data, model_type)
            elif all(isinstance(x, int) for x in input_data):
                # List of token IDs
                tokens = [self.idx_to_word.get(idx, "<UNK>") for idx in input_data]
                return self.process_token_batch(tokens, model_type)
            else:
                # Try to convert each element
                vectors = []
                for item in input_data:
                    if isinstance(item, str):
                        # Single token
                        if item in self.word_embeddings:
                            vectors.append(self.word_embeddings[item])
                        else:
                            vectors.append(self._create_random_embedding())
                    elif isinstance(item, (list, np.ndarray)):
                        # Convert to tensor and ensure compatibility
                        vectors.append(self.ensure_vector_compatibility(
                            torch.tensor(item, dtype=torch.float32, device=self.device),
                            model_type
                        ))
                    elif torch.is_tensor(item):
                        # Ensure tensor compatibility
                        vectors.append(self.ensure_vector_compatibility(item, model_type))
                    else:
                        # Unknown type, use zero vector
                        vectors.append(torch.zeros(self.embedding_dim, device=self.device))
                
                # Stack vectors into a batch
                return torch.stack(vectors)
        
        elif isinstance(input_data, np.ndarray):
            # Convert numpy array to tensor and ensure compatibility
            tensor = torch.tensor(input_data, dtype=torch.float32, device=self.device)
            return self.ensure_vector_compatibility(tensor, model_type)
        
        elif torch.is_tensor(input_data):
            # Ensure tensor compatibility
            return self.ensure_vector_compatibility(input_data, model_type)
        
        else:
            # Unknown input type
            logger.warning(f"Unknown input type for standardize_batch_processing: {type(input_data)}")
            # Return batch of zero vectors (assuming batch size 1 if unknown)
            batch_size = 1
            return torch.zeros(batch_size, self.embedding_dim, device=self.device)
    
    def _init_special_tokens(self):
        """Initialize special tokens in the vocabulary"""
        special_tokens = [
            "<PAD>",  # Padding token
            "<UNK>",  # Unknown token
            "<BOS>",  # Beginning of sequence
            "<EOS>",  # End of sequence
            "<MASK>"  # Mask token for training
        ]
        
        # Add special tokens to vocabulary
        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
            
            # Create random embedding for each special token
            self.word_embeddings[token] = self._create_random_embedding()
    
    def _init_model_projections(self):
        """Initialize model-specific projection matrices"""
        # Default models to support
        models = ["memory", "decision", "metacognitive", 
                  "statistical", "perceptual", "reasoning", "affective"]
        
        for model in models:
            # Create a random projection matrix for each model
            # This allows model-specific adaptations of the shared embeddings
            if self.sparsity > 0:
                # Create sparse projection matrix
                self.model_projections[model] = self._create_sparse_projection()
            else:
                # Create dense projection matrix as tensor
                self.model_projections[model] = torch.randn(
                    self.embedding_dim, self.embedding_dim, 
                    device=self.device
                ) * 0.1
            
            logger.info(f"Created projection matrix for {model} model")
    
    def _create_random_embedding(self) -> torch.Tensor:
        """Create a random embedding vector"""
        if self.sparsity > 0:
            # Create sparse random embedding
            embedding = torch.zeros(self.embedding_dim, device=self.device)
            
            # Determine number of non-zero elements
            non_zero = int(self.embedding_dim * (1 - self.sparsity))
            non_zero = max(1, non_zero)  # Ensure at least one non-zero element
            
            # Set random positions to random values
            indices = torch.randperm(self.embedding_dim, device=self.device)[:non_zero]
            values = torch.randn(non_zero, device=self.device)
            embedding[indices] = values
        else:
            # Create dense random embedding
            embedding = torch.randn(self.embedding_dim, device=self.device)
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _create_sparse_projection(self) -> torch.Tensor:
        """Create a sparse projection matrix"""
        projection = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        
        # For each output dimension, connect to a subset of input dimensions
        for i in range(self.embedding_dim):
            # Determine number of connections
            n_connections = int(self.embedding_dim * (1 - self.sparsity))
            n_connections = max(1, n_connections)
            
            # Choose random input dimensions
            input_dims = torch.randperm(self.embedding_dim, device=self.device)[:n_connections]
            
            # Set weights for these connections
            weights = torch.randn(n_connections, device=self.device)
            projection[i, input_dims] = weights
        
        # Normalize projection matrix
        for i in range(self.embedding_dim):
            norm = torch.norm(projection[i])
            if norm > 0:
                projection[i] = projection[i] / norm
        
        return projection
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        # Simple whitespace tokenization for now
        # Can be extended with more sophisticated tokenization methods
        if not isinstance(text, str):
            return []
        
        words = text.lower().strip().split()
        token_ids = []
        
        for word in words:
            # Update word frequency
            self.word_frequencies[word] += 1
            
            # Get token ID, adding to vocabulary if not present
            if word in self.word_to_idx:
                token_id = self.word_to_idx[word]
            elif len(self.word_to_idx) < self.vocab_size:
                # Add new word to vocabulary
                token_id = len(self.word_to_idx)
                self.word_to_idx[word] = token_id
                self.idx_to_word[token_id] = word
                
                # Create embedding for new word
                self.word_embeddings[word] = self._create_random_embedding()
            else:
                # Vocabulary is full, use unknown token
                token_id = self.word_to_idx["<UNK>"]
            
            token_ids.append(token_id)
        
        # Update context information for adaptive vectorization
        if len(words) > 1:
            for i, word in enumerate(words):
                # Add surrounding words as context
                context_window = 2
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                
                for j in range(start, end):
                    if j != i:
                        self.word_contexts[word].add(words[j])
        
        return token_ids
    
    def get_embedding(self, 
                      input_data: Union[str, List[int], List[str]], 
                      model_type: Optional[str] = None) -> torch.Tensor:
        """
        Get embedding vector for input data.
        
        Args:
            input_data: Input text, token IDs, or words
            model_type: Specific model to get embedding for (optional)
            
        Returns:
            Embedding vector (torch.Tensor)
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Input is text, tokenize first
            token_ids = self.tokenize(input_data)
            words = input_data.lower().strip().split()
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                return torch.zeros(self.embedding_dim, device=self.device)
                
            if isinstance(input_data[0], int):
                # Input is token IDs
                token_ids = input_data
                words = [self.idx_to_word.get(idx, "<UNK>") for idx in token_ids]
            elif isinstance(input_data[0], str):
                # Input is words
                words = input_data
                token_ids = [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) 
                            for word in words]
            else:
                logger.error(f"Unsupported input type: {type(input_data[0])}")
                return torch.zeros(self.embedding_dim, device=self.device)
        else:
            logger.error(f"Unsupported input type: {type(input_data)}")
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Get embeddings for each word
        embeddings = []
        for word in words:
            if word in self.word_embeddings:
                embeddings.append(self.word_embeddings[word])
            else:
                embeddings.append(self.word_embeddings["<UNK>"])
        
        if not embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Average embeddings to get a single vector
        embedding = torch.mean(torch.stack(embeddings), dim=0)
        
        # Apply model-specific projection if needed
        if model_type and self.model_specific_projections and model_type in self.model_projections:
            embedding = torch.matmul(embedding, self.model_projections[model_type])
            
            # Renormalize
            norm = torch.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def learn_concept_embedding(self, 
                               concept_name: str, 
                               features: Optional[torch.Tensor] = None,
                               text_description: Optional[str] = None,
                               related_concepts: Optional[List[str]] = None) -> torch.Tensor:
        """
        Learn embedding for a concept.
        
        Args:
            concept_name: Name of the concept
            features: Feature vector (optional)
            text_description: Text description of the concept (optional)
            related_concepts: List of related concept names (optional)
            
        Returns:
            Concept embedding vector
        """
        # Initialize embedding vectors list
        embedding_sources = []
        
        # 1. Use provided feature vector if available
        if features is not None:
            # Ensure features has embedding_dim dimensions
            if features.shape[0] != self.embedding_dim:
                if features.shape[0] > self.embedding_dim:
                    features = features[:self.embedding_dim]
                else:
                    features = torch.nn.functional.pad(features, (0, self.embedding_dim - features.shape[0]))
            
            embedding_sources.append(features)
        
        # 2. Use text description if available
        if text_description:
            text_embedding = self.get_embedding(text_description)
            embedding_sources.append(text_embedding)
        
        # 3. Add related concept influences
        if related_concepts:
            for rel_concept in related_concepts:
                if rel_concept in self.concept_embeddings:
                    embedding_sources.append(self.concept_embeddings[rel_concept])
        
        # 4. Fallback to concept name itself
        if not embedding_sources:
            name_embedding = self.get_embedding(concept_name)
            embedding_sources.append(name_embedding)
        
        # Average all embedding sources
        embedding = torch.mean(torch.stack(embedding_sources), dim=0)
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Store the concept embedding
        self.concept_embeddings[concept_name] = embedding
        
        return embedding
    
    def find_similar_concepts(self, 
                             query: Union[str, torch.Tensor], 
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts similar to the query.
        
        Args:
            query: Query text or embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (concept_name, similarity_score) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.get_embedding(query)
        else:
            query_embedding = query
        
        # Calculate similarities
        similarities = []
        for concept, embedding in self.concept_embeddings.items():
            similarity = torch.dot(query_embedding, embedding)
            similarities.append((concept, similarity.item()))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
    
    def adapt_embeddings_to_task(self, task_examples: List[Tuple[str, str]], learn_rate: float = 0.01):
        """
        Adapt embeddings based on task-specific examples.
        
        Args:
            task_examples: List of (input, output) text pairs
            learn_rate: Learning rate for adaptation
            
        Returns:
            None - updates internal embeddings
        """
        # Extract pairs of embeddings from examples
        embedding_pairs = []
        for input_text, output_text in task_examples:
            input_embedding = self.get_embedding(input_text)
            output_embedding = self.get_embedding(output_text)
            embedding_pairs.append((input_embedding, output_embedding))
        
        # Nothing to learn if no pairs
        if not embedding_pairs:
            return
        
        # Simple approach: move word embeddings appearing in inputs
        # toward their corresponding outputs
        for input_text, output_text in task_examples:
            input_words = input_text.lower().strip().split()
            output_embedding = self.get_embedding(output_text)
            
            for word in input_words:
                if word in self.word_embeddings:
                    # Move word embedding toward output embedding
                    current = self.word_embeddings[word]
                    direction = output_embedding - current
                    self.word_embeddings[word] = current + learn_rate * direction
                    
                    # Renormalize
                    self.word_embeddings[word] = self.word_embeddings[word] / torch.norm(self.word_embeddings[word])
    
    def cross_model_translate(self, 
                             embedding: torch.Tensor, 
                             source_model: str, 
                             target_model: str) -> torch.Tensor:
        """
        Translate embedding from source model space to target model space.
        
        Args:
            embedding: Source embedding vector
            source_model: Source model type
            target_model: Target model type
            
        Returns:
            Translated embedding in target model space
        """
        if not self.model_specific_projections:
            # No translation needed if not using model-specific projections
            return embedding
        
        if source_model not in self.model_projections or target_model not in self.model_projections:
            logger.warning(f"Model projection not found for {source_model} or {target_model}")
            return embedding
        
        # First, project back to shared space (approximate inverse)
        source_proj = self.model_projections[source_model]
        # Simple pseudoinverse approach
        source_inv = torch.pinverse(source_proj)
        shared_space = torch.matmul(embedding, source_inv)
        
        # Then, project to target model space
        target_proj = self.model_projections[target_model]
        target_embedding = torch.matmul(shared_space, target_proj)
        
        # Normalize
        norm = torch.norm(target_embedding)
        if norm > 0:
            target_embedding = target_embedding / norm
        
        return target_embedding
    
    def load_pretrained(self, filepath: str) -> int:
        """
        Load pretrained embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Number of embeddings loaded
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return 0
        
        count = 0
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.txt':
                # Text format (word2vec, GloVe, etc.)
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Check if first line is header (vocab_size, dim)
                    first_line = f.readline().strip().split()
                    if len(first_line) == 2 and all(x.isdigit() for x in first_line):
                        # Skip header and move to first word
                        pass
                    else:
                        # Rewind if not a header
                        f.seek(0)
                    
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) <= 1:
                            continue
                        
                        word = parts[0]
                        
                        # Skip if already in vocabulary
                        if word in self.word_embeddings:
                            continue
                        
                        # Extract vector values
                        try:
                            vector = torch.tensor([float(val) for val in parts[1:]], device=self.device)
                            
                            # Handle dimension mismatch
                            if vector.shape[0] != self.embedding_dim:
                                if vector.shape[0] > self.embedding_dim:
                                    vector = vector[:self.embedding_dim]
                                else:
                                    vector = torch.nn.functional.pad(vector, (0, self.embedding_dim - vector.shape[0]))
                            
                            # Normalize
                            norm = torch.norm(vector)
                            if norm > 0:
                                vector = vector / norm
                            
                            # Add to vocabulary
                            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                                idx = len(self.word_to_idx)
                                self.word_to_idx[word] = idx
                                self.idx_to_word[idx] = word
                            
                            # Store embedding
                            self.word_embeddings[word] = vector
                            count += 1
                            
                            # Limit loading to vocab size
                            if count >= self.vocab_size:
                                break
                                
                        except ValueError:
                            logger.warning(f"Error parsing embedding for word: {word}")
                            continue
                            
            elif file_ext in ['.json', '.jsonl']:
                # JSON format
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle different JSON formats
                    if isinstance(data, dict):
                        if 'embeddings' in data:
                            # Format: {"embeddings": {"word": [values]}}
                            embeddings_dict = data['embeddings']
                        else:
                            # Format: {"word": [values]}
                            embeddings_dict = data
                            
                        for word, vector_data in embeddings_dict.items():
                            vector = torch.tensor(vector_data, device=self.device)
                            
                            # Handle dimension mismatch
                            if vector.shape[0] != self.embedding_dim:
                                if vector.shape[0] > self.embedding_dim:
                                    vector = vector[:self.embedding_dim]
                                else:
                                    vector = torch.nn.functional.pad(vector, (0, self.embedding_dim - vector.shape[0]))
                            
                            # Normalize
                            norm = torch.norm(vector)
                            if norm > 0:
                                vector = vector / norm
                            
                            # Add to vocabulary
                            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                                idx = len(self.word_to_idx)
                                self.word_to_idx[word] = idx
                                self.idx_to_word[idx] = word
                            
                            # Store embedding
                            self.word_embeddings[word] = vector
                            count += 1
                            
                            # Limit loading to vocab size
                            if count >= self.vocab_size:
                                break
            
            elif file_ext in ['.pickle', '.pkl']:
                # Pickle format
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        for word, vector in data.items():
                            if not isinstance(word, str):
                                continue
                                
                            # Handle dimension mismatch
                            if vector.shape[0] != self.embedding_dim:
                                if vector.shape[0] > self.embedding_dim:
                                    vector = vector[:self.embedding_dim]
                                else:
                                    vector = torch.nn.functional.pad(vector, (0, self.embedding_dim - vector.shape[0]))
                            
                            # Normalize
                            norm = torch.norm(vector)
                            if norm > 0:
                                vector = vector / norm
                            
                            # Add to vocabulary
                            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                                idx = len(self.word_to_idx)
                                self.word_to_idx[word] = idx
                                self.idx_to_word[idx] = word
                            
                            # Store embedding
                            self.word_embeddings[word] = vector
                            count += 1
                            
                            # Limit loading to vocab size
                            if count >= self.vocab_size:
                                break
            
            logger.info(f"Loaded {count} embeddings from {filepath}")
            return count
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return 0
    
    def save_state(self, filepath: str) -> bool:
        """
        Save vectorization engine state to file.
        
        Args:
            filepath: Path to save state
            
        Returns:
            Success boolean
        """
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Prepare data to save
            state = {
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'vocab_size': self.vocab_size,
                    'strategy': self.strategy,
                    'sparsity': self.sparsity,
                    'shared_embeddings': self.shared_embeddings,
                    'model_specific_projections': self.model_specific_projections
                },
                'word_to_idx': self.word_to_idx,
                'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
                'word_frequencies': dict(self.word_frequencies),
                # Convert numpy arrays to lists for JSON serialization
                'model_projections': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                                    for k, v in self.model_projections.items()},
                'word_embeddings': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                                for k, v in self.word_embeddings.items()},
                'concept_embeddings': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                                    for k, v in self.concept_embeddings.items()}
            }
            
            # Save to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f)
                
            logger.info(f"Saved vectorization engine state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load vectorization engine state from file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            Success boolean
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
            
        try:
            # Load JSON state
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Load configuration
            config = state.get('config', {})
            self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.strategy = config.get('strategy', self.strategy)
            self.sparsity = config.get('sparsity', self.sparsity)
            self.shared_embeddings = config.get('shared_embeddings', self.shared_embeddings)
            self.model_specific_projections = config.get('model_specific_projections', 
                                                      self.model_specific_projections)
            
            # Load dictionaries
            self.word_to_idx = state.get('word_to_idx', {})
            self.idx_to_word = {int(k): v for k, v in state.get('idx_to_word', {}).items()}
            self.word_frequencies = defaultdict(int, state.get('word_frequencies', {}))
            
            # Load embeddings and projections, converting lists back to tensors
            self.word_embeddings = {k: torch.tensor(v, device=self.device) for k, v in state.get('word_embeddings', {}).items()}
            self.concept_embeddings = {k: torch.tensor(v, device=self.device) for k, v in state.get('concept_embeddings', {}).items()}
            self.model_projections = {k: torch.tensor(v, device=self.device) for k, v in state.get('model_projections', {}).items()}
            
            logger.info(f"Loaded vectorization engine state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def process_via_bidirectional(self, text_input, model_type=None):
        """
        Process text input through BidirectionalProcessor if available
        
        Args:
            text_input: Input text to process
            model_type: Optional model type for model-specific processing
            
        Returns:
            Vector representation of the input
        """
        if not self.bidirectional_processor:
            logger.warning("BidirectionalProcessor not available. Using standard processing.")
            return self.get_embedding(text_input, model_type)
            
        # Use BidirectionalProcessor to tokenize and vectorize
        token_ids = self.bidirectional_processor.tokenizer.encode(text_input)
        vectors = self.bidirectional_processor.get_vectors_for_tokens(token_ids)
        
        # Apply model-specific projection if needed
        if model_type and model_type != self.bidirectional_processor.model_type:
            # Need to convert between model types
            from_model = self.bidirectional_processor.model_type
            
            # First project back to generic space using bidirectional processor
            if hasattr(self.bidirectional_processor, 'from_model_projection'):
                generic_vectors = torch.matmul(vectors, self.bidirectional_processor.from_model_projection)
            else:
                generic_vectors = vectors
                
            # Then project to target model space
            if model_type in self.model_projections:
                target_vectors = torch.matmul(generic_vectors, self.model_projections[model_type])
            else:
                target_vectors = generic_vectors
                
            return target_vectors
        
        return vectors
    
    def convert_between_models(self, vectors, source_model="generic", target_model="generic"):
        """
        Convert vectors from one SNN model's space to another.
        
        Args:
            vectors: Input vectors to convert (tensor or list)
            source_model: Source model type (e.g., "memory", "perceptual")
            target_model: Target model type (e.g., "decision", "reasoning")
            
        Returns:
            Converted vectors in target model's space
        """
        # Convert input to tensor if it's a list
        if isinstance(vectors, list):
            vectors = torch.tensor(vectors, device=self.device, dtype=torch.float32)
        
        # Handle different input shapes
        if len(vectors.shape) == 1:
            # Single vector: [dim] -> [1, dim]
            vectors = vectors.unsqueeze(0)
            was_single = True
        else:
            was_single = False
        
        # Get source and target model projection matrices
        source_key = f"{source_model}_projection"
        target_key = f"{target_model}_projection"
        
        # Create projection matrices if they don't exist
        if source_key not in self.model_projections:
            logging.warning(f"Source model projection matrix not found: {source_key}")
            self.model_projections[source_key] = self._create_projection_matrix(
                source_model, self.embedding_dim
            )
        
        if target_key not in self.model_projections:
            logging.warning(f"Target model projection matrix not found: {target_key}")
            self.model_projections[target_key] = self._create_projection_matrix(
                target_model, self.embedding_dim
            )
        
        # Get projection matrices
        source_proj = self.model_projections[source_key]
        target_proj = self.model_projections[target_key]
        
        # Convert vectors using the projection matrices
        # First project from source model to generic space
        # For source projection, we use the inverse/transpose of the matrix
        # Since projection matrices are orthogonal, transpose ≈ inverse
        source_to_generic = torch.matmul(vectors, source_proj.transpose(0, 1))
        
        # Then project from generic space to target model
        generic_to_target = torch.matmul(source_to_generic, target_proj)
        
        # Normalize vectors if needed for consistency
        should_normalize = True  # Default to normalizing vectors
        
        if should_normalize:
            # Normalize each vector to unit length
            norms = torch.norm(generic_to_target, dim=1, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            generic_to_target = generic_to_target / norms
        
        # Return in original format
        if was_single:
            return generic_to_target.squeeze(0)
        else:
            return generic_to_target
    
    def standardize_vector_input(self, input_data, model_type=None):
        """
        Standardize vector input from various formats
        
        Args:
            input_data: Input data (text, tokens, or vectors)
            model_type: Optional model type for model-specific processing
            
        Returns:
            Standardized vectors
        """
        # Case 1: String input (text)
        if isinstance(input_data, str):
            # Check if we should use BidirectionalProcessor
            if self.bidirectional_processor:
                return self.process_via_bidirectional(input_data, model_type)
            else:
                return self.get_embedding(input_data, model_type)
                
        # Case 2: List of tokens
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return self.process_token_batch(input_data, model_type)
            
        # Case 3: NumPy array or PyTorch tensor (vector)
        elif isinstance(input_data, (np.ndarray, torch.Tensor)):
            return self.ensure_vector_compatibility(input_data, model_type)
            
        # Case 4: Something else - try to use standardize_batch_processing
        else:
            try:
                return self.standardize_batch_processing(input_data, model_type)
            except Exception as e:
                logger.error(f"Failed to standardize input: {e}")
                # Return empty vector as fallback
                return torch.zeros(self.embedding_dim, device=self.device)
    
    def _create_projection_matrix(self, model_type, output_dim):
        """
        Create a model-specific projection matrix.
        
        Args:
            model_type: Type of SNN model for projection
            output_dim: Dimension of the output vector space
            
        Returns:
            Projection matrix for the specified model
        """
        logging.info(f"Created projection matrix for {model_type} model")
        
        # Create a semi-random projection matrix with some structure based on model type
        # Use a random seed based on model type name for consistency
        seed = sum(ord(c) for c in model_type)
        rng = np.random.RandomState(seed)
        
        # Create an orthogonal projection matrix
        if torch is not None:
            # Create with PyTorch
            random_matrix = torch.randn(output_dim, output_dim, device=self.device, 
                                       dtype=torch.float32)
            # Make it orthogonal (each row is orthogonal to the others)
            q, r = torch.linalg.qr(random_matrix)
            proj_matrix = q
        else:
            # Create with NumPy
            random_matrix = rng.randn(output_dim, output_dim)
            # Make it orthogonal
            q, r = np.linalg.qr(random_matrix)
            proj_matrix = torch.tensor(q, device=self.device, dtype=torch.float32)
        
        # Add model-specific bias based on model type
        # This ensures different models have slightly different vector spaces
        # but the transformations are reversible
        if model_type == "memory":
            # Memory model: bias towards long-term structures
            proj_matrix = self._add_model_bias(proj_matrix, 0.05, "symmetric")
        elif model_type == "decision":
            # Decision model: bias towards discriminative features
            proj_matrix = self._add_model_bias(proj_matrix, 0.08, "diagonal")
        elif model_type == "reasoning":
            # Reasoning model: bias towards relational structures
            proj_matrix = self._add_model_bias(proj_matrix, 0.06, "block")
        elif model_type == "perceptual":
            # Perceptual model: bias towards sensory features
            proj_matrix = self._add_model_bias(proj_matrix, 0.07, "band")
        elif model_type == "affective":
            # Affective model: bias towards emotional features
            proj_matrix = self._add_model_bias(proj_matrix, 0.09, "sparse")
        elif model_type == "metacognitive":
            # Metacognitive model: balanced bias
            proj_matrix = self._add_model_bias(proj_matrix, 0.04, "symmetric")
        elif model_type == "statistical":
            # Statistical model: bias towards pattern features
            proj_matrix = self._add_model_bias(proj_matrix, 0.05, "block")
        
        return proj_matrix
    
    def _add_model_bias(self, matrix, strength, structure_type):
        """
        Add a model-specific bias to the projection matrix.
        
        Args:
            matrix: Base projection matrix
            strength: Strength of the bias
            structure_type: Type of structure to add
            
        Returns:
            Biased projection matrix
        """
        if not torch.is_tensor(matrix):
            matrix = torch.tensor(matrix, device=self.device, dtype=torch.float32)
            
        size = matrix.shape[0]
        
        if structure_type == "diagonal":
            # Add a diagonal bias
            diag_bias = torch.eye(size, device=self.device) * strength
            return matrix + diag_bias
        elif structure_type == "symmetric":
            # Add a symmetric bias
            bias = torch.randn(size, size, device=self.device) * strength
            bias = (bias + bias.t()) / 2  # Make symmetric
            return matrix + bias
        elif structure_type == "block":
            # Add block-structured bias
            bias = torch.zeros(size, size, device=self.device)
            block_size = max(1, size // 4)
            for i in range(0, size, block_size):
                end_i = min(i + block_size, size)
                bias[i:end_i, i:end_i] = torch.randn(end_i-i, end_i-i, device=self.device) * strength
            return matrix + bias
        elif structure_type == "band":
            # Add band-structured bias
            bias = torch.zeros(size, size, device=self.device)
            band_width = max(1, size // 8)
            for i in range(size):
                for j in range(max(0, i-band_width), min(size, i+band_width+1)):
                    bias[i, j] = torch.randn(1, device=self.device).item() * strength
            return matrix + bias
        elif structure_type == "sparse":
            # Add sparse random bias
            mask = (torch.rand(size, size, device=self.device) < 0.1).float()
            bias = torch.randn(size, size, device=self.device) * strength * mask
            return matrix + bias
        else:
            # No bias
            return matrix 