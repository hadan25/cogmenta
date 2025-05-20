#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bidirectional Encoding/Decoding Processor for SNN Models.

This module provides a unified interface for bidirectional processing:
- Text → Tokens → Vectors → Spikes (for SNN input)
- Spikes → Vectors → Tokens → Text (for SNN output)
"""

import os
import json
import pickle
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Import local modules
from models.snn.advanced_tokenizer import AdvancedTokenizer, create_tokenizer
from models.snn.spike_encoder import SpikeEncoder, create_encoder
from models.snn.spike_decoder import SpikeDecoder, create_decoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BidirectionalProcessor")

class BidirectionalProcessor:
    """
    Bidirectional processor for converting between text and spike patterns.
    
    Features:
    - Text → Tokens → Vectors → Spikes conversion
    - Spikes → Vectors → Tokens → Text conversion
    - Support for batched processing
    - Model-specific vector space projections
    - GPU acceleration support
    """
    
    def __init__(
        self,
        tokenizer: Optional[AdvancedTokenizer] = None,
        encoder: Optional[SpikeEncoder] = None,
        decoder: Optional[SpikeDecoder] = None,
        vector_dim: int = 300,
        device: Optional[str] = None,
        model_type: str = "generic",
        embedding_path: Optional[str] = None
    ):
        """
        Initialize the bidirectional processor.
        
        Args:
            tokenizer: Advanced tokenizer instance or None to create a new one
            encoder: Spike encoder instance or None to create a new one
            decoder: Spike decoder instance or None to create a new one
            vector_dim: Dimension of the vector space
            device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
            model_type: Type of SNN model for specific processing
            embedding_path: Path to pre-trained embeddings file (optional)
        """
        # Set device for GPU acceleration
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Bidirectional processor using device: {self.device}")
        
        # Initialize or set components
        self.tokenizer = tokenizer if tokenizer is not None else create_tokenizer()
        self.encoder = encoder if encoder is not None else create_encoder(device=self.device)
        self.decoder = decoder if decoder is not None else create_decoder(device=self.device)
        
        # Set vector dimension
        self.vector_dim = vector_dim
        
        # Set model type for specific processing
        self.model_type = model_type.lower()
        
        # Initialize vector embeddings
        self.token_embeddings = {}
        
        # Load pre-trained embeddings if provided
        if embedding_path and os.path.exists(embedding_path):
            self._load_pretrained_embeddings(embedding_path)
        
        # Initialize model-specific projections
        self._init_projections()
    
    def _load_pretrained_embeddings(self, embedding_path: str):
        """
        Load pre-trained embeddings from file.
        
        Args:
            embedding_path: Path to the embeddings file
        """
        logger.info(f"Loading pre-trained embeddings from {embedding_path}")
        try:
            # Try loading as numpy array first
            embeddings = np.load(embedding_path)
            if isinstance(embeddings, np.ndarray):
                # Convert numpy array to dictionary
                for i, token in enumerate(self.tokenizer.id_to_token.values()):
                    if i < len(embeddings):
                        self.token_embeddings[token] = torch.tensor(
                            embeddings[i], device=self.device
                        )
            else:
                # Try loading as pickle file
                with open(embedding_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    if isinstance(embeddings, dict):
                        self.token_embeddings = {
                            k: torch.tensor(v, device=self.device)
                            for k, v in embeddings.items()
                        }
            
            logger.info(f"Loaded {len(self.token_embeddings)} pre-trained embeddings")
        except Exception as e:
            logger.error(f"Failed to load pre-trained embeddings: {e}")
            logger.info("Falling back to random initialization")
    
    def _init_projections(self):
        """Initialize model-specific vector space projections"""
        # Create projection matrices to transform between
        # generic vector space and model-specific vector space
        
        # Initialize projection matrices as identity (no transformation) by default
        self.to_model_projection = torch.eye(self.vector_dim, device=self.device)
        self.from_model_projection = torch.eye(self.vector_dim, device=self.device)
        
        # Add specific projections for different model types
        if self.model_type == "memory":
            # Memory models emphasize temporal features
            logger.info("Initializing projection matrices for memory SNN model")
            # Create a projection that emphasizes temporal patterns
            temporal_weight = 0.7
            self.to_model_projection = torch.eye(self.vector_dim, device=self.device)
            self.to_model_projection[0, 0] = temporal_weight  # Emphasize first dimension
            self.from_model_projection = torch.inverse(self.to_model_projection)
        
        elif self.model_type == "perceptual":
            # Perceptual models emphasize spatial/visual features
            logger.info("Initializing projection matrices for perceptual SNN model")
            # Create a projection that emphasizes spatial patterns
            spatial_weight = 0.8
            self.to_model_projection = torch.eye(self.vector_dim, device=self.device)
            self.to_model_projection[1:4, 1:4] *= spatial_weight  # Emphasize spatial dimensions
            self.from_model_projection = torch.inverse(self.to_model_projection)
        
        elif self.model_type == "reasoning":
            # Reasoning models emphasize logical/relational features
            logger.info("Initializing projection matrices for reasoning SNN model")
            # Create a projection that emphasizes logical relationships
            logical_weight = 0.75
            self.to_model_projection = torch.eye(self.vector_dim, device=self.device)
            self.to_model_projection[4:8, 4:8] *= logical_weight  # Emphasize logical dimensions
            self.from_model_projection = torch.inverse(self.to_model_projection)
    
    def get_vectors_for_tokens(self, token_ids: List[int]) -> torch.Tensor:
        """
        Convert token IDs to vector representations.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Vector representation tensor
        """
        # Initialize an empty tensor for the result
        vectors = torch.zeros(len(token_ids), self.vector_dim, device=self.device)
        
        # Get embeddings for each token
        for i, token_id in enumerate(token_ids):
            # Convert token to string form
            token = self.tokenizer.id_to_token.get(token_id, "<UNK>")
            
            # Check if we have an embedding for this token
            if token in self.token_embeddings:
                vectors[i] = self.token_embeddings[token]
            else:
                # If no embedding exists, generate a random one
                # In practice, you'd want to use pre-trained embeddings or learn them
                random_vector = torch.randn(self.vector_dim, device=self.device)
                # Normalize to unit length
                random_vector = random_vector / (torch.norm(random_vector) + 1e-8)
                # Store for future use
                self.token_embeddings[token] = random_vector
                vectors[i] = random_vector
        
        # Apply model-specific projection
        if self.model_type != "generic":
            vectors = torch.matmul(vectors, self.to_model_projection)
        
        return vectors
    
    def get_tokens_for_vectors(self, vectors: torch.Tensor, max_length: int = 100) -> List[int]:
        """
        Convert vector representations to token IDs.
        
        Args:
            vectors: Vector representation tensor
            max_length: Maximum number of tokens to generate
            
        Returns:
            List of token IDs
        """
        # Apply inverse model-specific projection if needed
        if self.model_type != "generic":
            vectors = torch.matmul(vectors, self.from_model_projection)
        
        # Limit the number of vectors to process
        num_vectors = min(vectors.shape[0], max_length)
        vectors = vectors[:num_vectors]
        
        # Initialize results
        token_ids = []
        
        # Process each vector
        for vector in vectors:
            # Find the closest token embedding by cosine similarity
            if len(self.token_embeddings) > 0:
                # Calculate similarities with all known token embeddings
                similarities = {}
                for token, embedding in self.token_embeddings.items():
                    # Calculate cosine similarity
                    similarity = torch.dot(vector, embedding) / (
                        torch.norm(vector) * torch.norm(embedding) + 1e-8
                    )
                    similarities[token] = similarity.item()
                
                # Find token with highest similarity
                best_token = max(similarities.items(), key=lambda x: x[1])[0]
                token_id = self.tokenizer.token_to_id.get(best_token, 
                                                        self.tokenizer.token_to_id["<UNK>"])
            else:
                # If no embeddings exist, use <UNK> token
                token_id = self.tokenizer.token_to_id["<UNK>"]
            
            token_ids.append(token_id)
        
        return token_ids
    
    def text_to_spikes(
        self,
        text: str,
        timesteps: int = 20,
        add_special_tokens: bool = True
    ) -> torch.Tensor:
        """
        Convert text to spike patterns.
        
        Args:
            text: Input text
            timesteps: Number of time steps for spike encoding
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            
        Returns:
            Spike pattern tensor of shape [timesteps, neurons]
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id["<BOS>"]
            eos_id = self.tokenizer.token_to_id["<EOS>"]
            token_ids = [bos_id] + token_ids + [eos_id]
        
        # Convert tokens to vectors
        vectors = self.get_vectors_for_tokens(token_ids)
        
        # Combine vectors into a single representation
        # This is a simple approach - for more complex processing,
        # you could use an attention mechanism or other sequence models
        if len(vectors) > 0:
            combined_vector = vectors.mean(dim=0)
        else:
            combined_vector = torch.zeros(self.vector_dim, device=self.device)
        
        # Encode the vector to spikes
        spikes = self.encoder.encode_vector(combined_vector, timesteps=timesteps)
        
        return spikes
    
    def spikes_to_text(
        self,
        spike_pattern: torch.Tensor,
        max_length: int = 100,
        remove_special_tokens: bool = True
    ) -> str:
        """
        Convert spike patterns back to text.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            max_length: Maximum length of the generated text
            remove_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            Decoded text
        """
        # Decode spikes to vector
        vector = self.decoder.decode_spikes(spike_pattern, self.vector_dim)
        
        # Convert vector to tokens
        token_ids = self.get_tokens_for_vectors(vector.unsqueeze(0), max_length)
        
        # Remove special tokens if requested
        if remove_special_tokens:
            special_token_ids = [
                self.tokenizer.token_to_id[t] 
                for t in ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<CLS>", "<MASK>"]
                if t in self.tokenizer.token_to_id
            ]
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        # Decode tokens to text
        text = self.tokenizer.decode(token_ids)
        
        return text
    
    def text_sequence_to_spike_sequence(
        self,
        text: str,
        timesteps: int = 20,
        add_special_tokens: bool = True
    ) -> torch.Tensor:
        """
        Convert text to a sequence of spike patterns.
        
        Unlike text_to_spikes, this processes each token individually,
        creating a sequence of spike patterns (one per token).
        
        Args:
            text: Input text
            timesteps: Number of time steps for spike encoding
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            
        Returns:
            Sequence of spike patterns tensor of shape [sequence_length, timesteps, neurons]
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id["<BOS>"]
            eos_id = self.tokenizer.token_to_id["<EOS>"]
            token_ids = [bos_id] + token_ids + [eos_id]
        
        # Convert tokens to vectors
        vectors = self.get_vectors_for_tokens(token_ids)
        
        # Encode each vector to spikes
        sequence_length = vectors.shape[0]
        neuron_count = self.encoder.neuron_count
        spike_sequence = torch.zeros(sequence_length, timesteps, neuron_count, device=self.device)
        
        for i, vector in enumerate(vectors):
            spike_sequence[i] = self.encoder.encode_vector(vector, timesteps=timesteps)
        
        return spike_sequence
    
    def spike_sequence_to_text(
        self,
        spike_sequence: torch.Tensor,
        remove_special_tokens: bool = True
    ) -> str:
        """
        Convert a sequence of spike patterns back to text.
        
        Args:
            spike_sequence: Sequence of spike patterns tensor of shape 
                           [sequence_length, timesteps, neurons]
            remove_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            Decoded text
        """
        # Get sequence length
        sequence_length = spike_sequence.shape[0]
        
        # Decode each spike pattern to a vector
        vectors = torch.zeros(sequence_length, self.vector_dim, device=self.device)
        
        for i in range(sequence_length):
            vectors[i] = self.decoder.decode_spikes(spike_sequence[i], self.vector_dim)
        
        # Convert vectors to tokens
        token_ids = self.get_tokens_for_vectors(vectors, max_length=sequence_length)
        
        # Remove special tokens if requested
        if remove_special_tokens:
            special_token_ids = [
                self.tokenizer.token_to_id[t] 
                for t in ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<CLS>", "<MASK>"]
                if t in self.tokenizer.token_to_id
            ]
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        # Decode tokens to text
        text = self.tokenizer.decode(token_ids)
        
        return text
    
    def batch_text_to_spikes(
        self,
        texts: List[str],
        timesteps: int = 20,
        add_special_tokens: bool = True
    ) -> torch.Tensor:
        """
        Convert a batch of texts to spike patterns.
        
        Args:
            texts: List of input texts
            timesteps: Number of time steps for spike encoding
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            
        Returns:
            Batch of spike patterns tensor of shape [batch_size, timesteps, neurons]
        """
        # Initialize output tensor
        batch_size = len(texts)
        neuron_count = self.encoder.neuron_count
        batch_spikes = torch.zeros(batch_size, timesteps, neuron_count, device=self.device)
        
        # Process each text
        for i, text in enumerate(texts):
            batch_spikes[i] = self.text_to_spikes(text, timesteps, add_special_tokens)
        
        return batch_spikes
    
    def batch_spikes_to_text(
        self,
        batch_spikes: torch.Tensor,
        max_length: int = 100,
        remove_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert a batch of spike patterns back to texts.
        
        Args:
            batch_spikes: Batch of spike patterns tensor of shape 
                          [batch_size, timesteps, neurons]
            max_length: Maximum length of the generated text
            remove_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            List of decoded texts
        """
        # Get batch size
        batch_size = batch_spikes.shape[0]
        
        # Initialize results
        texts = []
        
        # Process each spike pattern
        for i in range(batch_size):
            texts.append(self.spikes_to_text(
                batch_spikes[i], max_length, remove_special_tokens
            ))
        
        return texts
    
    def train_embeddings(
        self,
        texts: List[str],
        epochs: int = 5,
        learning_rate: float = 0.01
    ):
        """
        Train token embeddings on a corpus of texts.
        
        Args:
            texts: List of training texts
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Number of tokens with learned embeddings
        """
        logger.info(f"Training token embeddings on {len(texts)} texts")
        
        # Ensure tokenizer is trained
        if not hasattr(self.tokenizer, 'is_trained') or not self.tokenizer.is_trained:
            logger.info("Tokenizer not trained. Training on provided texts.")
            self.tokenizer.train(texts)
        
        # Collect all tokens
        all_tokens = set()
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            for token_id in token_ids:
                token = self.tokenizer.id_to_token.get(token_id, "<UNK>")
                all_tokens.add(token)
        
        logger.info(f"Found {len(all_tokens)} unique tokens")
        
        # Initialize embeddings for all tokens
        for token in all_tokens:
            if token not in self.token_embeddings:
                # Initialize with random values
                random_vector = torch.randn(self.vector_dim, device=self.device)
                # Normalize
                random_vector = random_vector / (torch.norm(random_vector) + 1e-8)
                self.token_embeddings[token] = random_vector
        
        # Train embeddings using a simple co-occurrence based method
        # In practice, you'd want to use a more sophisticated approach like word2vec
        for epoch in range(epochs):
            total_loss = 0.0
            
            for text in texts:
                # Tokenize
                token_ids = self.tokenizer.encode(text)
                tokens = [self.tokenizer.id_to_token.get(tid, "<UNK>") for tid in token_ids]
                
                # Skip short texts
                if len(tokens) < 2:
                    continue
                
                # Process each token and update its embedding based on context
                for i, token in enumerate(tokens):
                    # Define context window
                    start = max(0, i - 2)
                    end = min(len(tokens), i + 3)
                    context = tokens[start:i] + tokens[i+1:end]
                    
                    # Skip if no context
                    if not context:
                        continue
                    
                    # Get current embedding
                    current_emb = self.token_embeddings[token]
                    
                    # Get context embeddings
                    context_embs = [self.token_embeddings[t] for t in context]
                    
                    # Average context embeddings
                    if context_embs:
                        context_avg = torch.stack(context_embs).mean(dim=0)
                        
                        # Calculate loss (simple distance)
                        loss = torch.norm(current_emb - context_avg)
                        total_loss += loss.item()
                        
                        # Update embedding to be more similar to context
                        update = learning_rate * (context_avg - current_emb)
                        new_emb = current_emb + update
                        # Normalize
                        new_emb = new_emb / (torch.norm(new_emb) + 1e-8)
                        self.token_embeddings[token] = new_emb
            
            # Log progress
            avg_loss = total_loss / len(texts)
            logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        return len(self.token_embeddings)
    
    def save(self, directory: str, prefix: str = "bidirectional_processor") -> bool:
        """
        Save processor components to files.
        
        Args:
            directory: Directory to save files in
            prefix: Prefix for saved files
            
        Returns:
            Success status
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Save tokenizer
            tokenizer_path = os.path.join(directory, f"{prefix}_tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            
            # Save embeddings
            embeddings_path = os.path.join(directory, f"{prefix}_embeddings.pkl")
            with open(embeddings_path, 'wb') as f:
                # Convert tensors to numpy arrays for better compatibility
                numpy_embeddings = {
                    token: emb.detach().cpu().numpy()
                    for token, emb in self.token_embeddings.items()
                }
                pickle.dump(numpy_embeddings, f)
            
            # Save projections
            projections_path = os.path.join(directory, f"{prefix}_projections.pkl")
            with open(projections_path, 'wb') as f:
                projections = {
                    'to_model': self.to_model_projection.detach().cpu().numpy(),
                    'from_model': self.from_model_projection.detach().cpu().numpy(),
                    'vector_dim': self.vector_dim,
                    'model_type': self.model_type
                }
                pickle.dump(projections, f)
            
            # Save configuration
            config_path = os.path.join(directory, f"{prefix}_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                config = {
                    'vector_dim': self.vector_dim,
                    'model_type': self.model_type,
                    'encoder_type': self.encoder.encoding_type,
                    'decoder_type': self.decoder.decoding_type,
                    'tokenizer_method': self.tokenizer.method,
                    'vocab_size': self.tokenizer.vocab_size
                }
                json.dump(config, f, indent=2)
            
            logger.info(f"Bidirectional processor saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processor: {e}")
            return False
    
    def load(self, directory: str, prefix: str = "bidirectional_processor") -> bool:
        """
        Load processor components from files.
        
        Args:
            directory: Directory to load files from
            prefix: Prefix for saved files
            
        Returns:
            Success status
        """
        try:
            # Load configuration
            config_path = os.path.join(directory, f"{prefix}_config.json")
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update configuration
            self.vector_dim = config.get('vector_dim', self.vector_dim)
            self.model_type = config.get('model_type', self.model_type)
            
            # Load tokenizer
            tokenizer_path = os.path.join(directory, f"{prefix}_tokenizer.json")
            if not os.path.exists(tokenizer_path):
                logger.error(f"Tokenizer file not found: {tokenizer_path}")
                return False
            
            self.tokenizer.load(tokenizer_path)
            
            # Load embeddings
            embeddings_path = os.path.join(directory, f"{prefix}_embeddings.pkl")
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    numpy_embeddings = pickle.load(f)
                    # Convert numpy arrays to tensors
                    self.token_embeddings = {
                        token: torch.tensor(emb, device=self.device)
                        for token, emb in numpy_embeddings.items()
                    }
            
            # Load projections
            projections_path = os.path.join(directory, f"{prefix}_projections.pkl")
            if os.path.exists(projections_path):
                with open(projections_path, 'rb') as f:
                    projections = pickle.load(f)
                    self.to_model_projection = torch.tensor(
                        projections['to_model'], device=self.device
                    )
                    self.from_model_projection = torch.tensor(
                        projections['from_model'], device=self.device
                    )
            
            # Update encoder and decoder settings
            if 'encoder_type' in config:
                self.encoder = create_encoder(
                    encoding_type=config['encoder_type'],
                    neuron_count=self.encoder.neuron_count,
                    device=self.device
                )
            
            if 'decoder_type' in config:
                self.decoder = create_decoder(
                    decoding_type=config['decoder_type'],
                    neuron_count=self.decoder.neuron_count,
                    device=self.device
                )
            
            logger.info(f"Bidirectional processor loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            return False

# Utility function to create a bidirectional processor
def create_processor(
    model_type: str = "generic",
    vector_dim: int = 300,
    device: Optional[str] = None
) -> BidirectionalProcessor:
    """
    Create a bidirectional processor for a specific SNN model type.
    
    Args:
        model_type: Type of SNN model ("memory", "perceptual", "reasoning", etc.)
        vector_dim: Dimension of the vector space
        device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
        
    Returns:
        BidirectionalProcessor instance
    """
    # Create components
    tokenizer = create_tokenizer()
    encoder = create_encoder(device=device)
    decoder = create_decoder(device=device)
    
    # Create and return processor
    return BidirectionalProcessor(
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        vector_dim=vector_dim,
        device=device,
        model_type=model_type
    )

# Simple demo/test function
def test_processor():
    """Test the bidirectional processor with a simple example"""
    # Create a processor
    processor = create_processor()
    
    # Test text to spikes
    text = "This is a test sentence for bidirectional encoding."
    spikes = processor.text_to_spikes(text)
    
    print(f"Original text: {text}")
    print(f"Converted to spikes shape: {spikes.shape}")
    
    # Test spikes to text
    decoded_text = processor.spikes_to_text(spikes)
    
    print(f"Decoded text: {decoded_text}")
    
    # Test sequence processing
    spike_sequence = processor.text_sequence_to_spike_sequence(text)
    print(f"Spike sequence shape: {spike_sequence.shape}")
    
    return processor

if __name__ == "__main__":
    test_processor() 