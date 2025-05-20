#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Spike Processor for SNN Models.

This module extends the existing SpikeEncoder and SpikeDecoder classes with
learnable parameters and adaptive mechanisms to optimize spike-based information transfer.
"""

import numpy as np
import torch
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import local modules
from models.snn.spike_encoder import SpikeEncoder, create_encoder
from models.snn.spike_decoder import SpikeDecoder, create_decoder
from models.snn.bidirectional_encoding import BidirectionalProcessor

# Import learning protocol
try:
    # When imported as package
    from models.snn.adaptive_learning_protocol import (
        AdaptiveLearningProtocol, 
        create_learning_protocol,
        LearningPhase,
        AdaptiveLossFunction
    )
except ImportError:
    # When run directly
    from adaptive_learning_protocol import (
        AdaptiveLearningProtocol, 
        create_learning_protocol,
        LearningPhase,
        AdaptiveLossFunction
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptiveSpikeProcessor")

class AdaptiveEncodingLayer(torch.nn.Module):
    """
    Learnable encoding layer that transforms vectors into optimized representations
    before spike encoding.
    
    This layer adapts to the statistical properties of the input data and
    can be trained to maximize information transfer through the spike encoding process.
    """
    
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 300,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize the adaptive encoding layer.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output vectors (should match encoder neuron expectations)
            dropout: Dropout rate for regularization
            activation: Activation function to use ("relu", "gelu", or "tanh")
        """
        super().__init__()
        
        # Define network layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
        # Set activation function
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            logger.warning(f"Unknown activation: {activation}, using GELU")
            self.activation = torch.nn.GELU()
        
        # Initialize weights to preserve variance
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
        # Initialize to approximate identity mapping for better starting point
        # This helps avoid initially negative retention scores
        if input_dim == output_dim:
            # Try to initialize close to identity mapping
            torch.nn.init.eye_(self.fc2.weight)
            self.fc2.bias.data.fill_(0.0)
        
        # Add learnable scaling for different vector components
        # This allows emphasis on important dimensions
        self.importance_scaling = torch.nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform input vectors.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Transformed tensor of shape [batch_size, output_dim]
        """
        # Apply first layer with activation and dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Apply second layer
        x = self.fc2(x)
        
        # Apply importance scaling
        x = x * self.importance_scaling
        
        # Apply normalization to help with spike encoding
        # (values between 0 and 1 work best for encoders)
        x = torch.sigmoid(x)
        
        return x

class AdaptiveDecodingLayer(torch.nn.Module):
    """
    Learnable decoding layer that transforms decoded vectors into optimized
    output representations.
    
    This layer adapts to the statistical properties of decoded spike data and
    can be trained to restore information lost in the spike encoding/decoding process.
    """
    
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 300,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize the adaptive decoding layer.
        
        Args:
            input_dim: Dimension of input vectors (from spike decoder)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output vectors (should match expected vector dim)
            dropout: Dropout rate for regularization
            activation: Activation function to use ("relu", "gelu", or "tanh")
        """
        super().__init__()
        
        # Define network layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
        # Set activation function
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            logger.warning(f"Unknown activation: {activation}, using GELU")
            self.activation = torch.nn.GELU()
        
        # Initialize weights to preserve variance
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
        # Initialize to approximate identity mapping for better starting point
        # This helps avoid initially negative retention scores
        if input_dim == output_dim:
            # Try to initialize close to identity mapping
            torch.nn.init.eye_(self.fc2.weight)
            self.fc2.bias.data.fill_(0.0)
        
        # Add learnable correction factors to compensate for information loss
        # These parameters help restore vector components that are typically
        # degraded in the spike encoding/decoding process
        self.correction_bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.correction_scale = torch.nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform decoded vectors.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Transformed tensor of shape [batch_size, output_dim]
        """
        # Apply first layer with activation and dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Apply second layer
        x = self.fc2(x)
        
        # Apply correction factors
        x = x * self.correction_scale + self.correction_bias
        
        return x

class ModalityAdapter(torch.nn.Module):
    """
    Adapter module for converting between different data modalities.
    
    This module can transform between various representations (text embeddings,
    image features, audio features, etc.) and the common vector space used
    for spike encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 300,
        modality: str = "text",
        hidden_dims: List[int] = None
    ):
        """
        Initialize the modality adapter.
        
        Args:
            input_dim: Dimension of input features for this modality
            output_dim: Dimension of output vectors
            modality: Type of modality ("text", "image", "audio", etc.)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modality = modality.lower()
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            # Create a more uniform size progression
            hidden_dims = [min(input_dim, output_dim) * 2, 
                          (input_dim + output_dim) // 2]
        
        # Create layers based on input and output dimensions
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Final layer to output dimension
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.network = torch.nn.Sequential(*layers)
        
        # Add modality-specific processing modules
        if modality == "image":
            # For image modality, handle both feature vectors and raw images
            # For feature vectors, use a simple MLP that only reshapes
            self.modality_processor = torch.nn.Linear(input_dim, input_dim)
            
            # Flag to indicate if we have a feature vector or raw image
            self.is_feature_vector = True
            
        elif modality == "audio":
            # For audio modality, add 1D convolutional layers for temporal features
            self.modality_processor = torch.nn.Linear(input_dim, input_dim)
        else:
            # For text and other modalities, use identity mapping
            self.modality_processor = torch.nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert input features to the common vector space.
        
        Args:
            x: Input tensor with modality-specific features
            
        Returns:
            Tensor in the common vector space
        """
        # Apply modality-specific processing
        if self.modality == "image" and len(x.shape) >= 3:
            # We have raw image data, not a feature vector
            # Need to handle this differently
            self.is_feature_vector = False
            # Just flatten and use a linear layer
            x = x.reshape(x.size(0), -1)
            # Use a simpler approach that is guaranteed to work
            x = torch.nn.functional.normalize(x, dim=1)
            x = x[:, :min(x.size(1), self.input_dim)]
            # If dimension is smaller than input_dim, pad with zeros
            if x.size(1) < self.input_dim:
                padding = torch.zeros(x.size(0), self.input_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        else:
            # Regular processing for feature vectors
            x = self.modality_processor(x)
        
        # Apply the main network
        x = self.network(x)
        
        return x

class AdaptiveSpikeProcessor(torch.nn.Module):
    """
    End-to-end learnable spike processing system combining encoding and decoding.
    
    Features:
    - Learnable encoding/decoding parameters
    - Automatic adaptation to input data statistics
    - Multi-modal processing capabilities
    - Optimization for information preservation
    - Support for different fidelity levels
    """
    
    def __init__(
        self,
        vector_dim: int = 300,
        neuron_count: int = 1000,
        encoding_type: str = "temporal",
        precision_level: int = 3,
        timesteps: int = 20,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        modalities: List[str] = ["text"],
        bidirectional_processor: Optional[BidirectionalProcessor] = None,
        learning_protocol: Optional[AdaptiveLearningProtocol] = None
    ):
        """
        Initialize the adaptive spike processor.
        
        Args:
            vector_dim: Dimension of input/output vectors
            neuron_count: Number of neurons in spike encoder/decoder
            encoding_type: Type of spike encoding ("temporal", "rate", "population")
            precision_level: Precision level for spike encoding
            timesteps: Number of time steps for spike patterns
            learning_rate: Learning rate for adaptive components
            device: Device to use for processing (cpu/cuda)
            modalities: List of supported modalities
            bidirectional_processor: Optional bidirectional processor for text
            learning_protocol: Optional explicit learning protocol
        """
        super().__init__()
        
        # Store parameters
        self.vector_dim = vector_dim
        self.neuron_count = neuron_count
        self.encoding_type = encoding_type
        self.precision_level = precision_level
        self.timesteps = timesteps
        self.modalities = [m.lower() for m in modalities]
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create adaptive encoding layer
        self.adaptive_encoder = AdaptiveEncodingLayer(
            input_dim=vector_dim,
            hidden_dim=min(vector_dim*2, 512),
            output_dim=vector_dim,
            dropout=0.1,
            activation="gelu"
        )
        
        # Create spike encoder
        self.spike_encoder = create_encoder(
            neuron_count=neuron_count,
            encoding_type=encoding_type,
            precision=precision_level
        )
        
        # Create spike decoder
        self.spike_decoder = create_decoder(
            neuron_count=neuron_count,
            vector_dim=vector_dim,
            decoding_type=encoding_type  # Match encoding type
        )
        
        # Create adaptive decoding layer
        self.adaptive_decoder = AdaptiveDecodingLayer(
            input_dim=vector_dim,
            hidden_dim=min(vector_dim*2, 512),
            output_dim=vector_dim,
            dropout=0.1,
            activation="gelu"
        )
        
        # Create modality adapters
        self.modality_adapters = torch.nn.ModuleDict()
        for modality in self.modalities:
            # Only create adapter if modality is not "vector" (direct input)
            if modality != "vector":
                adapter_input_dim = vector_dim  # Default
                if modality == "image":
                    adapter_input_dim = 2048  # Common for image features
                elif modality == "audio":
                    adapter_input_dim = 1024  # Common for audio features
                
                self.modality_adapters[modality] = ModalityAdapter(
                    input_dim=adapter_input_dim,
                    output_dim=vector_dim,
                    modality=modality
                )
        
        # Set up bidirectional processor for text
        self.bidirectional_processor = bidirectional_processor
        
        # Move to device
        self.to(self.device)
        
        # Set up learning protocol
        self.learning_protocol = learning_protocol or create_learning_protocol(
            learning_rate=learning_rate,
            loss_function="combined",
            initial_phase="pretraining",
            adaptation_threshold=0.95
        )
        
        # Create combined parameters and optimizer using protocol
        parameters = list(self.adaptive_encoder.parameters()) + \
                     list(self.adaptive_decoder.parameters()) + \
                     list(self.modality_adapters.parameters())
                     
        optimizer_config = self.learning_protocol.get_optimizer_config(parameters)
        self.optimizer = optimizer_config["optimizer"]
        
        # Initialize input statistics
        self.input_statistics = {
            "mean": torch.zeros(vector_dim, device=self.device),
            "std": torch.ones(vector_dim, device=self.device),
            "max": torch.zeros(vector_dim, device=self.device),
            "min": torch.zeros(vector_dim, device=self.device),
            "samples_seen": 0
        }
        
        # Initialize metrics
        self.metrics = {
            "reconstruction_loss": [],
            "cosine_similarity": [],
            "information_retention": []
        }
    
    def encode(
        self,
        vector: torch.Tensor,
        timesteps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a vector into spike patterns using the adaptive encoder.
        
        Args:
            vector: Input vector or batch of vectors to encode
            timesteps: Number of time steps (default: self.timesteps)
            
        Returns:
            Spike pattern tensor
        """
        # Use default timesteps if not specified
        if timesteps is None:
            timesteps = self.timesteps
        
        # Make sure input is on the correct device
        vector = vector.to(self.device)
        
        # Update input statistics
        self._update_statistics(vector)
        
        # Apply adaptive encoding transformation
        transformed = self.adaptive_encoder(vector)
        
        # Handle both single vectors and batches
        if len(vector.shape) == 1:
            # Single vector
            spike_pattern = self.spike_encoder.encode_vector(transformed, timesteps)
            
            # Convert sparse tensor to dense if needed
            if spike_pattern.is_sparse:
                spike_pattern = spike_pattern.to_dense()
        else:
            # Batch of vectors - process one by one to avoid sparse tensor issues
            batch_size = vector.shape[0]
            spike_patterns = []
            
            for i in range(batch_size):
                sp = self.spike_encoder.encode_vector(transformed[i], timesteps)
                # Convert sparse tensor to dense if needed
                if sp.is_sparse:
                    sp = sp.to_dense()
                spike_patterns.append(sp)
            
            # Stack results
            spike_pattern = torch.stack(spike_patterns)
        
        return spike_pattern
    
    def decode(
        self,
        spike_pattern: torch.Tensor,
        target_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode spike patterns back to vectors using the adaptive decoder.
        
        Args:
            spike_pattern: Spike pattern tensor to decode
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector tensor
        """
        # Use default vector dimension if not specified
        if target_dim is None:
            target_dim = self.vector_dim
        
        # Make sure input is on the correct device
        if isinstance(spike_pattern, torch.Tensor):
            spike_pattern = spike_pattern.to(self.device)
        
        # Handle both single patterns and batches
        if len(spike_pattern.shape) == 2:
            # Single spike pattern
            decoded = self.spike_decoder.decode_spikes(spike_pattern, target_dim)
        else:
            # Batch of spike patterns
            decoded = self.spike_decoder.decode_batch(spike_pattern, target_dim)
        
        # Apply adaptive decoding transformation
        transformed = self.adaptive_decoder(decoded)
        
        return transformed
    
    def process_text(
        self,
        text: str,
        timesteps: Optional[int] = None,
        return_vector: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process text input through the adaptive spike processor.
        
        Args:
            text: Input text to process
            timesteps: Number of time steps (default: self.timesteps)
            return_vector: Whether to return the intermediate vector representation
            
        Returns:
            Spike pattern tensor or tuple of (spike pattern, vector) if return_vector=True
        """
        if self.bidirectional_processor is None:
            raise ValueError("Bidirectional processor is required for text processing")
        
        # Tokenize and convert to vector
        vector = self.bidirectional_processor.get_vectors_for_tokens(
            self.bidirectional_processor.tokenizer.encode(text)
        )
        
        # Apply modality adapter for text
        if "text" in self.modality_adapters:
            vector = self.modality_adapters["text"](vector)
        
        # Encode to spikes
        spikes = self.encode(vector, timesteps)
        
        if return_vector:
            return spikes, vector
        else:
            return spikes
    
    def process_modality(
        self,
        input_data: torch.Tensor,
        modality: str,
        timesteps: Optional[int] = None,
        return_vector: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process input from a specific modality through the adaptive spike processor.
        
        Args:
            input_data: Input data tensor for the specified modality
            modality: The modality of the input ("text", "image", "audio", etc.)
            timesteps: Number of time steps (default: self.timesteps)
            return_vector: Whether to return the intermediate vector representation
            
        Returns:
            Spike pattern tensor or tuple of (spike pattern, vector) if return_vector=True
        """
        if modality not in self.modality_adapters:
            raise ValueError(f"Modality '{modality}' not supported by this processor")
        
        # Prepare input data based on modality
        if modality == "image":
            # For image features, reshape if needed
            if len(input_data.shape) == 1:
                # If we have a flat vector of image features (e.g. from a CNN)
                # Just use it directly without reshaping for convolution
                input_data = input_data.unsqueeze(0)  # Add batch dimension
            elif len(input_data.shape) == 3:
                # If we have [channels, height, width]
                input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        # Apply modality adapter
        vector = self.modality_adapters[modality](input_data)
        
        # Encode to spikes
        spikes = self.encode(vector, timesteps)
        
        if return_vector:
            return spikes, vector
        else:
            return spikes
    
    def reconstruct_vector(
        self,
        spike_pattern: torch.Tensor,
        original_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Reconstruct vector from spike pattern and calculate information retention.
        
        Args:
            spike_pattern: Spike pattern tensor to decode
            original_vector: Optional original vector for calculating retention metrics
            
        Returns:
            Tuple of (reconstructed vector, information retention score)
        """
        # Decode spike pattern
        reconstructed = self.decode(spike_pattern)
        
        # Calculate information retention if original vector is provided
        retention = None
        if original_vector is not None:
            # Calculate cosine similarity as information retention metric
            original = original_vector.to(self.device)
            sim = torch.nn.functional.cosine_similarity(
                reconstructed.view(1, -1),
                original.view(1, -1)
            )
            retention = sim.item()
            
            # Update metrics
            self.metrics["information_retention"].append(retention)
        
        return reconstructed, retention
    
    def train_step(
        self,
        input_vector: torch.Tensor,
        timesteps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step to optimize the adaptive parameters.
        
        Args:
            input_vector: Input vector or batch of vectors to encode and reconstruct
            timesteps: Number of time steps (default: self.timesteps)
            
        Returns:
            Dictionary of training metrics
        """
        # Use default timesteps if not specified
        if timesteps is None:
            timesteps = self.timesteps
        
        # Make sure input is on the correct device
        input_vector = input_vector.to(self.device)
        
        # Update input statistics
        self._update_statistics(input_vector)
        
        # Get loss function from learning protocol
        loss_fn = self.learning_protocol.get_loss_function()
        
        # Check if we should update parameters
        if self.learning_protocol.learning_phase != LearningPhase.DISABLED:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass: encode and then decode
            encoded = self.encode(input_vector, timesteps)
            reconstructed = self.decode(encoded)
            
            # Calculate loss using protocol's loss function
            loss = loss_fn(input_vector, reconstructed)
            
            # Calculate cosine similarity for monitoring
            if len(input_vector.shape) == 1:
                # For single vector
                cos_sim = torch.nn.functional.cosine_similarity(
                    reconstructed.view(1, -1),
                    input_vector.view(1, -1)
                ).mean()
            else:
                # For batch of vectors
                cos_sim = torch.nn.functional.cosine_similarity(
                    reconstructed,
                    input_vector
                ).mean()
            
            # Check if parameters should be updated based on performance
            if self.learning_protocol.should_update_parameters(cos_sim.item()):
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            
            # Update learning protocol performance metrics
            self.learning_protocol.update_performance(loss.item(), cos_sim.item())
            
            # Update internal metrics
            self.metrics["reconstruction_loss"].append(loss.item())
            self.metrics["cosine_similarity"].append(cos_sim.item())
            
            # Return metrics
            return {
                "reconstruction_loss": loss.item(),
                "cosine_similarity": cos_sim.item(),
                "learning_phase": self.learning_protocol.learning_phase.name,
                "parameters_updated": self.learning_protocol.should_update_parameters(cos_sim.item())
            }
        else:
            # If learning is disabled, just compute metrics but don't update
            with torch.no_grad():
                encoded = self.encode(input_vector, timesteps)
                reconstructed = self.decode(encoded)
                
                # Calculate metrics without gradient tracking
                if len(input_vector.shape) == 1:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        reconstructed.view(1, -1),
                        input_vector.view(1, -1)
                    ).mean()
                else:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        reconstructed,
                        input_vector
                    ).mean()
                
                # Return metrics
                return {
                    "cosine_similarity": cos_sim.item(),
                    "learning_phase": "DISABLED",
                    "parameters_updated": False
                }
    
    def _update_statistics(self, vector: torch.Tensor):
        """
        Update statistics for adaptive behavior.
        
        Args:
            vector: Input vector or batch of vectors
        """
        # Ensure vector is 2D for batch processing
        if len(vector.shape) == 1:
            vector = vector.unsqueeze(0)
        
        # Get batch statistics
        batch_mean = torch.mean(vector, dim=0)
        batch_std = torch.std(vector, dim=0)
        batch_max = torch.max(vector, dim=0)[0]
        batch_min = torch.min(vector, dim=0)[0]
        
        # Update running statistics with exponential moving average
        alpha = 0.05  # Weight for new observations
        
        if self.input_statistics["samples_seen"] > 0:
            self.input_statistics["mean"] = (1 - alpha) * self.input_statistics["mean"] + alpha * batch_mean
            self.input_statistics["std"] = (1 - alpha) * self.input_statistics["std"] + alpha * batch_std
            self.input_statistics["max"] = torch.max(self.input_statistics["max"], batch_max)
            self.input_statistics["min"] = torch.min(self.input_statistics["min"], batch_min)
        else:
            # First batch
            self.input_statistics["mean"] = batch_mean
            self.input_statistics["std"] = batch_std
            self.input_statistics["max"] = batch_max
            self.input_statistics["min"] = batch_min
        
        # Update sample count
        self.input_statistics["samples_seen"] += vector.shape[0]
    
    def adapt_parameters(self):
        """
        Adapt internal parameters based on observed statistics.
        
        This method updates encoding and decoding parameters based on the
        statistics of inputs processed so far.
        """
        if self.input_statistics["samples_seen"] < 100:
            logger.info("Not enough samples for adaptation")
            return
        
        # Adapt importance scaling based on observed std
        std = self.input_statistics["std"].clamp(min=0.01)
        inverse_std = 1.0 / std
        normalized_importance = inverse_std / inverse_std.mean()
        
        # Apply smooth update to importance scaling
        current = self.adaptive_encoder.importance_scaling.data
        updated = 0.9 * current + 0.1 * normalized_importance
        self.adaptive_encoder.importance_scaling.data = updated
        
        logger.info(f"Adapted encoding parameters based on {self.input_statistics['samples_seen']} samples")
    
    def save(
        self,
        directory: str,
        prefix: str = "adaptive_processor"
    ) -> bool:
        """
        Save the adaptive processor to files.
        
        Args:
            directory: Directory to save files in
            prefix: Prefix for filenames
            
        Returns:
            True if successful, False otherwise
        """
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Save model parameters
            model_path = os.path.join(directory, f"{prefix}_model.pt")
            torch.save(self.state_dict(), model_path)
            
            # Save optimizer state
            optimizer_path = os.path.join(directory, f"{prefix}_optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            # Save statistics and metrics
            stats_path = os.path.join(directory, f"{prefix}_stats.pkl")
            with open(stats_path, 'wb') as f:
                pickle.dump({
                    "input_statistics": {
                        k: v.cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in self.input_statistics.items()
                    },
                    "metrics": self.metrics,
                    "config": {
                        "vector_dim": self.vector_dim,
                        "neuron_count": self.neuron_count,
                        "encoding_type": self.encoding_type,
                        "precision_level": self.precision_level,
                        "timesteps": self.timesteps,
                        "modalities": self.modalities
                    }
                }, f)
            
            logger.info(f"Saved adaptive processor to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save adaptive processor: {e}")
            return False
    
    def load(
        self,
        directory: str,
        prefix: str = "adaptive_processor"
    ) -> bool:
        """
        Load the adaptive processor from files.
        
        Args:
            directory: Directory to load files from
            prefix: Prefix for filenames
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model parameters
            model_path = os.path.join(directory, f"{prefix}_model.pt")
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Load optimizer state if available
            optimizer_path = os.path.join(directory, f"{prefix}_optimizer.pt")
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            
            # Load statistics and metrics if available
            stats_path = os.path.join(directory, f"{prefix}_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Load statistics
                    if "input_statistics" in data:
                        for k, v in data["input_statistics"].items():
                            if k in self.input_statistics:
                                if isinstance(v, np.ndarray):
                                    self.input_statistics[k] = torch.tensor(v, device=self.device)
                                else:
                                    self.input_statistics[k] = v
                    
                    # Load metrics
                    if "metrics" in data:
                        self.metrics = data["metrics"]
            
            logger.info(f"Loaded adaptive processor from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adaptive processor: {e}")
            return False

def create_adaptive_processor(
    vector_dim: int = 300,
    neuron_count: int = 1000,
    encoding_type: str = "temporal",
    precision_level: int = 3,
    timesteps: int = 20,
    device: Optional[str] = None,
    modalities: List[str] = ["text"],
    bidirectional_processor: Optional[BidirectionalProcessor] = None,
    learning_rate: float = 0.001,
    learning_protocol: Optional[AdaptiveLearningProtocol] = None
) -> AdaptiveSpikeProcessor:
    """
    Create an adaptive spike processor with the specified configuration.
    
    Args:
        vector_dim: Dimension of input/output vectors
        neuron_count: Number of neurons in spike encoder/decoder
        encoding_type: Type of spike encoding ("temporal", "rate", "population")
        precision_level: Precision level for spike encoding
        timesteps: Number of time steps for spike patterns
        device: Device to use for processing (cpu/cuda)
        modalities: List of supported modalities
        bidirectional_processor: Optional bidirectional processor for text
        learning_rate: Learning rate for adaptive components
        learning_protocol: Optional explicit learning protocol
        
    Returns:
        Configured AdaptiveSpikeProcessor instance
    """
    # Create learning protocol if not provided
    if learning_protocol is None:
        learning_protocol = create_learning_protocol(
            learning_rate=learning_rate,
            loss_function="combined",
            initial_phase="pretraining"
        )
    
    # Create adaptive processor
    processor = AdaptiveSpikeProcessor(
        vector_dim=vector_dim,
        neuron_count=neuron_count,
        encoding_type=encoding_type,
        precision_level=precision_level,
        timesteps=timesteps,
        learning_rate=learning_rate,
        device=device,
        modalities=modalities,
        bidirectional_processor=bidirectional_processor,
        learning_protocol=learning_protocol
    )
    
    return processor

def test_adaptive_processor():
    """Test the adaptive spike processor with a simple example"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a sample vector
    vector_dim = 300
    test_vector = torch.rand(vector_dim)
    
    # Create an adaptive processor
    processor = create_adaptive_processor(
        vector_dim=vector_dim,
        neuron_count=500,
        encoding_type="temporal",
        precision_level=3
    )
    
    # Initial reconstruction test
    print("Testing initial reconstruction (before training)...")
    spikes = processor.encode(test_vector)
    reconstructed, retention = processor.reconstruct_vector(spikes, test_vector)
    
    print(f"Input vector shape: {test_vector.shape}")
    print(f"Spike pattern shape: {spikes.shape}")
    print(f"Reconstructed vector shape: {reconstructed.shape}")
    print(f"Information retention: {retention:.4f}")
    
    # Train the adaptive processor for a few steps
    print("\nTraining adaptive processor...")
    for i in range(10):
        # Generate random training vectors
        training_vectors = torch.rand(8, vector_dim)  # Batch of 8 vectors
        
        # Perform training step
        metrics = processor.train_step(training_vectors)
        
        if (i + 1) % 2 == 0:
            print(f"Step {i+1}: Loss = {metrics['reconstruction_loss']:.6f}, "
                  f"Similarity = {metrics['cosine_similarity']:.4f}")
    
    # Adapt parameters based on statistics
    processor.adapt_parameters()
    
    # Test again after training
    print("\nTesting reconstruction after training...")
    spikes = processor.encode(test_vector)
    reconstructed, retention = processor.reconstruct_vector(spikes, test_vector)
    
    print(f"Information retention: {retention:.4f}")
    
    # Test saving and loading
    print("\nTesting save/load functionality...")
    save_dir = "./temp_adaptive_processor"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the processor
    processor.save(save_dir)
    
    # Create a new processor and load the saved state
    new_processor = create_adaptive_processor(
        vector_dim=vector_dim,
        neuron_count=500
    )
    new_processor.load(save_dir)
    
    # Test the loaded processor
    spikes = new_processor.encode(test_vector)
    reconstructed, retention = new_processor.reconstruct_vector(spikes, test_vector)
    
    print(f"Information retention after load: {retention:.4f}")
    
    # Clean up
    import shutil
    shutil.rmtree(save_dir, ignore_errors=True)
    
    return processor

if __name__ == "__main__":
    test_adaptive_processor() 