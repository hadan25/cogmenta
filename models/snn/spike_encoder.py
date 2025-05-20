#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spike Encoder for SNN Models.

This module provides various encoding strategies to convert vector representations
into spike patterns for processing in Spiking Neural Networks.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpikeEncoder")

class SpikeEncoder:
    """
    Encoder for converting vector representations to spike patterns.
    
    Features:
    - Rate-based encoding (vector values → firing rates)
    - Temporal encoding (vector values → spike timing)
    - Population encoding (distributed representation)
    - Combined encoding strategies
    - GPU acceleration support
    """
    
    def __init__(
        self,
        neuron_count: int = 1000,
        encoding_type: str = "rate",
        temporal_window: int = 20,
        threshold: float = 0.1,
        device: Optional[str] = None,
        sparse_output: bool = True,
        precision_level: int = 3,
        model_type: str = "generic"
    ):
        """
        Initialize the spike encoder.
        
        Args:
            neuron_count: Number of neurons to use for encoding
            encoding_type: Type of encoding ("rate", "temporal", "population", "combined")
            temporal_window: Number of time steps for temporal encoding
            threshold: Minimum value threshold for generating spikes
            device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
            sparse_output: Whether to use sparse tensor representation for efficiency
            precision_level: Level of precision for temporal encoding (1-5)
            model_type: Type of SNN model for specialized encoding
        """
        # Set device for GPU acceleration
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Spike encoder using device: {self.device}")
        
        self.neuron_count = neuron_count
        self.encoding_type = encoding_type.lower()
        self.temporal_window = temporal_window
        self.threshold = threshold
        self.sparse_output = sparse_output
        self.precision_level = max(1, min(5, precision_level))  # Clamp between 1-5
        self.model_type = model_type.lower()
        
        # Set precision-dependent parameters
        self._set_precision_parameters()
        
        # Validate encoding type
        valid_types = ["rate", "temporal", "population", "combined"]
        if self.encoding_type not in valid_types:
            logger.warning(f"Unknown encoding type: {self.encoding_type}, falling back to rate-based encoding")
            self.encoding_type = "rate"
        
        # Initialize encoding parameters
        self._init_encoding_params()
    
    def _init_encoding_params(self):
        """Initialize encoding-specific parameters"""
        if self.encoding_type == "population":
            # For population encoding, create receptive fields
            self.receptive_fields = self._create_receptive_fields()
        elif self.encoding_type == "combined":
            # For combined encoding, initialize parameters for all methods
            self.receptive_fields = self._create_receptive_fields()
            
    def _create_receptive_fields(self):
        """
        Create receptive fields for population encoding.
        
        Each neuron responds to a range of input values, with preference
        for a specific value within that range.
        
        Returns:
            Tensor of receptive field parameters
        """
        # Create evenly spaced centers for receptive fields
        centers = torch.linspace(-1.0, 1.0, self.neuron_count).to(self.device)
        
        # Create width values for each receptive field
        # Narrower in the middle, wider at the edges
        widths = 0.1 + 0.2 * torch.abs(centers)
        
        # Combine into a single tensor [centers, widths]
        receptive_fields = torch.stack([centers, widths], dim=1)
        
        return receptive_fields
    
    def _set_precision_parameters(self):
        """Configure parameters based on precision level (1-5)"""
        # Set temporal resolution based on precision level
        precision_config = {
            1: {"jitter": 0.2, "noise": 0.15, "resolution": 0.2},  # Low precision, faster
            2: {"jitter": 0.15, "noise": 0.1, "resolution": 0.15},
            3: {"jitter": 0.1, "noise": 0.05, "resolution": 0.1},  # Medium/default
            4: {"jitter": 0.05, "noise": 0.02, "resolution": 0.05},
            5: {"jitter": 0.01, "noise": 0.01, "resolution": 0.01}  # High precision, slower
        }
        
        config = precision_config.get(self.precision_level, precision_config[3])
        
        # Temporal jitter (randomness in spike timing)
        self.temporal_jitter = config["jitter"]
        
        # Noise level for stochastic processes
        self.noise_level = config["noise"]
        
        # Temporal resolution (min time between spikes)
        self.temporal_resolution = config["resolution"]
        
        # Adjust window based on precision if needed
        if self.precision_level >= 4:
            # Higher precision may need larger window
            self.temporal_window = max(self.temporal_window, 25)
    
    def encode_vector(
        self,
        vector: Union[np.ndarray, torch.Tensor, List[float]],
        timesteps: int = None
    ) -> torch.Tensor:
        """
        Encode a vector into a spike pattern.
        
        Args:
            vector: Input vector to encode (numpy array, PyTorch tensor, or list)
            timesteps: Number of time steps (default: self.temporal_window)
            
        Returns:
            Spike pattern tensor of shape [timesteps, neurons] or sparse tensor
        """
        # Set default timesteps if not specified
        if timesteps is None:
            timesteps = self.temporal_window
        
        # Convert input to PyTorch tensor
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        elif isinstance(vector, list):
            vector = torch.tensor(vector, dtype=torch.float32)
        
        # Move tensor to the specified device
        vector = vector.to(self.device)
        
        # Normalize vector to range [0, 1] if not already normalized
        if vector.min() < 0 or vector.max() > 1:
            vector = (vector - vector.min()) / (vector.max() - vector.min() + 1e-8)
        
        # Choose encoding method
        if self.encoding_type == "rate":
            spike_pattern = self._rate_encode(vector, timesteps)
        elif self.encoding_type == "temporal":
            spike_pattern = self._temporal_encode(vector, timesteps)
        elif self.encoding_type == "population":
            spike_pattern = self._population_encode(vector, timesteps)
        elif self.encoding_type == "combined":
            spike_pattern = self._combined_encode(vector, timesteps)
        else:
            # Fallback to rate encoding
            spike_pattern = self._rate_encode(vector, timesteps)
        
        # Convert to sparse representation if requested
        if self.sparse_output and spike_pattern.is_sparse == False:
            indices = torch.nonzero(spike_pattern, as_tuple=True)
            values = spike_pattern[indices]
            
            # Only convert if sparsity would be beneficial (less than 20% non-zero)
            if values.numel() < 0.2 * spike_pattern.numel():
                sparse_spike = torch.sparse_coo_tensor(
                    indices=torch.stack(indices, dim=0), 
                    values=values,
                    size=spike_pattern.shape,
                    device=self.device
                )
                return sparse_spike
        
        return spike_pattern
    
    def _rate_encode(
        self,
        vector: torch.Tensor,
        timesteps: int
    ) -> torch.Tensor:
        """
        Encode vector using rate-based encoding.
        
        Higher vector values lead to more spikes across the time window.
        
        Args:
            vector: Normalized input vector
            timesteps: Number of time steps
            
        Returns:
            Spike pattern tensor of shape [timesteps, len(vector)]
        """
        # Ensure vector is at least 1D
        if vector.dim() == 0:
            vector = vector.unsqueeze(0)
        
        # Adjust vector length if needed
        vector_len = vector.size(0)
        if vector_len > self.neuron_count:
            # Truncate if too long
            vector = vector[:self.neuron_count]
        elif vector_len < self.neuron_count:
            # Pad with zeros if too short
            padding = torch.zeros(self.neuron_count - vector_len, device=self.device)
            vector = torch.cat([vector, padding])
        
        # Apply threshold for activation
        active_mask = vector > self.threshold
        
        # Determine firing probabilities based on value
        # Scale based on precision level - higher precision = more deterministic
        if self.precision_level >= 4:
            # High precision: More deterministic rate coding
            probs = vector.clamp(min=0.0, max=1.0)
            # Sharpen probabilities for high precision
            probs = torch.pow(probs, 2.0 - self.precision_level * 0.2)
        else:
            # Lower precision: More stochastic rate coding
            probs = vector.clamp(min=0.0, max=1.0)
        
        # Zero out probabilities for inactive neurons
        probs = probs * active_mask.float()
        
        # Generate spikes for each time step
        spikes = torch.zeros(timesteps, self.neuron_count, device=self.device)
        
        # Precision-based spike generation
        if self.precision_level >= 4:
            # Higher precision: Regular, structured spiking
            for i in range(self.neuron_count):
                if probs[i] > self.threshold:
                    # For high values, create regular spike train based on probability
                    # Calculate inter-spike interval
                    p = probs[i].item()
                    if p > 0:
                        interval = max(1, int(timesteps / (timesteps * p)))
                        # Regular spike times with small jitter for biological realism
                        spike_times = torch.arange(0, timesteps, interval, device=self.device)
                        if self.temporal_jitter > 0:
                            jitter = (torch.rand_like(spike_times) * 2 - 1) * self.temporal_jitter * interval
                            spike_times = (spike_times + jitter).long().clamp(0, timesteps - 1)
                        
                        for t in spike_times:
                            if t < timesteps:
                                spikes[t, i] = 1.0
        else:
            # Lower precision: Stochastic spiking based on probability
            for t in range(timesteps):
                # Generate random values
                random_vals = torch.rand(self.neuron_count, device=self.device)
                # Apply noise based on precision level
                if self.noise_level > 0:
                    noise = torch.randn(self.neuron_count, device=self.device) * self.noise_level
                    random_vals = random_vals + noise
                
                # Generate spikes where random value is less than probability
                spike_mask = (random_vals < probs).float()
                spikes[t] = spike_mask
        
        return spikes
    
    def _temporal_encode(
        self,
        vector: torch.Tensor,
        timesteps: int
    ) -> torch.Tensor:
        """
        Encode vector using temporal encoding.
        
        Higher vector values lead to earlier spikes within the time window.
        
        Args:
            vector: Normalized input vector
            timesteps: Number of time steps
            
        Returns:
            Spike pattern tensor of shape [timesteps, len(vector)]
        """
        # Ensure vector is at least 1D
        if vector.dim() == 0:
            vector = vector.unsqueeze(0)
        
        # Adjust vector length if needed
        vector_len = vector.size(0)
        if vector_len > self.neuron_count:
            # Truncate if too long
            vector = vector[:self.neuron_count]
        elif vector_len < self.neuron_count:
            # Pad with zeros if too short
            padding = torch.zeros(self.neuron_count - vector_len, device=self.device)
            vector = torch.cat([vector, padding])
        
        # Create empty spike pattern
        spikes = torch.zeros(timesteps, self.neuron_count, device=self.device)
        
        # Convert values to spike times
        # Higher values spike earlier, scale based on precision
        vector_clipped = vector.clamp(min=0.0, max=1.0)
        
        # Calculate precise spike timing (as continuous values)
        # Apply precision-level appropriate jitter for biological realism
        if self.precision_level > 1:
            jitter = torch.randn(self.neuron_count, device=self.device) * self.temporal_jitter
            spike_times_raw = (1.0 - vector_clipped) * (timesteps - 1) + jitter
        else:
            spike_times_raw = (1.0 - vector_clipped) * (timesteps - 1)
        
        # Clamp to valid range
        spike_times = spike_times_raw.clamp(0, timesteps - 1)
        
        # Ensure minimum temporal resolution between spikes based on precision
        if self.precision_level >= 3:
            # Force minimum distance between spikes for higher precision
            sorted_indices = torch.argsort(spike_times)
            for i in range(1, len(sorted_indices)):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                if spike_times[idx] - spike_times[prev_idx] < self.temporal_resolution:
                    spike_times[idx] = spike_times[prev_idx] + self.temporal_resolution
        
        # Convert to integer time steps
        spike_times_discrete = spike_times.long()
        
        # Create spike pattern
        for i in range(self.neuron_count):
            t = spike_times_discrete[i].item()
            if t < timesteps and vector[i] > self.threshold:  # Only spike if above threshold
                spikes[t, i] = 1.0
        
        return spikes
    
    def _population_encode(
        self,
        vector: torch.Tensor,
        timesteps: int
    ) -> torch.Tensor:
        """
        Encode vector using population encoding.
        
        Each value is encoded by the collective activity of multiple neurons,
        with each neuron having a preferred value range.
        
        Args:
            vector: Normalized input vector
            timesteps: Number of time steps
            
        Returns:
            Spike pattern tensor of shape [timesteps, self.neuron_count]
        """
        # Ensure vector is at least 1D
        if vector.dim() == 0:
            vector = vector.unsqueeze(0)
        
        # Rescale vector to [-1, 1] range for receptive fields
        if vector.min() >= 0 and vector.max() <= 1:
            vector = 2.0 * vector - 1.0
        
        # Initialize spike pattern
        spikes = torch.zeros(timesteps, self.neuron_count, device=self.device)
        
        # Process each dimension of the input vector
        dim = vector.size(0)
        
        # Divide neurons among dimensions (at least 1 per dimension)
        neurons_per_dim = max(1, self.neuron_count // dim)
        
        for d in range(min(dim, self.neuron_count)):
            value = vector[d]
            
            # Calculate neuron responses based on receptive fields
            start_idx = d * neurons_per_dim
            end_idx = min(start_idx + neurons_per_dim, self.neuron_count)
            
            # Get receptive fields for this dimension
            rf_centers = self.receptive_fields[start_idx:end_idx, 0]
            rf_widths = self.receptive_fields[start_idx:end_idx, 1]
            
            # Calculate activation based on Gaussian receptive fields
            distance = (value - rf_centers) / rf_widths
            activation = torch.exp(-(distance ** 2))
            
            # Generate spikes based on activation levels
            for t in range(timesteps):
                random_vals = torch.rand(len(activation), device=self.device)
                spike_mask = random_vals < activation
                spikes[t, start_idx:end_idx] = spike_mask.float()
        
        return spikes
    
    def _combined_encode(
        self,
        vector: torch.Tensor,
        timesteps: int
    ) -> torch.Tensor:
        """
        Encode vector using a combination of methods.
        
        Combines temporal precision with population coding principles.
        
        Args:
            vector: Normalized input vector
            timesteps: Number of time steps
            
        Returns:
            Spike pattern tensor of shape [timesteps, self.neuron_count]
        """
        # Ensure vector is at least 1D
        if vector.dim() == 0:
            vector = vector.unsqueeze(0)
        
        # Initialize spike patterns from different encoders
        rate_spikes = self._rate_encode(vector, timesteps)
        temporal_spikes = self._temporal_encode(vector, timesteps)
        
        # Weighted combination of encoding methods
        combined_spikes = 0.3 * rate_spikes + 0.7 * temporal_spikes
        
        # Convert to binary spikes using threshold
        binary_spikes = (combined_spikes > 0.5).float()
        
        return binary_spikes
    
    def encode_batch(
        self,
        vectors: Union[np.ndarray, torch.Tensor, List[List[float]]],
        timesteps: int = None
    ) -> torch.Tensor:
        """
        Encode a batch of vectors into spike patterns.
        
        Args:
            vectors: Batch of input vectors
            timesteps: Number of time steps
            
        Returns:
            Batch of spike patterns tensor of shape [batch_size, timesteps, neurons]
        """
        # Set default timesteps if not specified
        if timesteps is None:
            timesteps = self.temporal_window
        
        # Convert input to PyTorch tensor
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors).float()
        elif isinstance(vectors, list):
            vectors = torch.tensor(vectors, dtype=torch.float32)
        
        # Move tensor to the specified device
        vectors = vectors.to(self.device)
        
        # Get batch size
        batch_size = vectors.size(0)
        
        # Initialize output tensor
        spike_patterns = torch.zeros(batch_size, timesteps, self.neuron_count, device=self.device)
        
        # Encode each vector in the batch
        for i in range(batch_size):
            spike_patterns[i] = self.encode_vector(vectors[i], timesteps)
        
        return spike_patterns
    
    def visualize_spikes(
        self,
        spike_pattern: torch.Tensor,
        max_neurons: int = 50,
        title: str = "Spike Pattern Visualization"
    ):
        """
        Visualize spike pattern as a raster plot.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            max_neurons: Maximum number of neurons to display
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Move tensor to CPU and convert to numpy
            if isinstance(spike_pattern, torch.Tensor):
                spike_pattern = spike_pattern.cpu().numpy()
            
            # Get dimensions
            timesteps, neurons = spike_pattern.shape
            
            # Limit number of neurons to display
            neurons_to_show = min(neurons, max_neurons)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot spikes
            for n in range(neurons_to_show):
                spike_times = np.where(spike_pattern[:, n] > 0)[0]
                ax.scatter(spike_times, n * np.ones_like(spike_times), s=2, c='black')
            
            # Set labels and title
            ax.set_xlabel('Time step')
            ax.set_ylabel('Neuron index')
            ax.set_title(title)
            ax.set_xlim(-0.5, timesteps - 0.5)
            ax.set_ylim(-0.5, neurons_to_show - 0.5)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return None

# Utility function to create a spike encoder
def create_encoder(
    encoding_type="rate", 
    neuron_count=1000, 
    device=None, 
    sparse_output=True, 
    precision_level=3,
    model_type="generic"
):
    """
    Factory function to create a spike encoder with standard parameters.
    
    Args:
        encoding_type: Type of encoding ("rate", "temporal", "population", "combined")
        neuron_count: Number of neurons for the encoder
        device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
        sparse_output: Whether to use sparse tensor representation for efficiency
        precision_level: Level of precision for temporal encoding (1-5)
        model_type: Type of SNN model for specialized encoding
    
    Returns:
        Initialized SpikeEncoder instance
    """
    return SpikeEncoder(
        neuron_count=neuron_count,
        encoding_type=encoding_type,
        device=device,
        sparse_output=sparse_output,
        precision_level=precision_level,
        model_type=model_type
    )

# Simple demo/test function
def test_encoder():
    """Test the spike encoder with a simple example"""
    # Create a sample vector
    vector = np.random.rand(100)
    
    # Create encoder instances for different methods
    encoders = {
        "Rate": create_encoder("rate", 100),
        "Temporal": create_encoder("temporal", 100),
        "Population": create_encoder("population", 100),
        "Combined": create_encoder("combined", 100)
    }
    
    # Test each encoder
    for name, encoder in encoders.items():
        print(f"Testing {name} encoding...")
        
        # Encode vector
        spikes = encoder.encode_vector(vector, timesteps=20)
        
        # Print stats
        active_neurons = (spikes.sum(dim=0) > 0).sum().item()
        total_spikes = spikes.sum().item()
        
        print(f"  Input vector shape: {vector.shape}")
        print(f"  Output spikes shape: {spikes.shape}")
        print(f"  Active neurons: {active_neurons}/{encoder.neuron_count}")
        print(f"  Total spikes: {total_spikes}")
        print(f"  Average spikes per timestep: {total_spikes / spikes.shape[0]:.2f}")
        
        # Visualize (if matplotlib is available)
        fig = encoder.visualize_spikes(spikes, max_neurons=50, title=f"{name} Encoding Pattern")
        if fig:
            print("  Visualization created successfully")
    
    return encoders

if __name__ == "__main__":
    test_encoder() 