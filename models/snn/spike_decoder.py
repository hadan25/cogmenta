#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spike Decoder for SNN Models.

This module provides various decoding strategies to convert spike patterns
back into vector representations from Spiking Neural Networks.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpikeDecoder")

class SpikeDecoder:
    """
    Decoder for converting spike patterns back to vector representations.
    
    Features:
    - Rate-based decoding (spike counts → vector values)
    - Temporal decoding (spike timing → vector values)
    - Population decoding (group activity → vector values)
    - Ensemble decoding (combines multiple strategies)
    - GPU acceleration support
    """
    
    def __init__(
        self,
        neuron_count: int = 1000,
        decoding_type: str = "rate",
        temporal_window: int = 20,
        device: Optional[str] = None,
        precision_level: int = 3,
        model_type: str = "generic"
    ):
        """
        Initialize the spike decoder.
        
        Args:
            neuron_count: Number of neurons used in encoding
            decoding_type: Type of decoding ("rate", "temporal", "population", "ensemble")
            temporal_window: Number of time steps used in temporal encoding
            device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
            precision_level: Level of precision for temporal decoding (1-5)
            model_type: Type of SNN model for specialized decoding
        """
        # Set device for GPU acceleration
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Spike decoder using device: {self.device}")
        
        self.neuron_count = neuron_count
        self.decoding_type = decoding_type.lower()
        self.temporal_window = temporal_window
        self.precision_level = max(1, min(5, precision_level))  # Clamp between 1-5
        self.model_type = model_type.lower()
        
        # Set precision-dependent parameters
        self._set_precision_parameters()
        
        # Validate decoding type
        valid_types = ["rate", "temporal", "population", "ensemble"]
        if self.decoding_type not in valid_types:
            logger.warning(f"Unknown decoding type: {self.decoding_type}, falling back to rate-based decoding")
            self.decoding_type = "rate"
        
        # Initialize decoding parameters
        self._init_decoding_params()
    
    def _init_decoding_params(self):
        """Initialize decoding-specific parameters"""
        if self.decoding_type == "population":
            # For population decoding, create receptive field centers
            self.receptive_field_centers = torch.linspace(-1.0, 1.0, self.neuron_count).to(self.device)
    
    def _set_precision_parameters(self):
        """Configure parameters based on precision level (1-5)"""
        # Set temporal resolution based on precision level
        precision_config = {
            1: {"smoothing": 0.5, "noise_tolerance": 0.2, "threshold": 0.3},  # Low precision
            2: {"smoothing": 0.3, "noise_tolerance": 0.15, "threshold": 0.25},
            3: {"smoothing": 0.2, "noise_tolerance": 0.1, "threshold": 0.2},  # Medium/default
            4: {"smoothing": 0.1, "noise_tolerance": 0.05, "threshold": 0.15},
            5: {"smoothing": 0.05, "noise_tolerance": 0.02, "threshold": 0.1}  # High precision
        }
        
        config = precision_config.get(self.precision_level, precision_config[3])
        
        # Temporal smoothing factor for spike integration
        self.smoothing_factor = config["smoothing"]
        
        # Noise tolerance for filtering out noise spikes
        self.noise_tolerance = config["noise_tolerance"]
        
        # Activation threshold for spike detection
        self.activation_threshold = config["threshold"]
        
        # Adjust temporal window based on precision if needed
        if self.precision_level >= 4:
            # Higher precision may need larger window for decoding
            self.temporal_window = max(self.temporal_window, 25)
    
    def decode_spikes(
        self,
        spike_pattern: Union[np.ndarray, torch.Tensor],
        target_dim: int = 300
    ) -> torch.Tensor:
        """
        Decode spike pattern back to vector representation.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons] or sparse tensor
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector of shape [target_dim]
        """
        # Convert input to PyTorch tensor
        if isinstance(spike_pattern, np.ndarray):
            spike_pattern = torch.from_numpy(spike_pattern).float()
        
        # Handle sparse tensors
        if spike_pattern.is_sparse:
            # Convert sparse tensor to dense for processing
            spike_pattern = spike_pattern.to_dense()
        
        # Move tensor to the specified device
        spike_pattern = spike_pattern.to(self.device)
        
        # Choose decoding method
        if self.decoding_type == "rate":
            return self._rate_decode(spike_pattern, target_dim)
        elif self.decoding_type == "temporal":
            return self._temporal_decode(spike_pattern, target_dim)
        elif self.decoding_type == "population":
            return self._population_decode(spike_pattern, target_dim)
        elif self.decoding_type == "ensemble":
            return self._ensemble_decode(spike_pattern, target_dim)
        else:
            # Fallback to rate decoding
            return self._rate_decode(spike_pattern, target_dim)
    
    def _rate_decode(
        self,
        spike_pattern: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        Decode spike pattern using rate-based decoding.
        
        Higher spike counts lead to higher vector values.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector of shape [target_dim]
        """
        # Get dimensions
        timesteps, neurons = spike_pattern.shape
        
        # Apply precision-dependent processing
        if self.precision_level >= 3:
            # For higher precision, apply more sophisticated counting
            
            # 1. Apply noise filtering
            if self.noise_tolerance > 0:
                # Convert to float for filtering operations
                spike_pattern_float = spike_pattern.float()
                
                # Apply temporal smoothing
                filtered_pattern = spike_pattern_float.clone()
                
                # Temporal filtering (smoothing)
                kernel_size = max(2, min(5, 6 - self.precision_level))  # Smaller kernel for higher precision
                if kernel_size > 1 and timesteps > kernel_size:
                    smoothed = torch.zeros_like(filtered_pattern)
                    for t in range(timesteps):
                        start = max(0, t - kernel_size // 2)
                        end = min(timesteps, t + kernel_size // 2 + 1)
                        smoothed[t] = torch.mean(filtered_pattern[start:end], dim=0)
                    filtered_pattern = smoothed
                
                # Apply activation threshold to filter noise
                spike_mask = filtered_pattern > self.activation_threshold
                spike_pattern = spike_pattern.float() * spike_mask.float()
        
        # Count spikes for each neuron across all timesteps
        spike_counts = spike_pattern.sum(dim=0)  # Shape: [neurons]
        
        # Normalize to [0, 1] range based on precision
        if self.precision_level >= 4:
            # High precision: More linear normalization
            if spike_counts.max() > 0:
                normalized_values = spike_counts / min(timesteps, spike_counts.max() * 1.1)
            else:
                normalized_values = spike_counts
        else:
            # Lower precision: Standard normalization
            if spike_counts.max() > 0:
                normalized_values = spike_counts / timesteps
            else:
                normalized_values = spike_counts
        
        # Adjust vector length to target dimension
        if neurons > target_dim:
            # Average pooling to reduce dimension
            normalized_values = normalized_values.view(1, 1, neurons)
            output_vector = torch.nn.functional.avg_pool1d(
                normalized_values, 
                kernel_size=neurons // target_dim, 
                stride=neurons // target_dim
            ).view(target_dim)
        elif neurons < target_dim:
            # Padding or interpolation to increase dimension
            output_vector = torch.zeros(target_dim, device=self.device)
            # Copy available values
            output_vector[:neurons] = normalized_values
            # Interpolate remaining values if needed
            if neurons > 1:
                # Linear interpolation for remaining positions
                for i in range(neurons, target_dim):
                    ratio = (i - neurons) / (target_dim - neurons)
                    idx = int(ratio * (neurons - 1))
                    output_vector[i] = normalized_values[idx]
        else:
            # Same dimension, no adjustment needed
            output_vector = normalized_values
        
        return output_vector
    
    def _temporal_decode(
        self,
        spike_pattern: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        Decode spike pattern using temporal decoding.
        
        Earlier spikes lead to higher vector values.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector of shape [target_dim]
        """
        # Get dimensions
        timesteps, neurons = spike_pattern.shape
        
        # Initialize first spike time tensor with max timestep + 1 (no spike)
        first_spike_time = torch.full((neurons,), timesteps, device=self.device)
        
        # Apply precision-based noise filtering
        if self.precision_level >= 3:
            # For higher precision, apply noise filtering
            # Convert to float for filtering operations
            spike_pattern_float = spike_pattern.float()
            
            # Apply temporal smoothing based on precision
            if self.smoothing_factor > 0:
                smoothed_pattern = spike_pattern_float.clone()
                for t in range(1, timesteps):
                    smoothed_pattern[t] = (1 - self.smoothing_factor) * smoothed_pattern[t] + self.smoothing_factor * smoothed_pattern[t-1]
                
                # Only consider spikes above noise threshold
                spike_pattern = smoothed_pattern > self.noise_tolerance
        
        # Find first spike time for each neuron
        for t in range(timesteps):
            # Update first spike time for neurons that spike at this timestep
            # but haven't spiked before
            mask = (spike_pattern[t] > 0) & (first_spike_time == timesteps)
            first_spike_time[mask] = t
        
        # Convert to values (earlier spikes = higher values)
        normalized_values = torch.zeros(neurons, device=self.device)
        spiked_mask = first_spike_time < timesteps
        
        if spiked_mask.any():
            # Precision-dependent value calculation
            if self.precision_level >= 4:
                # High precision: More linear mapping from spike time to value
                normalized_values[spiked_mask] = 1.0 - (first_spike_time[spiked_mask] / (timesteps - 1))
            else:
                # Lower precision: Non-linear mapping for better noise tolerance
                normalized_values[spiked_mask] = torch.exp(-(first_spike_time[spiked_mask].float() / timesteps) * 3)
        
        # Adjust vector length to target dimension
        if neurons > target_dim:
            # Average pooling to reduce dimension
            # Reshape for 1D pooling operation
            normalized_values = normalized_values.view(1, 1, neurons)
            
            # Calculate pooling kernel size
            kernel_size = max(1, neurons // target_dim)
            
            # Perform pooling and reshape to target dimension
            pooled = torch.nn.functional.avg_pool1d(
                normalized_values, 
                kernel_size=kernel_size,
                stride=kernel_size
            )
            
            # Reshape and interpolate to exact target dimension
            pooled_size = pooled.shape[2]
            output_vector = torch.zeros(target_dim, device=self.device)
            
            # Copy and interpolate if needed
            for i in range(target_dim):
                src_idx = min(int(i * pooled_size / target_dim), pooled_size - 1)
                output_vector[i] = pooled[0, 0, src_idx]
                
        elif neurons < target_dim:
            # Padding or interpolation to increase dimension
            output_vector = torch.zeros(target_dim, device=self.device)
            # Copy available values
            output_vector[:neurons] = normalized_values
            # Interpolate remaining values if needed
            if neurons > 1:
                # Linear interpolation for remaining positions
                for i in range(neurons, target_dim):
                    ratio = (i - neurons) / (target_dim - neurons)
                    idx = int(ratio * (neurons - 1))
                    output_vector[i] = normalized_values[idx]
        else:
            # Same dimension, no adjustment needed
            output_vector = normalized_values
        
        return output_vector
    
    def _population_decode(
        self,
        spike_pattern: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        Decode spike pattern using population decoding.
        
        Estimates vector values based on the collective activity
        of neuron populations with different preferred values.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector of shape [target_dim]
        """
        # Get dimensions
        timesteps, neurons = spike_pattern.shape
        
        # Calculate total activity for each neuron
        neuron_activity = spike_pattern.sum(dim=0)  # Shape: [neurons]
        
        # Normalize activity if there are any spikes
        if neuron_activity.sum() > 0:
            neuron_activity = neuron_activity / neuron_activity.sum()
        
        # Determine dimension representation
        neurons_per_dim = max(1, neurons // target_dim)
        
        # Initialize output vector
        output_vector = torch.zeros(target_dim, device=self.device)
        
        # Process each dimension
        for d in range(min(target_dim, neurons // neurons_per_dim)):
            # Get receptive field centers for this dimension
            start_idx = d * neurons_per_dim
            end_idx = min(start_idx + neurons_per_dim, neurons)
            
            # Get neuron positions and activities for this dimension
            rf_centers = self.receptive_field_centers[start_idx:end_idx]
            activities = neuron_activity[start_idx:end_idx]
            
            # Skip if no activity in this dimension
            if activities.sum() == 0:
                continue
                
            # Weighted average of receptive field centers
            output_vector[d] = (rf_centers * activities).sum() / activities.sum()
        
        # Rescale from [-1, 1] to [0, 1]
        output_vector = (output_vector + 1.0) / 2.0
        
        # Fill any remaining dimensions with zeros or interpolated values
        if target_dim > neurons // neurons_per_dim:
            # Linear interpolation for remaining positions
            remaining_dims = range(neurons // neurons_per_dim, target_dim)
            if len(remaining_dims) > 0 and neurons // neurons_per_dim > 0:
                for i in remaining_dims:
                    ratio = (i - (neurons // neurons_per_dim - 1)) / (target_dim - (neurons // neurons_per_dim - 1))
                    idx = min(int(ratio * ((neurons // neurons_per_dim) - 1)), (neurons // neurons_per_dim) - 1)
                    output_vector[i] = output_vector[idx]
        
        return output_vector
    
    def _ensemble_decode(
        self,
        spike_pattern: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        Decode spike pattern using an ensemble of decoding methods.
        
        Combines rate and temporal decoding for a more robust result.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            target_dim: Target dimension for the output vector
            
        Returns:
            Decoded vector of shape [target_dim]
        """
        # Get results from different decoding methods
        rate_vector = self._rate_decode(spike_pattern, target_dim)
        temporal_vector = self._temporal_decode(spike_pattern, target_dim)
        
        # Weighted combination
        ensemble_vector = 0.6 * rate_vector + 0.4 * temporal_vector
        
        return ensemble_vector
    
    def decode_batch(
        self,
        spike_patterns: Union[np.ndarray, torch.Tensor],
        target_dim: int = 300
    ) -> torch.Tensor:
        """
        Decode a batch of spike patterns.
        
        Args:
            spike_patterns: Batch of spike patterns tensor of shape [batch_size, timesteps, neurons]
            target_dim: Target dimension for each output vector
            
        Returns:
            Batch of decoded vectors of shape [batch_size, target_dim]
        """
        # Convert input to PyTorch tensor
        if isinstance(spike_patterns, np.ndarray):
            spike_patterns = torch.from_numpy(spike_patterns).float()
        
        # Move tensor to the specified device
        spike_patterns = spike_patterns.to(self.device)
        
        # Get batch size
        batch_size = spike_patterns.size(0)
        
        # Initialize output tensor
        vectors = torch.zeros(batch_size, target_dim, device=self.device)
        
        # Decode each spike pattern in the batch
        for i in range(batch_size):
            vectors[i] = self.decode_spikes(spike_patterns[i], target_dim)
        
        return vectors
    
    def visualize_decoding(
        self,
        spike_pattern: torch.Tensor,
        original_vector: Optional[torch.Tensor] = None,
        target_dim: int = 10
    ):
        """
        Visualize the decoding process and compare with the original vector if provided.
        
        Args:
            spike_pattern: Spike pattern tensor of shape [timesteps, neurons]
            original_vector: Optional original vector for comparison
            target_dim: Target dimension for the decoded vector
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Decode spike pattern
            decoded_vector = self.decode_spikes(spike_pattern, target_dim)
            
            # Move tensors to CPU for plotting
            decoded_vector = decoded_vector.cpu().numpy()
            
            # Create figure
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot spike pattern
            spike_pattern_np = spike_pattern.cpu().numpy()
            axs[0].imshow(spike_pattern_np.T, aspect='auto', cmap='binary')
            axs[0].set_xlabel('Time step')
            axs[0].set_ylabel('Neuron index')
            axs[0].set_title('Spike Pattern')
            
            # Plot decoded vector
            axs[1].bar(range(target_dim), decoded_vector, alpha=0.7, label='Decoded')
            
            # Plot original vector if provided
            if original_vector is not None:
                original_vector = original_vector.cpu().numpy()
                # Ensure original vector has the right dimension
                if len(original_vector) != target_dim:
                    logger.warning(f"Original vector dimension ({len(original_vector)}) doesn't match target_dim ({target_dim})")
                    if len(original_vector) > target_dim:
                        original_vector = original_vector[:target_dim]
                    else:
                        temp = np.zeros(target_dim)
                        temp[:len(original_vector)] = original_vector
                        original_vector = temp
                
                axs[1].plot(range(target_dim), original_vector, 'ro-', alpha=0.7, label='Original')
                axs[1].legend()
            
            axs[1].set_xlabel('Dimension')
            axs[1].set_ylabel('Value')
            axs[1].set_title('Decoded Vector')
            axs[1].set_ylim(0, 1)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return None

def create_decoder(
    decoding_type="rate", 
    neuron_count=1000, 
    device=None,
    precision_level=3,
    model_type="generic"
):
    """
    Factory function to create a spike decoder with standard parameters.
    
    Args:
        decoding_type: Type of decoding ("rate", "temporal", "population", "ensemble")
        neuron_count: Number of neurons for the decoder
        device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
        precision_level: Level of precision for temporal decoding (1-5)
        model_type: Type of SNN model for specialized decoding
    
    Returns:
        Initialized SpikeDecoder instance
    """
    return SpikeDecoder(
        neuron_count=neuron_count,
        decoding_type=decoding_type,
        device=device,
        precision_level=precision_level,
        model_type=model_type
    )

# Simple demo/test function
def test_decoder():
    """Test the spike decoder with a simple example"""
    import torch
    
    # Create a sample vector
    original_vector = torch.rand(10)
    
    # Create a synthetic spike pattern (normally this would come from an encoder)
    timesteps = 20
    neurons = 100
    spike_pattern = torch.zeros(timesteps, neurons)
    
    # Generate spikes - higher value in original vector means more spikes
    # in the corresponding section of neurons
    neurons_per_dim = neurons // 10
    for i, val in enumerate(original_vector):
        start_idx = i * neurons_per_dim
        end_idx = start_idx + neurons_per_dim
        # Generate spikes with probability proportional to value
        for t in range(timesteps):
            spike_mask = torch.rand(neurons_per_dim) < val
            spike_pattern[t, start_idx:end_idx] = spike_mask.float()
    
    # Create decoder instances for different methods
    decoders = {
        "Rate": create_decoder("rate", neurons),
        "Temporal": create_decoder("temporal", neurons),
        "Population": create_decoder("population", neurons),
        "Ensemble": create_decoder("ensemble", neurons)
    }
    
    # Test each decoder
    for name, decoder in decoders.items():
        print(f"Testing {name} decoding...")
        
        # Decode spike pattern
        decoded = decoder.decode_spikes(spike_pattern, target_dim=10)
        
        # Calculate similarity/error metrics
        mse = torch.mean((decoded - original_vector) ** 2).item()
        correlation = torch.corrcoef(torch.stack([decoded, original_vector]))[0, 1].item()
        
        print(f"  Original vector: {original_vector}")
        print(f"  Decoded vector:  {decoded}")
        print(f"  Mean squared error: {mse:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        
        # Visualize (if matplotlib is available)
        fig = decoder.visualize_decoding(spike_pattern, original_vector, target_dim=10)
        if fig:
            print("  Visualization created successfully")
    
    return decoders

if __name__ == "__main__":
    test_decoder() 