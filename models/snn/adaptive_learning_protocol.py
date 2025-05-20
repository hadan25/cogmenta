"""
Explicit Learning Protocol for Adaptive SNN Components

This module implements the learning protocol for adaptive components in the SNN framework.
It defines loss functions, optimization strategies, and mechanisms for freezing/unfreezing
adaptiveness based on performance metrics.

The protocol ensures that adaptive encoding/decoding layers are explicitly trained
to maximize information preservation while adapting to the statistical properties
of the data being processed.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adaptive_learning_protocol")


class LearningPhase(Enum):
    """Phases of adaptive learning."""
    DISABLED = 0       # Adaptiveness is disabled (frozen)
    PRETRAINING = 1    # Initial training phase
    JOINT = 2          # Joint training with other components
    FINE_TUNING = 3    # Targeted fine-tuning for specific tasks
    CONTINUOUS = 4     # Continuous adaptation during operation


class AdaptiveLossFunction(Enum):
    """Loss functions for adaptive component training."""
    MSE = "mse"                    # Mean squared error
    COSINE_SIMILARITY = "cosine"   # Cosine similarity (negative)
    INFORMATION_RETENTION = "info" # Information retention reward
    COMBINED = "combined"          # Weighted combination of multiple losses


class AdaptiveLearningProtocol:
    """
    Protocol for training adaptive components in SNN models.
    
    This class defines how adaptive layers should be trained, including:
    - Loss functions and optimization strategies
    - When and how adaptiveness is enabled/disabled
    - Parameter freezing/unfreezing mechanisms
    - Adaptation schedules
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        loss_function: Union[str, AdaptiveLossFunction] = AdaptiveLossFunction.COMBINED,
        initial_phase: Union[str, LearningPhase] = LearningPhase.PRETRAINING,
        adaptation_threshold: float = 0.95,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the adaptive learning protocol.
        
        Args:
            learning_rate: Learning rate for adaptive components
            loss_function: Loss function to use for training
            initial_phase: Initial learning phase
            adaptation_threshold: Performance threshold for adaptiveness
            loss_weights: Weights for combined loss function components
        """
        # Store parameters
        self.learning_rate = learning_rate
        
        # Set loss function
        if isinstance(loss_function, str):
            try:
                self.loss_function = AdaptiveLossFunction(loss_function)
            except ValueError:
                logger.warning(f"Unknown loss function: {loss_function}, using COMBINED")
                self.loss_function = AdaptiveLossFunction.COMBINED
        else:
            self.loss_function = loss_function
        
        # Set learning phase
        if isinstance(initial_phase, str):
            try:
                self.learning_phase = LearningPhase[initial_phase.upper()]
            except KeyError:
                logger.warning(f"Unknown learning phase: {initial_phase}, using PRETRAINING")
                self.learning_phase = LearningPhase.PRETRAINING
        else:
            self.learning_phase = initial_phase
        
        # Set adaptation threshold and loss weights
        self.adaptation_threshold = adaptation_threshold
        self.loss_weights = loss_weights or {
            "mse": 1.0,
            "cosine": 0.5,
            "info": 1.0
        }
        
        # Initialize training stats
        self.training_stats = {
            "iterations": 0,
            "loss_history": [],
            "performance_history": [],
            "phase_changes": []
        }
        
        logger.info(f"Initialized AdaptiveLearningProtocol with {self.loss_function.value} loss function")
        logger.info(f"Initial learning phase: {self.learning_phase.name}")
    
    def get_loss_function(self) -> Callable:
        """
        Get the appropriate loss function based on current settings.
        
        Returns:
            Callable loss function that takes original and reconstructed vectors
        """
        if self.loss_function == AdaptiveLossFunction.MSE:
            return self._mse_loss
        elif self.loss_function == AdaptiveLossFunction.COSINE_SIMILARITY:
            return self._cosine_loss
        elif self.loss_function == AdaptiveLossFunction.INFORMATION_RETENTION:
            return self._information_retention_loss
        else:  # COMBINED
            return self._combined_loss
    
    def _mse_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Mean squared error loss."""
        return torch.nn.functional.mse_loss(reconstructed, original)
    
    def _cosine_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity loss (higher similarity = lower loss)."""
        if len(original.shape) == 1:
            # For single vector
            cos_sim = torch.nn.functional.cosine_similarity(
                reconstructed.view(1, -1),
                original.view(1, -1)
            )
        else:
            # For batch of vectors
            cos_sim = torch.nn.functional.cosine_similarity(
                reconstructed,
                original
            )
        # Convert to loss (higher similarity = lower loss)
        return 1.0 - cos_sim.mean()
    
    def _information_retention_loss(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor,
        retention_target: float = 0.99
    ) -> torch.Tensor:
        """
        Information retention loss based on entropy and mutual information.
        
        This is a simplified approximation using cosine similarity as a proxy for
        information retention.
        """
        # Calculate cosine similarity
        cos_loss = self._cosine_loss(original, reconstructed)
        
        # Calculate retention gap penalty (more severe as we approach target)
        retention = 1.0 - cos_loss.item()  # Convert loss back to similarity
        retention_gap = max(0, retention_target - retention)
        retention_penalty = torch.tensor(retention_gap ** 2, device=original.device)
        
        return cos_loss + retention_penalty
    
    def _combined_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Weighted combination of multiple loss functions."""
        # Calculate individual losses
        mse = self._mse_loss(original, reconstructed)
        cosine = self._cosine_loss(original, reconstructed)
        
        # Calculate information retention loss if needed
        if self.loss_weights.get("info", 0) > 0:
            info_loss = self._information_retention_loss(original, reconstructed)
            # Combine all losses with weights
            return (
                self.loss_weights.get("mse", 1.0) * mse +
                self.loss_weights.get("cosine", 0.5) * cosine +
                self.loss_weights.get("info", 1.0) * info_loss
            )
        else:
            # Combine just MSE and cosine losses
            return (
                self.loss_weights.get("mse", 1.0) * mse +
                self.loss_weights.get("cosine", 0.5) * cosine
            )
    
    def should_update_parameters(self, performance_metric: float) -> bool:
        """
        Determine if adaptive parameters should be updated based on performance.
        
        Args:
            performance_metric: Current performance metric (e.g., cosine similarity)
            
        Returns:
            Boolean indicating whether parameters should be updated
        """
        # Always update during pretraining or fine-tuning
        if self.learning_phase in [LearningPhase.PRETRAINING, LearningPhase.FINE_TUNING]:
            return True
        
        # In joint training, update only if performance is below threshold
        elif self.learning_phase == LearningPhase.JOINT:
            return performance_metric < self.adaptation_threshold
        
        # In continuous mode, use adaptive strategy based on performance
        elif self.learning_phase == LearningPhase.CONTINUOUS:
            # Update more frequently when performance is lower
            update_probability = max(0, 1.0 - performance_metric)
            return np.random.random() < update_probability
        
        # Don't update if disabled
        else:
            return False
    
    def set_learning_phase(self, phase: Union[str, LearningPhase]):
        """
        Set the learning phase.
        
        Args:
            phase: New learning phase
        """
        old_phase = self.learning_phase
        
        # Convert string to enum if needed
        if isinstance(phase, str):
            try:
                self.learning_phase = LearningPhase[phase.upper()]
            except KeyError:
                logger.warning(f"Unknown learning phase: {phase}, keeping current phase")
                return
        else:
            self.learning_phase = phase
        
        # Record phase change
        self.training_stats["phase_changes"].append({
            "iteration": self.training_stats["iterations"],
            "from": old_phase.name,
            "to": self.learning_phase.name
        })
        
        logger.info(f"Changed learning phase from {old_phase.name} to {self.learning_phase.name}")
    
    def update_performance(self, loss: float, performance_metric: float):
        """
        Update training statistics with latest performance metrics.
        
        Args:
            loss: Current loss value
            performance_metric: Current performance metric
        """
        self.training_stats["iterations"] += 1
        self.training_stats["loss_history"].append(loss)
        self.training_stats["performance_history"].append(performance_metric)
    
    def get_optimizer_config(self, parameters) -> Dict:
        """
        Get optimizer configuration based on current learning phase.
        
        Args:
            parameters: Model parameters to optimize
            
        Returns:
            Dictionary with optimizer configuration
        """
        # Base learning rate varies by phase
        if self.learning_phase == LearningPhase.PRETRAINING:
            lr = self.learning_rate
        elif self.learning_phase == LearningPhase.JOINT:
            lr = self.learning_rate * 0.5
        elif self.learning_phase == LearningPhase.FINE_TUNING:
            lr = self.learning_rate * 0.1
        elif self.learning_phase == LearningPhase.CONTINUOUS:
            lr = self.learning_rate * 0.01
        else:  # DISABLED
            lr = 0.0
        
        # Create optimizer with phase-specific learning rate
        optimizer = torch.optim.Adam(parameters, lr=lr)
        
        return {
            "optimizer": optimizer,
            "learning_rate": lr,
            "phase": self.learning_phase.name
        }
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training statistics.
        
        Returns:
            Dictionary with training summary
        """
        # Calculate summary statistics if we have history
        if len(self.training_stats["loss_history"]) > 0:
            avg_loss = np.mean(self.training_stats["loss_history"][-100:])
            min_loss = np.min(self.training_stats["loss_history"])
            
            avg_performance = np.mean(self.training_stats["performance_history"][-100:])
            max_performance = np.max(self.training_stats["performance_history"])
        else:
            avg_loss = 0.0
            min_loss = 0.0
            avg_performance = 0.0
            max_performance = 0.0
        
        return {
            "iterations": self.training_stats["iterations"],
            "current_phase": self.learning_phase.name,
            "loss_function": self.loss_function.value,
            "avg_loss": avg_loss,
            "min_loss": min_loss,
            "avg_performance": avg_performance,
            "max_performance": max_performance,
            "phase_changes": self.training_stats["phase_changes"]
        }


def create_learning_protocol(
    learning_rate: float = 0.001,
    loss_function: str = "combined",
    initial_phase: str = "pretraining",
    adaptation_threshold: float = 0.95,
    loss_weights: Optional[Dict[str, float]] = None
) -> AdaptiveLearningProtocol:
    """
    Create an adaptive learning protocol with the specified configuration.
    
    Args:
        learning_rate: Learning rate for adaptive components
        loss_function: Loss function to use for training
        initial_phase: Initial learning phase
        adaptation_threshold: Performance threshold for adaptiveness
        loss_weights: Weights for combined loss function components
        
    Returns:
        Configured AdaptiveLearningProtocol instance
    """
    return AdaptiveLearningProtocol(
        learning_rate=learning_rate,
        loss_function=loss_function,
        initial_phase=initial_phase,
        adaptation_threshold=adaptation_threshold,
        loss_weights=loss_weights
    )


# Demo/test function
def test_learning_protocol():
    """Test the learning protocol with sample data."""
    # Create protocol
    protocol = create_learning_protocol()
    
    # Create sample tensors
    original = torch.rand(10, 300)
    reconstructed = original + 0.1 * torch.randn(10, 300)  # Add noise
    
    # Get loss function
    loss_fn = protocol.get_loss_function()
    
    # Calculate loss
    loss = loss_fn(original, reconstructed)
    
    # Calculate performance metric (cosine similarity)
    cosine_sim = torch.nn.functional.cosine_similarity(
        original, reconstructed
    ).mean().item()
    
    # Update performance metrics
    protocol.update_performance(loss.item(), cosine_sim)
    
    # Test phase changes
    for phase in ["pretraining", "joint", "fine_tuning", "continuous", "disabled"]:
        protocol.set_learning_phase(phase)
        should_update = protocol.should_update_parameters(cosine_sim)
        print(f"Phase: {phase}, Should update: {should_update}")
    
    # Get training summary
    summary = protocol.get_training_summary()
    print(f"Training summary: {summary}")
    
    return protocol


if __name__ == "__main__":
    test_learning_protocol() 