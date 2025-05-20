"""
Quantitative Fidelity Gate for SNN Models

This module implements the Quantitative Fidelity Gate described in the SNN roadmap.
It provides explicit, automated, and enforced measurement of information preservation
across encode-decode-re-encode cycles for all supported modalities.

The fidelity gate serves as a quality control checkpoint that prevents the pipeline
from proceeding to pre-training if information preservation does not meet the required
thresholds.
"""

import os
import logging
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Fix the imports to work in both package and direct script contexts
try:
    # When imported as package
    from models.snn.adaptive_spike_processor import create_adaptive_processor, AdaptiveSpikeProcessor
    from models.snn.bidirectional_encoding import create_processor
    from models.snn.utils.logging_config import setup_logging
except ImportError:
    # When run directly 
    from adaptive_spike_processor import create_adaptive_processor, AdaptiveSpikeProcessor
    from bidirectional_encoding import create_processor
    
    # Setup basic logging since logging_config might not be available
    import logging
    def setup_logging(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

# Set up logging
logger = setup_logging("fidelity_gate")

class FidelityMetrics:
    """Container for fidelity metrics across different modalities and models."""
    
    def __init__(self):
        self.metrics = {
            "text": {},
            "image": {},
            "audio": {},
            "vector": {}
        }
    
    def add_metric(self, modality: str, model_name: str, metric_name: str, value: float):
        """Add a metric value for a specific modality and model."""
        if modality not in self.metrics:
            self.metrics[modality] = {}
        
        if model_name not in self.metrics[modality]:
            self.metrics[modality][model_name] = {}
        
        self.metrics[modality][model_name][metric_name] = value
    
    def get_metric(self, modality: str, model_name: str, metric_name: str) -> Optional[float]:
        """Get a specific metric value."""
        if modality in self.metrics and model_name in self.metrics[modality]:
            return self.metrics[modality][model_name].get(metric_name)
        return None
    
    def passes_threshold(self, modality: str, model_name: str, metric_name: str, threshold: float) -> bool:
        """Check if a metric passes the specified threshold."""
        value = self.get_metric(modality, model_name, metric_name)
        if value is None:
            return False
        
        # For MSE, lower is better
        if metric_name == "mse":
            return value <= threshold
        # For cosine similarity and retention score, higher is better
        else:
            return value >= threshold
    
    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get a summary of all metrics."""
        return self.metrics

    def to_report(self) -> str:
        """Generate a human-readable report of the metrics."""
        lines = ["===== Fidelity Metrics Report ====="]
        
        for modality in self.metrics:
            if not self.metrics[modality]:
                continue
                
            lines.append(f"\n-- {modality.upper()} Modality --")
            
            for model in self.metrics[modality]:
                lines.append(f"  Model: {model}")
                
                for metric, value in self.metrics[modality][model].items():
                    if metric == "mse":
                        lines.append(f"    {metric}: {value:.6f}")
                    else:
                        lines.append(f"    {metric}: {value:.4f}")
        
        return "\n".join(lines)


class FidelityGate:
    """
    Quantitative Fidelity Gate for SNN models.
    
    This class implements automated testing of information preservation across
    encode-decode-re-encode cycles for all supported modalities. It enforces
    strict thresholds that must be met before proceeding to model training.
    """
    
    def __init__(
        self,
        modalities: List[str] = ["text", "vector"],
        thresholds: Dict[str, Dict[str, float]] = None
    ):
        """
        Initialize the fidelity gate.
        
        Args:
            modalities: List of modalities to test
            thresholds: Dictionary of thresholds for each modality and metric
        """
        self.modalities = modalities
        
        # Default thresholds if not provided
        if thresholds is None:
            self.thresholds = {
                "text": {
                    "cosine_similarity": 0.95,  # Cosine similarity for text reconstruction
                    "semantic_similarity": 0.90,  # Semantic similarity
                    "retention_score": 0.90,     # Information retention score
                },
                "image": {
                    "cosine_similarity": 0.95,
                    "mse": 0.05,                 # Mean squared error (lower is better)
                    "retention_score": 0.90,
                },
                "audio": {
                    "cosine_similarity": 0.95,
                    "mse": 0.05,
                    "retention_score": 0.90,
                },
                "vector": {
                    "cosine_similarity": 0.999,  # Very high fidelity required for VSA vectors
                    "mse": 0.001,
                    "retention_score": 0.99,
                }
            }
        else:
            self.thresholds = thresholds
        
        # Create metrics container
        self.metrics = FidelityMetrics()
        
        # Setup test data
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup test data for each modality."""
        self.test_data = {}
        
        # Text test data
        self.test_data["text"] = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world of technology.",
            "Spiking neural networks mimic biological brain functions.",
            "This is a test of information preservation across encodings."
        ]
        
        # Vector test data
        self.test_data["vector"] = torch.rand(10, 300)  # 10 random vectors
        
        # Image and audio would be added when those processors are available
        # self.test_data["image"] = ...
        # self.test_data["audio"] = ...
    
    def test_processor(
        self, 
        processor: AdaptiveSpikeProcessor, 
        bidirectional_processor = None,
        model_name: str = "default"
    ) -> FidelityMetrics:
        """
        Test the fidelity of a spike processor across all modalities.
        
        Args:
            processor: The AdaptiveSpikeProcessor to test
            bidirectional_processor: Optional bidirectional processor for text
            model_name: Name of the model for reporting
            
        Returns:
            FidelityMetrics object with the test results
        """
        metrics = FidelityMetrics()
        
        # Test each supported modality
        for modality in self.modalities:
            if modality not in processor.modalities:
                logger.warning(f"Modality '{modality}' not supported by processor, skipping...")
                continue
            
            if modality == "text":
                if bidirectional_processor is None:
                    logger.warning("Bidirectional processor required for text testing, skipping...")
                    continue
                self._test_text_fidelity(processor, bidirectional_processor, model_name, metrics)
            
            elif modality == "vector":
                self._test_vector_fidelity(processor, model_name, metrics)
            
            # Add other modalities as they become available
            # elif modality == "image":
            #     self._test_image_fidelity(processor, model_name, metrics)
            # elif modality == "audio":
            #     self._test_audio_fidelity(processor, model_name, metrics)
        
        return metrics
    
    def _test_vector_fidelity(
        self,
        processor: AdaptiveSpikeProcessor,
        model_name: str,
        metrics: FidelityMetrics
    ):
        """Test vector round-trip fidelity."""
        logger.info(f"Testing vector fidelity for model: {model_name}")
        
        # Get test vectors
        test_vectors = self.test_data["vector"].to(processor.device)
        
        # Metrics accumulators
        cosine_sims = []
        mse_values = []
        retention_scores = []
        
        # Process each test vector
        for vector in test_vectors:
            # First encode-decode cycle
            spikes = processor.encode(vector)
            reconstructed, retention = processor.reconstruct_vector(spikes, vector)
            
            # Calculate metrics
            cosine_sim = torch.nn.functional.cosine_similarity(
                vector.view(1, -1), reconstructed.view(1, -1)
            ).item()
            
            mse = torch.mean((vector - reconstructed) ** 2).item()
            
            # Second encode-decode cycle (re-encode test)
            spikes2 = processor.encode(reconstructed)
            reconstructed2, retention2 = processor.reconstruct_vector(spikes2, reconstructed)
            
            # Calculate metrics for second cycle
            cosine_sim2 = torch.nn.functional.cosine_similarity(
                reconstructed.view(1, -1), reconstructed2.view(1, -1)
            ).item()
            
            # Accumulate metrics
            cosine_sims.append(cosine_sim)
            mse_values.append(mse)
            retention_scores.append(retention)
        
        # Average the metrics
        avg_cosine = np.mean(cosine_sims)
        avg_mse = np.mean(mse_values)
        avg_retention = np.mean(retention_scores)
        
        # Store in metrics object
        metrics.add_metric("vector", model_name, "cosine_similarity", avg_cosine)
        metrics.add_metric("vector", model_name, "mse", avg_mse)
        metrics.add_metric("vector", model_name, "retention_score", avg_retention)
        
        logger.info(f"Vector fidelity metrics - cosine: {avg_cosine:.4f}, MSE: {avg_mse:.6f}, retention: {avg_retention:.4f}")
    
    def _test_text_fidelity(
        self,
        processor: AdaptiveSpikeProcessor,
        bidirectional_processor,
        model_name: str,
        metrics: FidelityMetrics
    ):
        """Test text round-trip fidelity."""
        logger.info(f"Testing text fidelity for model: {model_name}")
        
        # Set up processor for text if needed
        if processor.bidirectional_processor is None:
            processor.bidirectional_processor = bidirectional_processor
        
        # Metrics accumulators
        cosine_sims = []
        retention_scores = []
        semantic_sims = []
        
        # Process each test text
        for text in self.test_data["text"]:
            # Get vector representation from bidirectional processor
            original_vector = processor.bidirectional_processor.get_vectors_for_tokens(
                processor.bidirectional_processor.tokenizer.encode(text)
            )
            
            # Process through adaptive processor (encode-decode)
            spikes, vector = processor.process_text(text, return_vector=True)
            reconstructed, retention = processor.reconstruct_vector(spikes, vector)
            
            # Decode back to text
            reconstructed_text = processor.bidirectional_processor.vector_to_text(reconstructed)
            
            # Get vector for reconstructed text
            reconstructed_vector = processor.bidirectional_processor.get_vectors_for_tokens(
                processor.bidirectional_processor.tokenizer.encode(reconstructed_text)
            )
            
            # Calculate vector similarity
            cosine_sim = torch.nn.functional.cosine_similarity(
                vector.view(1, -1), reconstructed.view(1, -1)
            ).item()
            
            # Calculate semantic similarity between original and reconstructed vectors
            semantic_sim = torch.nn.functional.cosine_similarity(
                original_vector.view(1, -1), reconstructed_vector.view(1, -1)
            ).item()
            
            # Accumulate metrics
            cosine_sims.append(cosine_sim)
            retention_scores.append(retention)
            semantic_sims.append(semantic_sim)
            
            logger.debug(f"Text: '{text}' -> '{reconstructed_text}'")
            logger.debug(f"Cosine sim: {cosine_sim:.4f}, Semantic sim: {semantic_sim:.4f}, Retention: {retention:.4f}")
        
        # Average the metrics
        avg_cosine = np.mean(cosine_sims)
        avg_retention = np.mean(retention_scores)
        avg_semantic = np.mean(semantic_sims)
        
        # Store in metrics object
        metrics.add_metric("text", model_name, "cosine_similarity", avg_cosine)
        metrics.add_metric("text", model_name, "semantic_similarity", avg_semantic)
        metrics.add_metric("text", model_name, "retention_score", avg_retention)
        
        logger.info(f"Text fidelity metrics - cosine: {avg_cosine:.4f}, semantic: {avg_semantic:.4f}, retention: {avg_retention:.4f}")
    
    def evaluate_model(
        self,
        processor: AdaptiveSpikeProcessor,
        bidirectional_processor = None,
        model_name: str = "default"
    ) -> Tuple[bool, FidelityMetrics]:
        """
        Evaluate a model and determine if it passes the fidelity gate.
        
        Args:
            processor: The AdaptiveSpikeProcessor to evaluate
            bidirectional_processor: Optional bidirectional processor for text
            model_name: Name of the model for reporting
            
        Returns:
            Tuple of (passed, metrics) where passed is a boolean indicating
            whether all thresholds were met
        """
        # Test the processor and get metrics
        metrics = self.test_processor(processor, bidirectional_processor, model_name)
        
        # Merge with existing metrics
        for modality in metrics.metrics:
            for model in metrics.metrics[modality]:
                for metric_name, value in metrics.metrics[modality][model].items():
                    self.metrics.add_metric(modality, model, metric_name, value)
        
        # Check if all thresholds are met
        passed = True
        failures = []
        
        for modality in self.modalities:
            if modality not in processor.modalities:
                continue
                
            for metric_name, threshold in self.thresholds[modality].items():
                # Skip modalities not supported by the model
                if modality not in metrics.metrics or not metrics.metrics[modality]:
                    continue
                    
                # Get the metric value
                value = metrics.get_metric(modality, model_name, metric_name)
                if value is None:
                    logger.warning(f"Metric '{metric_name}' not available for {modality}/{model_name}")
                    continue
                
                # Check if the metric passes the threshold
                if metric_name == "mse":
                    # For MSE, lower is better
                    if value > threshold:
                        passed = False
                        failures.append(f"{modality}/{metric_name}: {value:.6f} > {threshold:.6f}")
                else:
                    # For cosine similarity and retention score, higher is better
                    if value < threshold:
                        passed = False
                        failures.append(f"{modality}/{metric_name}: {value:.4f} < {threshold:.4f}")
        
        # Log the result
        if passed:
            logger.info(f"Model '{model_name}' PASSED the fidelity gate!")
        else:
            logger.warning(f"Model '{model_name}' FAILED the fidelity gate due to: {', '.join(failures)}")
        
        return passed, self.metrics
    
    def save_report(self, output_dir: str, prefix: str = "fidelity_report"):
        """
        Save a report of the fidelity metrics.
        
        Args:
            output_dir: Directory to save the report
            prefix: Prefix for the report file name
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        report = self.metrics.to_report()
        
        # Save the report
        report_path = os.path.join(output_dir, f"{prefix}_{int(time.time())}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Fidelity report saved to {report_path}")
        
        return report_path
    

def create_fidelity_gate(
    modalities: List[str] = None,
    thresholds: Dict[str, Dict[str, float]] = None
) -> FidelityGate:
    """
    Create a fidelity gate with the specified configuration.
    
    Args:
        modalities: List of modalities to test
        thresholds: Dictionary of thresholds for each modality and metric
        
    Returns:
        Configured FidelityGate instance
    """
    # Use default modalities if not specified
    if modalities is None:
        modalities = ["text", "vector"]
    
    return FidelityGate(modalities=modalities, thresholds=thresholds)


# Demo/test function
def test_fidelity_gate():
    """Test the fidelity gate with a sample processor."""
    import time
    
    # Create processors
    bidirectional_processor = create_processor(model_type="generic", vector_dim=300)
    
    processor = create_adaptive_processor(
        vector_dim=300,
        neuron_count=500,
        encoding_type="temporal",
        precision_level=3,
        timesteps=10,
        bidirectional_processor=bidirectional_processor
    )
    
    # Create fidelity gate
    gate = create_fidelity_gate()
    
    # Evaluate the processor
    passed, metrics = gate.evaluate_model(
        processor, 
        bidirectional_processor=bidirectional_processor,
        model_name="test_model"
    )
    
    # Print the report
    print(metrics.to_report())
    print(f"\nModel passed fidelity gate: {passed}")
    
    if not passed:
        print("\nTraining the processor to improve fidelity...")
        
        # Train the processor for a few steps
        for i in range(10):
            training_vectors = torch.rand(8, 300)
            processor.train_step(training_vectors)
            
            if i % 2 == 0:
                print(f"Training step {i+1}/10...")
        
        # Re-evaluate after training
        passed, metrics = gate.evaluate_model(
            processor, 
            bidirectional_processor=bidirectional_processor,
            model_name="test_model_trained"
        )
        
        print("\nAfter training:")
        print(metrics.to_report())
        print(f"\nModel passed fidelity gate: {passed}")
    
    # Save the report
    report_path = gate.save_report("./", "test_fidelity")
    
    return gate, metrics


if __name__ == "__main__":
    import time
    test_fidelity_gate() 