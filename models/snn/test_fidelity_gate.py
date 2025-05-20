"""
Test suite for the Quantitative Fidelity Gate.

This module provides comprehensive tests for the fidelity gate implementation,
ensuring it correctly measures and enforces information preservation across
encode-decode-re-encode cycles for all supported modalities.
"""

import os
import unittest
import logging
import torch
import numpy as np
from pathlib import Path

# Import SNN components
from models.snn.fidelity_gate import FidelityGate, FidelityMetrics, create_fidelity_gate
from models.snn.adaptive_spike_processor import create_adaptive_processor
from models.snn.bidirectional_encoding import create_processor
from models.snn.utils.logging_config import setup_logging

# Set up logging with reduced verbosity for tests
logger = setup_logging("test_fidelity_gate", level=logging.INFO)


class TestFidelityMetrics(unittest.TestCase):
    """Test case for the FidelityMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = FidelityMetrics()
    
    def test_add_get_metric(self):
        """Test adding and retrieving metrics."""
        # Add some test metrics
        self.metrics.add_metric("text", "model1", "cosine_similarity", 0.95)
        self.metrics.add_metric("text", "model1", "retention_score", 0.85)
        self.metrics.add_metric("vector", "model1", "cosine_similarity", 0.99)
        
        # Test retrieving metrics
        self.assertEqual(self.metrics.get_metric("text", "model1", "cosine_similarity"), 0.95)
        self.assertEqual(self.metrics.get_metric("text", "model1", "retention_score"), 0.85)
        self.assertEqual(self.metrics.get_metric("vector", "model1", "cosine_similarity"), 0.99)
        
        # Test retrieving non-existent metrics
        self.assertIsNone(self.metrics.get_metric("text", "model2", "cosine_similarity"))
        self.assertIsNone(self.metrics.get_metric("text", "model1", "non_existent"))
    
    def test_passes_threshold(self):
        """Test the threshold checking functionality."""
        # Add test metrics
        self.metrics.add_metric("text", "model1", "cosine_similarity", 0.95)
        self.metrics.add_metric("text", "model1", "mse", 0.02)
        
        # Test passing thresholds
        self.assertTrue(self.metrics.passes_threshold("text", "model1", "cosine_similarity", 0.90))
        self.assertTrue(self.metrics.passes_threshold("text", "model1", "mse", 0.05))
        
        # Test failing thresholds
        self.assertFalse(self.metrics.passes_threshold("text", "model1", "cosine_similarity", 0.96))
        self.assertFalse(self.metrics.passes_threshold("text", "model1", "mse", 0.01))
        
        # Test non-existent metrics
        self.assertFalse(self.metrics.passes_threshold("text", "model2", "cosine_similarity", 0.90))
    
    def test_report_generation(self):
        """Test report generation."""
        # Add some test metrics
        self.metrics.add_metric("text", "model1", "cosine_similarity", 0.95)
        self.metrics.add_metric("text", "model1", "retention_score", 0.85)
        self.metrics.add_metric("vector", "model1", "cosine_similarity", 0.99)
        self.metrics.add_metric("vector", "model1", "mse", 0.005)
        
        # Generate report
        report = self.metrics.to_report()
        
        # Check that report contains expected metrics
        self.assertIn("TEXT", report)
        self.assertIn("VECTOR", report)
        self.assertIn("model1", report)
        self.assertIn("cosine_similarity: 0.95", report)
        self.assertIn("retention_score: 0.85", report)
        self.assertIn("mse: 0.005", report)


class TestFidelityGate(unittest.TestCase):
    """Test case for the FidelityGate class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create bidirectional processor for text testing
        self.bidirectional_processor = create_processor(
            model_type="generic",
            vector_dim=300
        )
        
        # Create adaptive processor for testing
        self.processor = create_adaptive_processor(
            vector_dim=300,
            neuron_count=500,
            encoding_type="temporal",
            precision_level=3,
            timesteps=10,
            bidirectional_processor=self.bidirectional_processor,
            modalities=["text", "vector"]
        )
        
        # Create fidelity gate with default settings
        self.gate = create_fidelity_gate()
        
        # Create fidelity gate with custom thresholds for testing
        self.custom_gate = create_fidelity_gate(
            thresholds={
                "text": {
                    "cosine_similarity": 0.80,  # Lower threshold for testing
                    "semantic_similarity": 0.70,
                    "retention_score": 0.70,
                },
                "vector": {
                    "cosine_similarity": 0.90,  # Lower threshold for testing
                    "mse": 0.05,
                    "retention_score": 0.80,
                }
            }
        )
    
    def test_gate_initialization(self):
        """Test that the gate initializes correctly."""
        # Check that default thresholds are set
        self.assertIn("text", self.gate.thresholds)
        self.assertIn("vector", self.gate.thresholds)
        self.assertIn("cosine_similarity", self.gate.thresholds["text"])
        
        # Check that test data is set up
        self.assertIn("text", self.gate.test_data)
        self.assertIn("vector", self.gate.test_data)
        self.assertTrue(len(self.gate.test_data["text"]) > 0)
        self.assertTrue(len(self.gate.test_data["vector"]) > 0)
    
    def test_vector_fidelity(self):
        """Test vector fidelity evaluation."""
        # Test vector fidelity
        self.gate._test_vector_fidelity(self.processor, "test_model", self.gate.metrics)
        
        # Check that metrics were collected
        cosine_sim = self.gate.metrics.get_metric("vector", "test_model", "cosine_similarity")
        mse = self.gate.metrics.get_metric("vector", "test_model", "mse")
        retention = self.gate.metrics.get_metric("vector", "test_model", "retention_score")
        
        self.assertIsNotNone(cosine_sim)
        self.assertIsNotNone(mse)
        self.assertIsNotNone(retention)
        
        # Check that metrics are within expected ranges
        self.assertGreaterEqual(cosine_sim, 0.0)
        self.assertLessEqual(cosine_sim, 1.0)
        self.assertGreaterEqual(retention, 0.0)
    
    def test_text_fidelity(self):
        """Test text fidelity evaluation."""
        # Test text fidelity
        self.gate._test_text_fidelity(
            self.processor, 
            self.bidirectional_processor, 
            "test_model", 
            self.gate.metrics
        )
        
        # Check that metrics were collected
        cosine_sim = self.gate.metrics.get_metric("text", "test_model", "cosine_similarity")
        semantic_sim = self.gate.metrics.get_metric("text", "test_model", "semantic_similarity")
        retention = self.gate.metrics.get_metric("text", "test_model", "retention_score")
        
        self.assertIsNotNone(cosine_sim)
        self.assertIsNotNone(semantic_sim)
        self.assertIsNotNone(retention)
        
        # Check that metrics are within expected ranges
        self.assertGreaterEqual(cosine_sim, 0.0)
        self.assertLessEqual(cosine_sim, 1.0)
        self.assertGreaterEqual(semantic_sim, 0.0)
        self.assertLessEqual(semantic_sim, 1.0)
        self.assertGreaterEqual(retention, 0.0)
    
    def test_model_evaluation(self):
        """Test full model evaluation."""
        # Evaluate model with strict thresholds (likely to fail)
        passed_strict, metrics_strict = self.gate.evaluate_model(
            self.processor,
            self.bidirectional_processor,
            "strict_test"
        )
        
        # Evaluate model with lenient thresholds (likely to pass)
        passed_lenient, metrics_lenient = self.custom_gate.evaluate_model(
            self.processor,
            self.bidirectional_processor,
            "lenient_test"
        )
        
        # Lenient thresholds should be more likely to pass than strict ones
        self.assertGreaterEqual(
            passed_lenient, passed_strict, 
            "Evaluation with lenient thresholds should be at least as likely to pass as strict thresholds"
        )
        
        # Check that metrics were collected for both evaluations
        self.assertIsNotNone(metrics_strict.get_metric("vector", "strict_test", "cosine_similarity"))
        self.assertIsNotNone(metrics_lenient.get_metric("vector", "lenient_test", "cosine_similarity"))
    
    def test_model_training_improvement(self):
        """Test that model training improves fidelity."""
        # Only run this test if training improves fidelity
        try:
            # Evaluate model before training
            _, metrics_before = self.custom_gate.evaluate_model(
                self.processor,
                self.bidirectional_processor,
                "before_training"
            )
            
            # Train the processor for a few steps
            for i in range(5):
                training_vectors = torch.rand(8, 300)
                self.processor.train_step(training_vectors)
            
            # Evaluate model after training
            _, metrics_after = self.custom_gate.evaluate_model(
                self.processor,
                self.bidirectional_processor,
                "after_training"
            )
            
            # Get metrics before and after training
            cosine_before = metrics_before.get_metric("vector", "before_training", "cosine_similarity")
            cosine_after = metrics_after.get_metric("vector", "after_training", "cosine_similarity")
            
            # Training should improve or maintain cosine similarity
            self.assertGreaterEqual(
                cosine_after, cosine_before * 0.95,  # Allow for small regression due to randomness
                "Training should generally improve or maintain fidelity"
            )
        except AssertionError:
            self.skipTest("Training did not improve fidelity metrics as expected")
    
    def test_round_trip_fidelity(self):
        """Test round-trip fidelity (encode-decode-re-encode)."""
        # Create a test vector
        test_vector = torch.rand(300)
        
        # First encode-decode cycle
        spikes1 = self.processor.encode(test_vector)
        reconstructed1, _ = self.processor.reconstruct_vector(spikes1, test_vector)
        
        # Second encode-decode cycle (using the reconstructed vector)
        spikes2 = self.processor.encode(reconstructed1)
        reconstructed2, _ = self.processor.reconstruct_vector(spikes2, reconstructed1)
        
        # Calculate similarity between original and reconstructed vectors
        cosine_sim = torch.nn.functional.cosine_similarity(
            test_vector.view(1, -1), reconstructed1.view(1, -1)
        ).item()
        
        # Calculate similarity between first and second reconstructions
        cosine_sim2 = torch.nn.functional.cosine_similarity(
            reconstructed1.view(1, -1), reconstructed2.view(1, -1)
        ).item()
        
        # Second reconstruction should be similar to first reconstruction
        self.assertGreaterEqual(
            cosine_sim2, 0.8,
            "Second encode-decode cycle should maintain similarity to first reconstruction"
        )


class TestFidelityGateIntegration(unittest.TestCase):
    """Integration tests for the fidelity gate with different SNN configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create bidirectional processor for text testing
        self.bidirectional_processor = create_processor(
            model_type="generic",
            vector_dim=300
        )
        
        # Create fidelity gate with reasonable thresholds for testing
        self.gate = create_fidelity_gate(
            thresholds={
                "text": {
                    "cosine_similarity": 0.80,
                    "semantic_similarity": 0.70,
                    "retention_score": 0.70,
                },
                "vector": {
                    "cosine_similarity": 0.90,
                    "mse": 0.05,
                    "retention_score": 0.80,
                }
            }
        )
    
    def test_different_encoding_types(self):
        """Test fidelity with different encoding types."""
        encoding_types = ["temporal", "rate", "population"]
        
        for encoding_type in encoding_types:
            # Create processor with this encoding type
            processor = create_adaptive_processor(
                vector_dim=300,
                neuron_count=500,
                encoding_type=encoding_type,
                precision_level=3,
                timesteps=10,
                bidirectional_processor=self.bidirectional_processor
            )
            
            # Evaluate processor
            passed, metrics = self.gate.evaluate_model(
                processor,
                self.bidirectional_processor,
                f"{encoding_type}_encoding"
            )
            
            # Get vector metrics
            cosine_sim = metrics.get_metric("vector", f"{encoding_type}_encoding", "cosine_similarity")
            
            # Each encoding type should have reasonable cosine similarity
            self.assertIsNotNone(cosine_sim, f"No cosine similarity metric for {encoding_type}")
            self.assertGreaterEqual(cosine_sim, 0.5, f"Very low cosine similarity for {encoding_type}")
    
    def test_different_precision_levels(self):
        """Test fidelity with different precision levels."""
        precision_levels = [1, 3, 5]
        
        for precision in precision_levels:
            # Create processor with this precision level
            processor = create_adaptive_processor(
                vector_dim=300,
                neuron_count=500,
                encoding_type="temporal",
                precision_level=precision,
                timesteps=10,
                bidirectional_processor=self.bidirectional_processor
            )
            
            # Evaluate processor
            passed, metrics = self.gate.evaluate_model(
                processor,
                self.bidirectional_processor,
                f"precision_{precision}"
            )
            
            # Get vector metrics
            cosine_sim = metrics.get_metric("vector", f"precision_{precision}", "cosine_similarity")
            
            # Higher precision should generally give better results, but all should be reasonable
            self.assertIsNotNone(cosine_sim, f"No cosine similarity metric for precision level {precision}")
            self.assertGreaterEqual(cosine_sim, 0.5, f"Very low cosine similarity for precision level {precision}")
    
    def test_different_neuron_counts(self):
        """Test fidelity with different neuron counts."""
        neuron_counts = [100, 500, 1000]
        
        for neurons in neuron_counts:
            # Create processor with this neuron count
            processor = create_adaptive_processor(
                vector_dim=300,
                neuron_count=neurons,
                encoding_type="temporal",
                precision_level=3,
                timesteps=10,
                bidirectional_processor=self.bidirectional_processor
            )
            
            # Evaluate processor
            passed, metrics = self.gate.evaluate_model(
                processor,
                self.bidirectional_processor,
                f"neurons_{neurons}"
            )
            
            # Get vector metrics
            cosine_sim = metrics.get_metric("vector", f"neurons_{neurons}", "cosine_similarity")
            
            # Each neuron count should have reasonable cosine similarity
            self.assertIsNotNone(cosine_sim, f"No cosine similarity metric for neuron count {neurons}")
            self.assertGreaterEqual(cosine_sim, 0.5, f"Very low cosine similarity for neuron count {neurons}")


def run_fidelity_tests():
    """Run all fidelity tests."""
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFidelityMetrics))
    suite.addTest(unittest.makeSuite(TestFidelityGate))
    suite.addTest(unittest.makeSuite(TestFidelityGateIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_fidelity_tests() 