"""
SNN Training Pipeline with Fidelity Gate Integration

This script demonstrates how to integrate the Quantitative Fidelity Gate into
the SNN training pipeline. It ensures that models meet the required fidelity
thresholds before proceeding to training.

The pipeline consists of these main stages:
1. Model initialization and configuration
2. Fidelity gate evaluation (with strict thresholds)
3. Model training (if fidelity gate passes)
4. Evaluation and reporting

Usage:
    python snn_pipeline.py --config pipeline_config.json
"""

import os
import sys
import json
import argparse
import logging
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import SNN components
from models.snn.fidelity_gate import create_fidelity_gate, FidelityGate, FidelityMetrics
from models.snn.adaptive_spike_processor import create_adaptive_processor, AdaptiveSpikeProcessor
from models.snn.bidirectional_encoding import create_processor
from models.snn.utils.logging_config import setup_logging
from models.snn.train_snn_with_rewards import SNNTrainer

# Set up logging
logger = setup_logging("snn_pipeline")


class SNNPipeline:
    """
    Pipeline for training SNN models with fidelity gate integration.
    
    This class implements a complete pipeline for SNN model training,
    including fidelity checking, training, and evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with a configuration file.
        
        Args:
            config_path: Path to the pipeline configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get("output_dir", "./snn_pipeline_output")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.processors = {}
        self.bidirectional_processor = None
        self.fidelity_gate = None
        self.trainer = None
        
        # Setup pipeline components
        self._setup_bidirectional_processor()
        self._setup_processors()
        self._setup_fidelity_gate()
        self._setup_trainer()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)
    
    def _setup_bidirectional_processor(self):
        """Setup the bidirectional processor for text processing."""
        bidirectional_config = self.config.get("bidirectional_processor", {})
        vector_dim = bidirectional_config.get("vector_dim", 300)
        model_type = bidirectional_config.get("model_type", "generic")
        
        logger.info(f"Setting up bidirectional processor (type: {model_type}, dim: {vector_dim})")
        
        self.bidirectional_processor = create_processor(
            model_type=model_type,
            vector_dim=vector_dim
        )
    
    def _setup_processors(self):
        """Setup the SNN processors based on configuration."""
        processor_configs = self.config.get("processors", {})
        
        for model_name, config in processor_configs.items():
            logger.info(f"Setting up processor: {model_name}")
            
            # Extract processor parameters
            vector_dim = config.get("vector_dim", 300)
            neuron_count = config.get("neuron_count", 500)
            encoding_type = config.get("encoding_type", "temporal")
            precision_level = config.get("precision_level", 3)
            timesteps = config.get("timesteps", 10)
            modalities = config.get("modalities", ["text", "vector"])
            
            # Create processor
            processor = create_adaptive_processor(
                vector_dim=vector_dim,
                neuron_count=neuron_count,
                encoding_type=encoding_type,
                precision_level=precision_level,
                timesteps=timesteps,
                modalities=modalities,
                bidirectional_processor=self.bidirectional_processor
            )
            
            # Load saved model state if provided
            if "model_path" in config and config["model_path"]:
                model_path = config["model_path"]
                logger.info(f"Loading model state for {model_name} from {model_path}")
                success = processor.load(model_path)
                if not success:
                    logger.warning(f"Failed to load model state from {model_path}, using fresh initialization")
            
            # Store processor
            self.processors[model_name] = processor
    
    def _setup_fidelity_gate(self):
        """Setup the fidelity gate based on configuration."""
        gate_config = self.config.get("fidelity_gate", {})
        
        # Get modalities to test
        modalities = gate_config.get("modalities", ["text", "vector"])
        
        # Get custom thresholds if provided
        thresholds = gate_config.get("thresholds", None)
        
        logger.info(f"Setting up fidelity gate (modalities: {', '.join(modalities)})")
        
        # Create fidelity gate
        self.fidelity_gate = create_fidelity_gate(
            modalities=modalities,
            thresholds=thresholds
        )
    
    def _setup_trainer(self):
        """Setup the SNN trainer based on configuration."""
        trainer_config = self.config.get("trainer", {})
        
        logger.info("Setting up SNN trainer")
        
        # Create trainer
        self.trainer = SNNTrainer(
            vector_dim=trainer_config.get("vector_dim", 300),
            neuron_count=trainer_config.get("neuron_count", 500),
            training_approach=trainer_config.get("training_approach", "hybrid"),
            learning_rate=trainer_config.get("learning_rate", 0.001),
            temperature=trainer_config.get("temperature", 0.5),
            reward_scale=trainer_config.get("reward_scale", 1.0),
            device=trainer_config.get("device", None)
        )
    
    def run_fidelity_gate(self) -> Dict[str, bool]:
        """
        Run the fidelity gate on all processors.
        
        Returns:
            Dictionary of model names and their pass/fail status
        """
        logger.info("Running fidelity gate on all processors...")
        
        # Check if fidelity gate is enabled
        if not self.config.get("enable_fidelity_gate", True):
            logger.warning("Fidelity gate is disabled in configuration. Skipping.")
            return {model: True for model in self.processors}
        
        gate_config = self.config.get("fidelity_gate", {})
        fail_action = gate_config.get("fail_action", "exit")
        
        results = {}
        fidelity_dir = os.path.join(self.output_dir, "fidelity_reports")
        os.makedirs(fidelity_dir, exist_ok=True)
        
        # Evaluate each processor
        for model_name, processor in self.processors.items():
            logger.info(f"Evaluating {model_name} against fidelity gate...")
            
            # Run fidelity gate
            passed, metrics = self.fidelity_gate.evaluate_model(
                processor,
                self.bidirectional_processor,
                model_name
            )
            
            # Save report
            timestamp = int(time.time())
            report_path = os.path.join(fidelity_dir, f"fidelity_report_{model_name}_{timestamp}.txt")
            with open(report_path, 'w') as f:
                f.write(metrics.to_report())
                f.write(f"\n\nPASSED: {passed}")
            
            # Store result
            results[model_name] = passed
            
            # Handle failure
            if not passed:
                if fail_action == "exit":
                    logger.error(f"Model '{model_name}' FAILED the fidelity gate. Pipeline cannot proceed.")
                    sys.exit(1)
                elif fail_action == "warn":
                    logger.warning(f"Model '{model_name}' FAILED the fidelity gate, but continuing as requested.")
                else:
                    logger.info(f"Model '{model_name}' did not meet all fidelity thresholds.")
            else:
                logger.info(f"Model '{model_name}' PASSED the fidelity gate!")
        
        # Summarize results
        logger.info("\nFidelity Gate Summary:")
        logger.info("=====================")
        for model, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{model}: {status}")
        
        return results
    
    def train_models(self, fidelity_results: Dict[str, bool]) -> Dict[str, Any]:
        """
        Train the models that passed the fidelity gate.
        
        Args:
            fidelity_results: Dictionary of model names and their pass/fail status
            
        Returns:
            Dictionary of training results
        """
        logger.info("Starting model training...")
        
        trainer_config = self.config.get("trainer", {})
        training_dir = os.path.join(self.output_dir, "training")
        os.makedirs(training_dir, exist_ok=True)
        
        # Check if training is enabled
        if not self.config.get("enable_training", True):
            logger.warning("Training is disabled in configuration. Skipping.")
            return {}
        
        # Get models to train
        train_only_passed = trainer_config.get("train_only_passed", True)
        models_to_train = {}
        
        for model_name, processor in self.processors.items():
            if train_only_passed and not fidelity_results.get(model_name, False):
                logger.warning(f"Skipping training for {model_name} because it failed the fidelity gate")
            else:
                models_to_train[model_name] = processor
        
        if not models_to_train:
            logger.warning("No models to train. Skipping training phase.")
            return {}
        
        # Train each model
        training_results = {}
        
        for model_name, processor in models_to_train.items():
            logger.info(f"Training model: {model_name}")
            
            # Get model-specific training config
            model_config = self.config.get("processors", {}).get(model_name, {})
            epochs = model_config.get("epochs", trainer_config.get("epochs", 10))
            batch_size = model_config.get("batch_size", trainer_config.get("batch_size", 32))
            training_approach = model_config.get("training_approach", trainer_config.get("training_approach", "hybrid"))
            
            # Create model output directory
            model_dir = os.path.join(training_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Generate some mock training data
            # This would be replaced with real data in a production pipeline
            train_data = torch.rand(500, processor.vector_dim)
            
            # Train the model
            train_result = self.trainer.train_single_model(
                processor=processor,
                model_name=model_name,
                train_data=train_data,
                epochs=epochs,
                batch_size=batch_size,
                training_approach=training_approach,
                output_dir=model_dir
            )
            
            # Save the trained model
            save_path = os.path.join(model_dir, "trained_model")
            os.makedirs(save_path, exist_ok=True)
            processor.save(save_path)
            
            # Store result
            training_results[model_name] = train_result
        
        # Train joint model if configured
        if trainer_config.get("enable_joint_training", False) and len(models_to_train) > 1:
            logger.info("Starting joint model training...")
            
            joint_result = self.trainer.train_joint_models(
                processors=models_to_train,
                train_data=torch.rand(200, processor.vector_dim),  # Mock data
                epochs=trainer_config.get("joint_epochs", 5),
                batch_size=trainer_config.get("batch_size", 32),
                output_dir=os.path.join(training_dir, "joint")
            )
            
            training_results["joint"] = joint_result
        
        return training_results
    
    def evaluate_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the trained models.
        
        Args:
            training_results: Dictionary of training results
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating trained models...")
        
        # Check if evaluation is enabled
        if not self.config.get("enable_evaluation", True):
            logger.warning("Evaluation is disabled in configuration. Skipping.")
            return {}
        
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Re-run fidelity gate on trained models
        fidelity_results = {}
        
        for model_name, processor in self.processors.items():
            if model_name not in training_results:
                continue
                
            logger.info(f"Re-evaluating fidelity for trained model: {model_name}")
            
            # Run fidelity gate
            passed, metrics = self.fidelity_gate.evaluate_model(
                processor,
                self.bidirectional_processor,
                f"{model_name}_after_training"
            )
            
            # Save report
            timestamp = int(time.time())
            report_path = os.path.join(eval_dir, f"fidelity_report_{model_name}_after_{timestamp}.txt")
            with open(report_path, 'w') as f:
                f.write(metrics.to_report())
                f.write(f"\n\nPASSED: {passed}")
            
            # Store result
            fidelity_results[model_name] = {
                "passed": passed,
                "metrics": metrics.get_summary()
            }
        
        # Perform additional evaluation tasks here...
        # This could include model-specific testing, end-to-end testing, etc.
        
        return fidelity_results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete SNN pipeline.
        
        Returns:
            Dictionary of pipeline results
        """
        logger.info("Starting SNN pipeline...")
        
        # Run fidelity gate
        fidelity_results = self.run_fidelity_gate()
        
        # Train models
        training_results = self.train_models(fidelity_results)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(training_results)
        
        # Create pipeline report
        pipeline_results = {
            "fidelity_results": fidelity_results,
            "training_results": training_results,
            "evaluation_results": evaluation_results
        }
        
        # Save pipeline results
        results_path = os.path.join(self.output_dir, f"pipeline_results_{int(time.time())}.json")
        with open(results_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.dumps(pipeline_results, default=lambda o: str(o) if not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o)
            f.write(serializable_results)
        
        logger.info(f"Pipeline completed. Results saved to {results_path}")
        
        return pipeline_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the SNN pipeline with fidelity gate")
    
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to pipeline configuration JSON file"
    )
    
    parser.add_argument(
        "--skip-fidelity", 
        action="store_true",
        help="Skip the fidelity gate check"
    )
    
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="Skip the training phase"
    )
    
    parser.add_argument(
        "--skip-evaluation", 
        action="store_true",
        help="Skip the evaluation phase"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        # Override configuration based on command line arguments
        config = json.load(open(args.config, 'r'))
        
        if args.skip_fidelity:
            config["enable_fidelity_gate"] = False
        
        if args.skip_training:
            config["enable_training"] = False
        
        if args.skip_evaluation:
            config["enable_evaluation"] = False
        
        # Save modified configuration
        modified_config_path = f"{args.config}.modified"
        with open(modified_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run pipeline
        pipeline = SNNPipeline(modified_config_path)
        results = pipeline.run_pipeline()
        
        # Clean up
        if os.path.exists(modified_config_path):
            os.remove(modified_config_path)
        
    except Exception as e:
        logger.error(f"Error running SNN pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 