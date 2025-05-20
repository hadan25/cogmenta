#!/usr/bin/env python
"""
Fidelity Gate CLI Tool

This script provides a command-line interface for running the Quantitative Fidelity Gate
as part of a model training pipeline. It ensures that models meet the required
fidelity thresholds before proceeding to training.

Usage:
    python fidelity_gate_cli.py --model-config config.json --output-dir ./reports
    python fidelity_gate_cli.py --model-dir ./models --output-dir ./reports
"""

import os
import sys
import json
import argparse
import logging
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from models.snn.fidelity_gate import create_fidelity_gate, FidelityGate, FidelityMetrics
from models.snn.adaptive_spike_processor import create_adaptive_processor, AdaptiveSpikeProcessor
from models.snn.bidirectional_encoding import create_processor
from models.snn.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("fidelity_gate_cli")


def load_model_from_config(config_path: str) -> Dict:
    """
    Load model configuration from a JSON file.
    
    Args:
        config_path: Path to the model configuration JSON file
        
    Returns:
        Dictionary of model configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load model config from {config_path}: {e}")
        sys.exit(1)


def create_processor_from_config(config: Dict) -> AdaptiveSpikeProcessor:
    """
    Create an adaptive spike processor from a configuration dictionary.
    
    Args:
        config: Dictionary of model configuration
        
    Returns:
        Configured AdaptiveSpikeProcessor instance
    """
    # Extract processor parameters from config
    vector_dim = config.get("vector_dim", 300)
    neuron_count = config.get("neuron_count", 500)
    encoding_type = config.get("encoding_type", "temporal")
    precision_level = config.get("precision_level", 3)
    timesteps = config.get("timesteps", 10)
    modalities = config.get("modalities", ["text", "vector"])
    
    # Create bidirectional processor if needed for text
    bidirectional_processor = None
    if "text" in modalities:
        bidirectional_processor = create_processor(
            model_type=config.get("bidirectional_model_type", "generic"),
            vector_dim=vector_dim
        )
    
    # Create the processor
    processor = create_adaptive_processor(
        vector_dim=vector_dim,
        neuron_count=neuron_count,
        encoding_type=encoding_type,
        precision_level=precision_level,
        timesteps=timesteps,
        modalities=modalities,
        bidirectional_processor=bidirectional_processor
    )
    
    # Load saved model state if provided
    if "model_path" in config and config["model_path"]:
        model_path = config["model_path"]
        logger.info(f"Loading model state from {model_path}")
        success = processor.load(model_path)
        if not success:
            logger.warning(f"Failed to load model state from {model_path}, using fresh initialization")
    
    return processor, bidirectional_processor


def load_custom_thresholds(threshold_path: Optional[str]) -> Optional[Dict]:
    """
    Load custom thresholds from a JSON file.
    
    Args:
        threshold_path: Path to the thresholds JSON file
        
    Returns:
        Dictionary of thresholds or None if not provided
    """
    if not threshold_path:
        return None
    
    try:
        with open(threshold_path, 'r') as f:
            thresholds = json.load(f)
        return thresholds
    except Exception as e:
        logger.error(f"Failed to load thresholds from {threshold_path}: {e}")
        logger.info("Using default thresholds instead")
        return None


def run_fidelity_gate(
    processor: AdaptiveSpikeProcessor,
    bidirectional_processor,
    model_name: str,
    output_dir: str,
    thresholds: Optional[Dict] = None,
    fail_action: str = "exit",
    modalities: Optional[List[str]] = None
) -> bool:
    """
    Run the fidelity gate on a model.
    
    Args:
        processor: The AdaptiveSpikeProcessor to evaluate
        bidirectional_processor: Optional bidirectional processor for text
        model_name: Name of the model for reporting
        output_dir: Directory to save reports
        thresholds: Optional custom thresholds
        fail_action: Action to take if gate fails ("exit", "warn", or "continue")
        modalities: Optional list of modalities to test
        
    Returns:
        Boolean indicating whether the model passed the gate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create fidelity gate with custom thresholds if provided
    gate = create_fidelity_gate(
        modalities=modalities or processor.modalities,
        thresholds=thresholds
    )
    
    # Evaluate the model
    logger.info(f"Evaluating model '{model_name}' against fidelity gate...")
    passed, metrics = gate.evaluate_model(
        processor,
        bidirectional_processor,
        model_name
    )
    
    # Save report
    timestamp = int(time.time())
    report_path = os.path.join(output_dir, f"fidelity_report_{model_name}_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(metrics.to_report())
        f.write(f"\n\nPASSED: {passed}")
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f"fidelity_metrics_{model_name}_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics.get_summary(), f, indent=2)
    
    logger.info(f"Fidelity report saved to {report_path}")
    logger.info(f"Fidelity metrics saved to {metrics_path}")
    
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
    
    return passed


def run_batch_evaluation(
    model_dir: str,
    output_dir: str,
    thresholds_path: Optional[str] = None,
    fail_action: str = "warn"
) -> Dict[str, bool]:
    """
    Evaluate all models in a directory.
    
    Args:
        model_dir: Directory containing model configuration files
        output_dir: Directory to save reports
        thresholds_path: Optional path to custom thresholds file
        fail_action: Action to take if gate fails
        
    Returns:
        Dictionary of model names and their pass/fail status
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom thresholds if provided
    thresholds = load_custom_thresholds(thresholds_path)
    
    # Find all model config files
    config_files = list(Path(model_dir).glob("**/config.json"))
    if not config_files:
        logger.error(f"No model configuration files found in {model_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(config_files)} model configuration files")
    
    # Track results
    results = {}
    
    # Process each model
    for config_path in config_files:
        # Extract model name from directory
        model_name = config_path.parent.name
        logger.info(f"Processing model: {model_name}")
        
        # Load config
        config = load_model_from_config(str(config_path))
        
        # Create processor
        processor, bidirectional_processor = create_processor_from_config(config)
        
        # Run fidelity gate
        passed = run_fidelity_gate(
            processor,
            bidirectional_processor,
            model_name,
            output_dir,
            thresholds,
            fail_action
        )
        
        # Store result
        results[model_name] = passed
    
    # Summarize results
    logger.info("\nFidelity Gate Summary:")
    logger.info("=====================")
    for model, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{model}: {status}")
    
    # Check if any model failed
    if not all(results.values()) and fail_action == "exit":
        logger.error("One or more models failed the fidelity gate. Pipeline cannot proceed.")
        sys.exit(1)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Quantitative Fidelity Gate")
    
    # Define mutually exclusive group for model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-config", 
        help="Path to model configuration JSON file"
    )
    model_group.add_argument(
        "--model-dir", 
        help="Directory containing model configurations to evaluate"
    )
    
    # Other arguments
    parser.add_argument(
        "--output-dir", 
        default="./fidelity_reports",
        help="Directory to save fidelity reports"
    )
    parser.add_argument(
        "--thresholds", 
        help="Path to custom thresholds JSON file"
    )
    parser.add_argument(
        "--model-name", 
        default="model",
        help="Name of the model (used with --model-config)"
    )
    parser.add_argument(
        "--fail-action", 
        choices=["exit", "warn", "continue"],
        default="exit",
        help="Action to take if fidelity gate fails"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["text", "vector", "image", "audio"],
        help="Modalities to test (default: all supported by model)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Process batch of models
        if args.model_dir:
            run_batch_evaluation(
                args.model_dir,
                args.output_dir,
                args.thresholds,
                args.fail_action
            )
        # Process single model
        else:
            # Load model config
            config = load_model_from_config(args.model_config)
            
            # Create processor
            processor, bidirectional_processor = create_processor_from_config(config)
            
            # Load custom thresholds if provided
            thresholds = load_custom_thresholds(args.thresholds)
            
            # Run fidelity gate
            run_fidelity_gate(
                processor,
                bidirectional_processor,
                args.model_name,
                args.output_dir,
                thresholds,
                args.fail_action,
                args.modalities
            )
    
    except Exception as e:
        logger.error(f"Error running fidelity gate: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 