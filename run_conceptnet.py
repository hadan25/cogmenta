#!/usr/bin/env python
"""
Runner script for ConceptNet training

This script provides a simple entry point to run the ConceptNet training system
without needing to deal with Python module import issues.
"""

import sys
import os
import argparse
import time
import yaml

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the ConceptNetTrainer
from training.trainers.concept_net_trainer import ConceptNetTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ConceptNet training")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="conceptnet_training_output",
        help="Directory to store training outputs"
    )
    
    parser.add_argument(
        "--max_facts",
        type=int,
        default=1000,
        help="Maximum number of ConceptNet facts to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="training/training_config.yaml",
        help="Path to training configuration file"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Run the ConceptNet training system."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Initializing ConceptNet Training System with output to: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the ConceptNetTrainer
    trainer = ConceptNetTrainer(output_dir=args.output_dir)
    
    # Start training timer
    start_time = time.time()
    
    # Run training with configuration
    print(f"Starting training with {args.max_facts} facts and {args.epochs} epochs...")
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        config=config['knowledge_base']['components']['concept_net']['params']
    )
    
    # Calculate training duration
    duration = time.time() - start_time
    
    # Print training results
    print("\nTraining complete!")
    print(f"Training duration: {duration:.2f} seconds")
    
    # Print final metrics from the last epoch
    if trainer.training_stats['relation_prediction']:
        last_metrics = trainer.training_stats['relation_prediction'][-1]
        print(f"\nFinal metrics:")
        print(f"Overall Accuracy: {last_metrics['accuracy']:.4f}")
        print(f"Loss: {last_metrics['loss']:.4f}")
        
        print("\nPer-relation metrics:")
        for rel_type, metrics in last_metrics['per_relation'].items():
            print(f"  {rel_type}: Accuracy={metrics['accuracy']:.4f}, Count={metrics['count']}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 