#!/usr/bin/env python
"""
Comprehensive Runner Script for All Cogmenta Trainers

This script runs each of the specialized trainers individually in sequence,
providing a way to test all trainers without the complexity of the full
training plan.
"""

import sys
import os
import time
import argparse

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import each trainer class
from training.trainers.concept_net_trainer import ConceptNetTrainer
from training.trainers.nli_trainer import NLITrainer
from training.trainers.logiqa_trainer import LogiQATrainer
from training.trainers.atomic_trainer import AtomicTrainer
from training.trainers.rule_taker_trainer import RuleTakerTrainer

# Import core components if needed
from cogmenta.prolog.prolog_engine import PrologEngine
from cogmenta.bridge.neurosymbolic_bridge import NeuroSymbolicBridge
from cogmenta.visualization.thought_trace import ThoughtTrace

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all Cogmenta trainers sequentially")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trainers_output",
        help="Base directory to store training outputs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for each trainer"
    )
    
    parser.add_argument(
        "--trainers",
        type=str,
        default="all",
        help="Comma-separated list of trainers to run (concept_net,nli,logiqa,atomic,rule_taker) or 'all'"
    )
    
    return parser.parse_args()

def main():
    """Run each trainer sequentially."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize shared components
    prolog_engine = PrologEngine()
    bridge = NeuroSymbolicBridge(prolog_engine=prolog_engine)
    thought_trace = ThoughtTrace()
    
    # Determine which trainers to run
    if args.trainers.lower() == "all":
        trainers_to_run = ["concept_net", "nli", "logiqa", "atomic", "rule_taker"]
    else:
        trainers_to_run = [t.strip() for t in args.trainers.split(",")]
    
    start_time = time.time()
    results = {}
    
    print(f"Starting training for trainers: {', '.join(trainers_to_run)}")
    
    # Run each trainer
    for trainer_name in trainers_to_run:
        trainer_output_dir = os.path.join(args.output_dir, trainer_name)
        os.makedirs(trainer_output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running {trainer_name.upper()} trainer")
        print(f"{'='*50}")
        
        if trainer_name == "concept_net":
            trainer = ConceptNetTrainer(output_dir=trainer_output_dir)
            config = {
                'max_conceptnet_facts': 500,
                'symbol_grounding_epochs': args.epochs,
                'relation_extraction_epochs': args.epochs,
                'abstraction_epochs': max(1, args.epochs//2),
                'integration_epochs': max(1, args.epochs//2),
                'batch_size': 16,
                'save_checkpoints': True
            }
            result = trainer.run_full_training(config=config)
            
        elif trainer_name == "nli":
            trainer = NLITrainer(
                bridge=bridge,
                thought_trace=thought_trace,
                output_dir=trainer_output_dir
            )
            result = trainer.train(epochs=args.epochs, dataset="snli")
            
        elif trainer_name == "logiqa":
            trainer = LogiQATrainer(
                bridge=bridge,
                prolog_engine=prolog_engine,
                thought_trace=thought_trace,
                output_dir=trainer_output_dir
            )
            result = trainer.train(epochs=args.epochs, dataset="logiqa")
            
        elif trainer_name == "atomic":
            trainer = AtomicTrainer(
                bridge=bridge,
                prolog_engine=prolog_engine,
                thought_trace=thought_trace,
                output_dir=trainer_output_dir
            )
            result = trainer.train(epochs=args.epochs)
            
        elif trainer_name == "rule_taker":
            trainer = RuleTakerTrainer(
                bridge=bridge,
                prolog_engine=prolog_engine,
                thought_trace=thought_trace,
                output_dir=trainer_output_dir
            )
            result = trainer.train(max_depth=3, epochs=args.epochs)
            
        else:
            print(f"Unknown trainer: {trainer_name}")
            continue
        
        # Store result
        results[trainer_name] = result
        
        # Print result summary
        if isinstance(result, dict):
            if "duration" in result:
                print(f"\nTraining duration: {result['duration']}")
            
            for metric in ["accuracy", "final_accuracy"]:
                if metric in result:
                    print(f"{metric}: {result[metric]:.4f}")
        
        print(f"Results saved to: {trainer_output_dir}")
    
    # Print overall summary
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Trainers executed: {len(results)}/{len(trainers_to_run)}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 