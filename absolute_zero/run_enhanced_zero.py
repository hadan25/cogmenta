#!/usr/bin/env python
"""
Run Enhanced Absolute Zero with all SNN components.

This script provides a simplified interface to run the Enhanced Absolute Zero
training loop with all the SNN components integrated. It includes the 
standard Statistical, Affective, and Metacognitive SNNs along with the 
extended Reasoning, Decision, Perceptual, and Memory SNNs for a complete
neuro-symbolic learning system.
"""

import argparse
import time
import os
import sys
import json
import numpy as np

# Import enhanced trainer
from absolute_zero.enhanced_trainer import create_trainer
from absolute_zero.test_absolute_zero import MockVSAEngine, MockSymbolicEngine
from absolute_zero.snn_adapter import create_snn_components

# Try to import real symbolic engines
try:
    from models.symbolic.vector_symbolic import VectorSymbolicEngine
    from models.symbolic.prolog_engine import PrologEngine
    HAS_REAL_SYMBOLIC = True
except ImportError:
    print("Warning: Could not import real symbolic engines. Will use mock implementations if requested.")
    HAS_REAL_SYMBOLIC = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Enhanced Absolute Zero with all SNN components')
    
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=1000, 
        help='Number of training iterations'
    )
    
    parser.add_argument(
        '--log-interval', 
        type=int, 
        default=50, 
        help='Interval for logging progress'
    )
    
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='results/enhanced', 
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--real-snns', 
        action='store_true', 
        help='Use real SNN implementations instead of mock adaptors'
    )
    
    parser.add_argument(
        '--real-symbolic', 
        action='store_true', 
        help='Use real symbolic engines (VSA and Prolog) instead of mock implementations'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true', 
        help='Generate visualizations of learning progress'
    )
    
    parser.add_argument(
        '--task-complexity',
        type=float,
        default=0.5,
        help='Initial task complexity (0.0-1.0)'
    )
    
    parser.add_argument(
        '--emphasize',
        choices=['statistical', 'affective', 'metacognitive', 'reasoning', 'decision', 'perceptual', 'memory'],
        help='Emphasize a particular SNN component in the learning process'
    )
    
    return parser.parse_args()

def create_save_directory(save_dir):
    """Create a timestamped directory for saving results."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir

def initialize_real_symbolic_engines():
    """Initialize real symbolic engines (VSA and Prolog)."""
    if not HAS_REAL_SYMBOLIC:
        print("Warning: Real symbolic engines not available. Using mock implementations.")
        return MockVSAEngine(), MockSymbolicEngine()
    
    try:
        print("Initializing real Vector Symbolic Architecture engine...")
        vsa_engine = VectorSymbolicEngine(dimension=1000, sparsity=0.1)
        
        print("Initializing real Prolog engine...")
        symbolic_engine = PrologEngine()
        
        print("Real symbolic engines initialized successfully!")
        return vsa_engine, symbolic_engine
    except Exception as e:
        print(f"Error initializing real symbolic engines: {e}")
        print("Falling back to mock implementations.")
        return MockVSAEngine(), MockSymbolicEngine()

def visualize_results(trainer, save_dir):
    """Generate visualization of training metrics."""
    try:
        import matplotlib.pyplot as plt
        
        # Plot all available metrics
        metrics = []
        if trainer.training_stats['accuracy_history']:
            metrics.append(('accuracy_history', 'Accuracy', 'b-'))
        if 'reasoning_accuracy' in trainer.training_stats and trainer.training_stats['reasoning_accuracy']:
            metrics.append(('reasoning_accuracy', 'Reasoning Accuracy', 'g-'))
        if 'decision_confidence' in trainer.training_stats and trainer.training_stats['decision_confidence']:
            metrics.append(('decision_confidence', 'Decision Confidence', 'r-'))
        if 'memory_utilization' in trainer.training_stats and trainer.training_stats['memory_utilization']:
            metrics.append(('memory_utilization', 'Memory Utilization', 'y-'))
        
        # Create a combined plot
        plt.figure(figsize=(12, 8))
        for metric_name, label, style in metrics:
            plt.plot(trainer.training_stats[metric_name], style, label=label)
        
        plt.title('Enhanced Absolute Zero Training Metrics')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'combined_metrics.png'))
        
        # Create a sliding window average plot for smoother visualization
        plt.figure(figsize=(12, 8))
        window_size = min(50, len(trainer.training_stats['accuracy_history']))
        
        for metric_name, label, style in metrics:
            values = trainer.training_stats[metric_name]
            # Calculate moving average if we have enough data points
            if len(values) >= window_size:
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed, style, label=f"{label} (Moving Avg)")
            else:
                plt.plot(values, style, label=label)
        
        plt.title('Enhanced Absolute Zero Training Metrics (Smoothed)')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'smoothed_metrics.png'))
        
        print(f"Visualizations saved to {save_dir}")
    except ImportError:
        print("Matplotlib not available. Skipping visualizations.")

def run_enhanced_zero(args_dict=None):
    """
    Run Enhanced Absolute Zero with the provided arguments.
    This function allows direct calling from other scripts.
    
    Args:
        args_dict: Dictionary with arguments (optional)
        
    Returns:
        Dictionary with results
    """
    if args_dict is None:
        # Parse command line arguments if not provided
        args = parse_arguments()
        args_dict = vars(args)
    else:
        # Convert to object with attributes for compatibility
        class ArgsObject:
            def __init__(self, args_dict):
                for key, value in args_dict.items():
                    setattr(self, key, value)
        args = ArgsObject(args_dict)
    
    # Create save directory
    save_dir = args_dict.get('save_dir', 'results/enhanced')
    run_dir = create_save_directory(save_dir)
    print(f"Results will be saved to {run_dir}")
    
    # Initialize required engines
    use_real_symbolic = args_dict.get('real_symbolic', False)
    if use_real_symbolic:
        print("Using real symbolic engines...")
        vsa_engine, symbolic_engine = initialize_real_symbolic_engines()
    else:
        print("Using mock symbolic engines...")
        vsa_engine = MockVSAEngine()
        symbolic_engine = MockSymbolicEngine()
    
    # Initialize SNN components directly
    use_real_snns = args_dict.get('real_snns', False)
    print(f"Creating SNN components with use_real_snns={use_real_snns}")
    snns = create_snn_components(use_real_snns)
    
    # Create trainer with snns and appropriate settings
    trainer = create_trainer(
        snns=snns,
        symbolic_engine=symbolic_engine,
        vsa_engine=vsa_engine
    )
    
    # Save configuration
    config = {
        'iterations': args_dict.get('iterations', 1000),
        'log_interval': args_dict.get('log_interval', 50),
        'use_real_snns': use_real_snns,
        'use_real_symbolic': use_real_symbolic,
        'task_complexity': args_dict.get('task_complexity', 0.5),
        'emphasize': args_dict.get('emphasize', None),
        'run_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set initial task complexity
    if hasattr(trainer, 'task_generator') and hasattr(trainer.task_generator, 'set_complexity'):
        trainer.task_generator.set_complexity(args_dict.get('task_complexity', 0.5))
    
    # Emphasize a particular SNN component if requested
    emphasize = args_dict.get('emphasize')
    if emphasize and hasattr(trainer, 'emphasize_component'):
        trainer.emphasize_component(emphasize)
    
    # Run training
    start_time = time.time()
    iterations = args_dict.get('iterations', 1000)
    log_interval = args_dict.get('log_interval', 50)
    
    try:
        trainer.train(iterations=iterations, log_interval=log_interval)
        
        # Generate visualizations if requested
        if args_dict.get('visualize', False):
            visualize_results(trainer, run_dir)
        
        # Save final results
        results_path = os.path.join(run_dir, 'results.json')
        results = {
            'accuracy': float(np.mean(trainer.training_stats['accuracy_history'][-100:]) if trainer.training_stats['accuracy_history'] else 0.0),
            'total_reward': float(trainer.training_stats['total_reward']),
            'training_time': time.time() - start_time,
            'iterations_completed': trainer.training_stats['iterations'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")
        print(f"Results saved to {results_path}")
        
        return {
            'success': True,
            'results': results,
            'config': config,
            'save_dir': run_dir
        }
        
    except Exception as e:
        import traceback
        print(f"Error in Enhanced Absolute Zero training: {e}")
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'save_dir': run_dir,
            'config': config
        }

def main():
    run_enhanced_zero()

if __name__ == "__main__":
    main() 