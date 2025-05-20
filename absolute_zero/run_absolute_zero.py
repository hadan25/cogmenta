#!/usr/bin/env python
import argparse
import numpy as np
import time
import random
import os
import sys
from absolute_zero.test_absolute_zero import (
    MockSymbolicEngine, 
    MockVSAEngine
)
from absolute_zero.zero_trainer import AbsoluteZeroTrainer
from absolute_zero.enhanced_trainer import create_trainer
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
    parser = argparse.ArgumentParser(description='Run Absolute Zero self-learning with SNN and symbolic components')
    
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=100, 
        help='Number of training iterations'
    )
    
    parser.add_argument(
        '--log-interval', 
        type=int, 
        default=10, 
        help='Interval for logging progress'
    )
    
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='results', 
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true', 
        help='Use mock components for testing'
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
        '--enhanced', 
        action='store_true', 
        help='Use enhanced trainer with all SNN components'
    )
    
    return parser.parse_args()

def run_absolute_zero(args):
    """
    Run Absolute Zero optimization with provided arguments dictionary.
    This function allows direct calling from other scripts without argparse.
    
    Args:
        args: Dictionary with the following possible keys:
            - 'model_type': Type of model to optimize
            - 'iterations': Number of training iterations
            - 'input_dir': Input directory with model to optimize
            - 'output_dir': Directory to save results
            - 'mock': Whether to use mock components
            - 'enhanced': Whether to use enhanced trainer
            - 'real_symbolic': Whether to use real symbolic engines
    
    Returns:
        Dictionary with results including success status
    """
    print(f"Starting Absolute Zero run with args: {args}")
    start_time = time.time()
    
    # Set default values
    iterations = args.get('iterations', 100)
    mock = args.get('mock', True)  # Default to mock for safety
    enhanced = args.get('enhanced', False)
    real_symbolic = args.get('real_symbolic', False)
    output_dir = args.get('output_dir', 'results')
    
    # Setup components
    if mock:
        snns, symbolic_engine, vsa_engine = setup_mock_components()
        print("Using mock SNN components for testing")
    else:
        snns, symbolic_engine, vsa_engine = setup_real_components()
        print("Using real SNN components")
    
    # Override symbolic engines if real-symbolic flag is set
    if real_symbolic and not mock:
        print("Using real symbolic engines...")
        vsa_engine, symbolic_engine = initialize_real_symbolic_engines()
    
    # Create save directory
    run_dir = create_save_directory(output_dir)
    print(f"Results will be saved to {run_dir}")
    
    # Initialize trainer based on enhanced flag
    if enhanced:
        # Use the enhanced trainer with all SNN components
        trainer = create_trainer(
            use_real_snns=not mock,
            symbolic_engine=symbolic_engine,
            vsa_engine=vsa_engine
        )
        print("Using enhanced trainer with all SNN components")
    else:
        # Use the basic trainer with just the core SNN components
        trainer = AbsoluteZeroTrainer(snns, symbolic_engine, vsa_engine)
        print("Using basic trainer with core SNN components")
    
    try:
        # Run training
        print(f"Starting Absolute Zero training for {iterations} iterations...")
        trainer.train(iterations=iterations, log_interval=10)
        
        # Save training statistics
        stats_dict = {
            'iterations': trainer.training_stats['iterations'],
            'total_reward': float(trainer.training_stats['total_reward']),
            'accuracy_history': [float(acc) for acc in trainer.training_stats['accuracy_history']]
        }
        
        # Add enhanced metrics if available
        if enhanced:
            if 'reasoning_accuracy' in trainer.training_stats:
                stats_dict['reasoning_accuracy'] = [float(acc) for acc in trainer.training_stats['reasoning_accuracy']]
            if 'decision_confidence' in trainer.training_stats:
                stats_dict['decision_confidence'] = [float(conf) for conf in trainer.training_stats['decision_confidence']]
            if 'memory_utilization' in trainer.training_stats:
                stats_dict['memory_utilization'] = [float(util) for util in trainer.training_stats['memory_utilization']]
        
        # Add configuration info
        stats_dict['config'] = {
            'use_mock': mock,
            'use_real_symbolic': real_symbolic,
            'enhanced': enhanced,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save stats to file
        stats_path = os.path.join(run_dir, 'training_stats.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
            
        print(f"Saved training statistics to {stats_path}")
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        
        return {
            'success': True,
            'stats': stats_dict,
            'output_dir': run_dir,
            'duration': end_time - start_time
        }
        
    except Exception as e:
        import traceback
        print(f"Error in Absolute Zero training: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'output_dir': run_dir,
            'duration': time.time() - start_time
        }

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

def setup_real_components():
    # Use adapter-based approach with real SNNs
    snns = create_snn_components(use_real_snn=True)
    
    # Use real symbolic engines if available, otherwise use mock
    vsa_engine, symbolic_engine = initialize_real_symbolic_engines()
    
    return snns, symbolic_engine, vsa_engine

def setup_mock_components():
    # Use adapter-based approach with mock SNNs
    snns = create_snn_components(use_real_snn=False)
    symbolic_engine = MockSymbolicEngine()
    vsa_engine = MockVSAEngine()
    
    return snns, symbolic_engine, vsa_engine

def create_save_directory(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a timestamp-based run directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir

def visualize_results(trainer, save_dir):
    try:
        import matplotlib.pyplot as plt
        
        # Create accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.training_stats['accuracy_history'])
        plt.title('Accuracy Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy.png'))
        
        # Other visualizations could be added here
        
        print(f"Visualizations saved to {save_dir}")
    except ImportError:
        print("Matplotlib not available. Skipping visualizations.")

def main():
    args = parse_arguments()
    
    # Setup components
    if args.mock:
        snns, symbolic_engine, vsa_engine = setup_mock_components()
        print("Using mock SNN components for testing")
    else:
        snns, symbolic_engine, vsa_engine = setup_real_components()
        print("Using real SNN components")
    
    # Override symbolic engines if real-symbolic flag is set
    if args.real_symbolic and not args.mock:
        print("Using real symbolic engines...")
        vsa_engine, symbolic_engine = initialize_real_symbolic_engines()
    
    # Create save directory
    run_dir = create_save_directory(args.save_dir)
    print(f"Results will be saved to {run_dir}")
    
    # Initialize trainer based on enhanced flag
    if args.enhanced:
        # Use the enhanced trainer with all SNN components
        trainer = create_trainer(
            use_real_snns=not args.mock,
            symbolic_engine=symbolic_engine,
            vsa_engine=vsa_engine
        )
        print("Using enhanced trainer with all SNN components")
    else:
        # Use the basic trainer with just the core SNN components
        trainer = AbsoluteZeroTrainer(snns, symbolic_engine, vsa_engine)
        print("Using basic trainer with core SNN components")
    
    # Print symbolic engine status
    using_real_vsa = not isinstance(vsa_engine, MockVSAEngine)
    using_real_symbolic = not isinstance(symbolic_engine, MockSymbolicEngine)
    print(f"- VSA ENGINE: {'REAL' if using_real_vsa else 'MOCK'}")
    print(f"- SYMBOLIC ENGINE: {'REAL' if using_real_symbolic else 'MOCK'}")
    print("")
    
    # Run training
    print(f"Starting Absolute Zero training for {args.iterations} iterations...")
    start_time = time.time()
    
    trainer.train(iterations=args.iterations, log_interval=args.log_interval)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Save and visualize results
    if args.visualize:
        visualize_results(trainer, run_dir)
    
    # Save training statistics
    try:
        import json
        stats_dict = {
            'iterations': trainer.training_stats['iterations'],
            'total_reward': float(trainer.training_stats['total_reward']),
            'accuracy_history': [float(acc) for acc in trainer.training_stats['accuracy_history']]
        }
        
        # Add enhanced metrics if available
        if args.enhanced:
            if 'reasoning_accuracy' in trainer.training_stats:
                stats_dict['reasoning_accuracy'] = [float(acc) for acc in trainer.training_stats['reasoning_accuracy']]
            if 'decision_confidence' in trainer.training_stats:
                stats_dict['decision_confidence'] = [float(conf) for conf in trainer.training_stats['decision_confidence']]
            if 'memory_utilization' in trainer.training_stats:
                stats_dict['memory_utilization'] = [float(util) for util in trainer.training_stats['memory_utilization']]
        
        # Add configuration info
        stats_dict['config'] = {
            'use_mock': args.mock,
            'use_real_symbolic': args.real_symbolic,
            'enhanced': args.enhanced,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(run_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats_dict, f, indent=2)
    except Exception as e:
        print(f"Error saving statistics: {e}")
    
    print("Run completed successfully!")

if __name__ == "__main__":
    main() 