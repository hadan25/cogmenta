#!/usr/bin/env python
"""
Runner script for Cogmenta training

This script provides a simple entry point to run the Cogmenta training system
without needing to deal with Python module import issues.
"""

import sys
import os

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the training module
from training import TrainingPlan

def main():
    """Run the Cogmenta training system."""
    print("Initializing Cogmenta Training System...")
    
    # Create a training plan with default configuration
    training_plan = TrainingPlan(output_dir="training_output")
    
    # Execute the training plan
    print("Executing training plan...")
    metrics = training_plan.execute()
    
    # Print the summary
    print("\nTraining Summary:")
    print(training_plan.summarize())
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 