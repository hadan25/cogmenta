"""
Test script for NLITrainer

This script tests the NLITrainer by initializing it and trying to load the SNLI dataset.
"""

import sys
import os
from pathlib import Path

# Print debug info
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the project root to the path if needed
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print(f"Added {os.getcwd()} to Python path")

# Import the NLITrainer
try:
    from training.trainers.nli_trainer import NLITrainer
    print("Successfully imported NLITrainer")
except Exception as e:
    print(f"Error importing NLITrainer: {e}")
    sys.exit(1)

# Check if SNLI dataset exists
snli_path = Path("training/datasets/nli/snli.json")
print(f"SNLI path: {snli_path}")
print(f"SNLI absolute path: {snli_path.resolve()}")
print(f"SNLI exists: {snli_path.exists()}")

# Initialize the trainer
try:
    trainer = NLITrainer(output_dir="nli_test_output")
    print("Successfully initialized NLITrainer")
except Exception as e:
    print(f"Error initializing NLITrainer: {e}")
    sys.exit(1)

# Try to load the dataset
try:
    dataset_path = trainer._ensure_dataset("snli")
    if dataset_path:
        print(f"Successfully found dataset at: {dataset_path}")
        
        # Try to load the examples
        examples = trainer._load_examples(dataset_path)
        print(f"Successfully loaded {len(examples)} examples from the dataset")
    else:
        print("Failed to find or create dataset")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

print("Test completed successfully!") 