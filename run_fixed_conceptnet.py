"""
Fixed script to run the ConceptNet training with Prolog syntax fixes.
This version applies patches to handle Prolog syntax errors properly.
"""
import os
import sys
from pathlib import Path

# Import the ConceptNetTrainer
from training.conceptnet_trainer import ConceptNetTrainer
# Import our fixes
from prolog_fix import apply_prolog_fixes

# Apply the Prolog fixes first
print("Applying Prolog syntax fixes...")
if not apply_prolog_fixes():
    print("Failed to apply Prolog fixes, exiting.")
    sys.exit(1)
print("Prolog fixes applied successfully.")

# Set up sample data path
sample_data_path = Path("training/datasets/conceptnet/sample_data.csv")

print(f"Using sample data from: {sample_data_path}")

if not sample_data_path.exists():
    print(f"ERROR: Sample data file not found: {sample_data_path}")
    sys.exit(1)

# Create output directory
output_dir = "conceptnet_training_output"
os.makedirs(output_dir, exist_ok=True)

# Initialize trainer
print("Initializing ConceptNetTrainer...")
trainer = ConceptNetTrainer(output_dir=output_dir)

# Override download method to use our sample data
original_fn = trainer.download_and_process_conceptnet
trainer.download_and_process_conceptnet = lambda *args, **kwargs: str(sample_data_path)

# Configure training with integration enabled
config = {
    'max_conceptnet_facts': 100,
    'symbol_grounding_epochs': 2,
    'relation_extraction_epochs': 2,
    'abstraction_epochs': 1,
    'integration_epochs': 1,  # Integration training enabled
    'batch_size': 16,
    'train_size': 10,
    'val_size': 3,
    'test_size': 3,
    'save_checkpoints': True,
    'visualize_training': True
}

print("Starting training with configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Run training
print("\nRunning ConceptNet training with fixes applied...")
results = trainer.run_full_training(config=config)

# Restore original method
trainer.download_and_process_conceptnet = original_fn

# Print results
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print(f"Duration: {results['duration']}")

# Print evaluation results
print("\nEvaluation Results:")
for category, metrics in results['evaluation'].items():
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        print(f"  {category} accuracy: {metrics['accuracy']:.4f}")

print(f"\nResults saved to: {output_dir}")
print("="*50) 