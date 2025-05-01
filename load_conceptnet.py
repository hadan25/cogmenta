"""
Simple script to load ConceptNet sample data into the symbolic KB.
"""
import os
import sys
from pathlib import Path

# Import the ConceptNetTrainer
from training.conceptnet_trainer import ConceptNetTrainer

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

# Print information about the initialization
print("\nTrainer components initialized:")
print(f"- Concept System: {trainer.concept_system.__class__.__name__}")
print(f"- Vector Symbolic: {trainer.vector_symbolic.__class__.__name__}")
print(f"- Prolog Engine: {trainer.prolog_engine.__class__.__name__}")

# Integrate knowledge from sample data
print("\nIntegrating ConceptNet knowledge to symbolic KB...")
symbolic_facts = trainer.integrate_to_symbolic(
    str(sample_data_path), 
    max_facts=100
)
print(f"Added {symbolic_facts} symbolic facts")

print("\nIntegrating ConceptNet knowledge to vector KB...")
vector_facts = trainer.integrate_to_vector(
    str(sample_data_path), 
    max_facts=100
)
print(f"Added {vector_facts} vector facts")

print("\n" + "="*50)
print("LOADING COMPLETE!")
print(f"Total symbolic facts: {symbolic_facts}")
print(f"Total vector facts: {vector_facts}")
print("="*50) 