
import os
import sys
# Add directory to path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
abs_zero_dir = os.path.join(current_dir, 'absolute_zero')
sys.path.insert(0, current_dir)

# Now import works
from absolute_zero.zero_trainer import AbsoluteZeroTrainer
from absolute_zero.run_absolute_zero import setup_mock_components, create_save_directory
from absolute_zero.snn_adapter import create_snn_components

# Create components
snns = create_snn_components(use_real_snn=False)
_, symbolic_engine, vsa_engine = setup_mock_components()

# Create save directory
save_dir = "C:\\Users\\ha25a\\.vscode\\cogmenta_core\\fixed_hybrid_results\\all_20250517_000232\\memory\\absolute"
os.makedirs(save_dir, exist_ok=True)

# Create trainer
trainer = AbsoluteZeroTrainer(snns, symbolic_engine, vsa_engine)

# Run training
iterations = 5
trainer.train(iterations=iterations, log_interval=max(1, iterations//10))

# Save results
print(f"Absolute Zero training completed with {trainer.training_stats}")
        