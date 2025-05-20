
import os
import sys
import traceback

# Ensure correct path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    # Verify imports are working
    from absolute_zero.zero_trainer import AbsoluteZeroTrainer
    from absolute_zero.run_absolute_zero import setup_mock_components, create_save_directory
    from absolute_zero.snn_adapter import create_snn_components
    
    print("Successfully imported Absolute Zero modules")
except ImportError as e:
    print(f"Error importing Absolute Zero modules: {e}")
    print(f"Paths: current_dir={current_dir}, parent_dir={parent_dir}")
    print(f"sys.path={sys.path}")
    print(traceback.format_exc())
    sys.exit(1)

# Create components
snns = create_snn_components(use_real_snn=False)
_, symbolic_engine, vsa_engine = setup_mock_components()

# Create save directory
save_dir = "C:\\Users\\ha25a\\.vscode\\cogmenta_core\\fixed_hybrid_results\\all_20250517_142033\\affective\\cycle_1\\absolute"
os.makedirs(save_dir, exist_ok=True)

# Replace the specified SNN with a real one if possible
model_type = "affective"
try:
    if model_type == "memory":
        from models.snn.memory_snn import MemorySNN
        from absolute_zero.snn_adapter import MemorySNNAdapter
        
        class CustomMemorySNNAdapter(MemorySNNAdapter):
            def __init__(self, memory_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = memory_snn
                self.stored_memories = []
        
        model = MemorySNN(neuron_count=129721)
        snns["memory"] = CustomMemorySNNAdapter(model, use_real_snn=True)
        print(f"Created real Memory SNN adapter")
        
    elif model_type == "decision":
        from models.snn.decision_snn import DecisionSNN
        from absolute_zero.snn_adapter import DecisionSNNAdapter
        
        class CustomDecisionSNNAdapter(DecisionSNNAdapter):
            def __init__(self, decision_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = decision_snn
        
        model = DecisionSNN(neuron_count=129721)
        snns["decision"] = CustomDecisionSNNAdapter(model, use_real_snn=True)
        print(f"Created real Decision SNN adapter")
        
    elif model_type == "metacognitive":
        from models.snn.metacognitive_snn import MetacognitiveSNN
        from absolute_zero.snn_adapter import MetacognitiveSNNAdapter
        
        class CustomMetacognitiveSNNAdapter(MetacognitiveSNNAdapter):
            def __init__(self, metacognitive_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = metacognitive_snn
        
        model = MetacognitiveSNN(neuron_count=129721)
        snns["metacognitive"] = CustomMetacognitiveSNNAdapter(model, use_real_snn=True)
        print(f"Created real Metacognitive SNN adapter")
        
    elif model_type == "statistical":
        from models.snn.statistical_snn import StatisticalSNN
        from absolute_zero.snn_adapter import StatisticalSNNAdapter
        
        class CustomStatisticalSNNAdapter(StatisticalSNNAdapter):
            def __init__(self, statistical_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = statistical_snn
        
        model = StatisticalSNN(input_size=64, hidden_size=128, output_size=32)
        snns["statistical"] = CustomStatisticalSNNAdapter(model, use_real_snn=True)
        print(f"Created real Statistical SNN adapter")
        
    elif model_type == "perceptual":
        from models.snn.perceptual_snn import PerceptualSNN
        from absolute_zero.snn_adapter import PerceptualSNNAdapter
        
        class CustomPerceptualSNNAdapter(PerceptualSNNAdapter):
            def __init__(self, perceptual_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = perceptual_snn
        
        model = PerceptualSNN(neuron_count=129721)
        snns["perceptual"] = CustomPerceptualSNNAdapter(model, use_real_snn=True)
        print(f"Created real Perceptual SNN adapter")
        
    elif model_type == "reasoning":
        from models.snn.reasoning_snn import ReasoningSNN
        from absolute_zero.snn_adapter import ReasoningSNNAdapter
        
        class CustomReasoningSNNAdapter(ReasoningSNNAdapter):
            def __init__(self, reasoning_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = reasoning_snn
        
        model = ReasoningSNN(neuron_count=129721)
        snns["reasoning"] = CustomReasoningSNNAdapter(model, use_real_snn=True)
        print(f"Created real Reasoning SNN adapter")
        
    elif model_type == "affective":
        from models.snn.affective_snn import AffectiveSNN
        from absolute_zero.snn_adapter import AffectiveSNNAdapter
        
        class CustomAffectiveSNNAdapter(AffectiveSNNAdapter):
            def __init__(self, affective_snn, use_real_snn=True):
                self.use_real_snn = use_real_snn
                self.snn = affective_snn
        
        model = AffectiveSNN(neuron_count=129721)
        snns["affective"] = CustomAffectiveSNNAdapter(model, use_real_snn=True)
        print(f"Created real Affective SNN adapter")
        
except Exception as e:
    print(f"Error creating real SNN adapter: {e}")
    print("Using mock implementation instead")

# Create trainer
trainer = AbsoluteZeroTrainer(snns, symbolic_engine, vsa_engine)

# Run training
iterations = 100
trainer.train(iterations=iterations, log_interval=max(1, iterations//10))

# Save results
print(f"Absolute Zero training completed with {trainer.training_stats}")

# Save metrics
metrics_file = os.path.join(save_dir, 'absolute_zero_metrics.json')
with open(metrics_file, 'w') as f:
    import json
    metrics = {
        'accuracy_history': trainer.training_stats.get('accuracy_history', []),
        'total_reward': trainer.training_stats.get('total_reward', 0.0),
        'iterations': trainer.training_stats.get('iterations', 0)
    }
    json.dump(metrics, f, indent=2)
        