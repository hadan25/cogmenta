"""
Enhanced Absolute Zero Trainer with comprehensive SNN integration.

This trainer extends the basic AbsoluteZeroTrainer to incorporate more
SNN components from the models/snn directory, implementing a full
neuro-symbolic learning loop with all neural components.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
import os
import sys

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Absolute Zero components
from .zero_trainer import AbsoluteZeroTrainer
from .task_generator import CogmentaTaskGenerator
from .reward_calculator import RewardCalculator
from .self_play import CogmentaSelfPlay
from .curriculum_manager import VSACurriculum
from .snn_adapter import create_snn_components

# No need to directly import SNN modules anymore, as we'll use adapters


class EnhancedAbsoluteZeroTrainer(AbsoluteZeroTrainer):
    """Enhanced trainer that integrates all SNN components."""
    
    def __init__(self, snns=None, symbolic_engine=None, vsa_engine=None, use_real_snns=False):
        """
        Initialize the enhanced trainer.
        
        Args:
            snns: Pre-created SNN components (if None, they will be created)
            symbolic_engine: Symbolic engine for symbolic reasoning
            vsa_engine: Vector symbolic architecture engine
            use_real_snns: Whether to use real SNNs (only used if snns is None)
        """
        # Create SNNs if not provided
        if snns is None:
            print("No SNNs provided, creating them with use_real_snns =", use_real_snns)
            snns = create_snn_components(use_real_snns)
        
        # Initialize the base trainer
        super().__init__(snns, symbolic_engine, vsa_engine)
        
        # Add enhanced tracking metrics
        self.training_stats.update({
            'reasoning_accuracy': [],
            'perceptual_accuracy': [],
            'memory_utilization': [],
            'decision_confidence': []
        })
        
        # Print SNN status
        print("\nEnhanced Absolute Zero Components Status:")
        print("----------------------------------------")
        for name, adapter in self.snns.items():
            using_real = hasattr(adapter, 'use_real_snn') and adapter.use_real_snn
            status = "REAL" if using_real else "MOCK"
            print(f"- {name.upper()} SNN: {status}")
        print("")
    
    def train(self, iterations: int = 1000, log_interval: int = 100):
        """Enhanced training loop with full SNN integration."""
        print(f"Starting Enhanced Absolute Zero training for {iterations} iterations...")
        
        for i in range(iterations):
            # 1. Generate and validate task novelty (same as base trainer)
            novel_task = False
            attempts = 0
            task = None
            
            while not novel_task and attempts < 10:
                # Adjust task difficulty based on curriculum
                if self.curriculum.adjust_curriculum() == 'increase_difficulty':
                    self.task_generator.increase_complexity()
                
                # Generate task
                task = self.task_generator.generate_task()
                novel_task = self.curriculum.is_novel_task(task)
                attempts += 1
            
            if task is None:
                continue
            
            # 2. Process task through the perceptual and memory SNNs
            self._process_task_perception(task)
            
            # 3. Execute self-play iteration (enhanced version)
            metrics = self._enhanced_self_play_iteration(task)
            
            # 4. Update curriculum with results
            self.curriculum.add_task_result(task, metrics)
            
            # 5. Track statistics
            self._update_training_stats(metrics)
            
            # 6. Affective modulation
            self._apply_affective_modulation(metrics)
            
            # 7. Logging
            if i % log_interval == 0:
                self._enhanced_log_progress(i, metrics)
    
    def _process_task_perception(self, task: Dict):
        """Process the task through perceptual and memory SNNs."""
        if 'perceptual' in self.snns:
            # Encode the task using the perceptual SNN
            task_input = str(task.get('input', {}))
            encoded_perception = self.snns['perceptual'].process(task_input)
            
            # Store perceived representation
            self.current_perception = encoded_perception
            
            # Store in memory if available
            if 'memory' in self.snns:
                self.snns['memory'].store({
                    'task_type': task.get('type', 'unknown'),
                    'perception': encoded_perception,
                    'timestamp': time.time()
                })
    
    def _enhanced_self_play_iteration(self, task: Dict) -> Dict:
        """Enhanced self-play iteration using all SNN components."""
        # 1. Use reasoning SNN if available for logical tasks
        reasoning_result = None
        if task.get('type') in ['deduction', 'abduction', 'induction'] and 'reasoning' in self.snns:
            reasoning_input = {
                'rules': task.get('input', {}).get('rules', []),
                'facts': task.get('input', {}).get('facts', []),
                'type': task.get('type')
            }
            reasoning_result = self.snns['reasoning'].process(reasoning_input)
        
        # 2. Use decision SNN to select action if available
        selected_action = None
        if 'decision' in self.snns:
            # Prepare context for decision
            decision_context = {
                'task_type': task.get('type', 'unknown'),
                'complexity': task.get('complexity', 1.0),
                'reasoning_result': reasoning_result
            }
            selected_action = self.snns['decision'].select_action(decision_context)
        
        # 3. Use base self-play to handle the core learning loop
        base_metrics = self.self_play.self_play_iteration(task)
        
        # 4. Enhance metrics with extended SNN information
        enhanced_metrics = dict(base_metrics)
        
        if reasoning_result:
            # Compare reasoning result with actual output
            reasoning_correct = self._verify_reasoning(reasoning_result, task.get('output', {}))
            enhanced_metrics['reasoning_accuracy'] = float(reasoning_correct)
            
            # Update reasoning SNN
            reasoning_reward = base_metrics.get('accuracy', 0.0)
            self.snns['reasoning'].update(reasoning_reward)
        
        if selected_action:
            # Add decision metrics
            enhanced_metrics['decision_confidence'] = selected_action.get('confidence', 0.5)
            
            # Update decision SNN
            decision_reward = base_metrics.get('combined_reward', 0.0)
            self.snns['decision'].update_policy(decision_reward)
        
        # Add memory utilization if available
        if 'memory' in self.snns:
            memory_stats = self.snns['memory'].get_stats()
            enhanced_metrics['memory_utilization'] = memory_stats.get('utilization', 0.0)
        
        return enhanced_metrics
    
    def _verify_reasoning(self, reasoning_result: Any, expected_output: Any) -> bool:
        """Verify if the reasoning result matches the expected output."""
        # Simple verification for demonstration
        if isinstance(reasoning_result, dict) and isinstance(expected_output, dict):
            return reasoning_result == expected_output
        elif isinstance(reasoning_result, list) and isinstance(expected_output, list):
            return set(reasoning_result) == set(expected_output)
        else:
            return str(reasoning_result) == str(expected_output)
    
    def _update_training_stats(self, metrics: Dict):
        """Update training statistics with enhanced metrics."""
        # Update base stats
        self.training_stats['iterations'] += 1
        self.training_stats['total_reward'] += metrics.get('combined_reward', 0.0)
        self.training_stats['accuracy_history'].append(metrics.get('accuracy', 0.0))
        
        # Update extended stats
        if 'reasoning_accuracy' in metrics:
            self.training_stats['reasoning_accuracy'].append(metrics['reasoning_accuracy'])
        
        if 'decision_confidence' in metrics:
            self.training_stats['decision_confidence'].append(metrics['decision_confidence'])
        
        if 'memory_utilization' in metrics:
            self.training_stats['memory_utilization'].append(metrics['memory_utilization'])
    
    def _apply_affective_modulation(self, metrics: Dict):
        """Apply affective modulation to the learning process."""
        if 'affective' in self.snns and hasattr(self.snns['affective'], 'influence_processing'):
            # Create affective input
            affect_input = {
                'sentiment': metrics.get('accuracy', 0.5) - 0.5,  # Convert to sentiment
                'intensity': abs(metrics.get('combined_reward', 0.0))
            }
            
            # Get affective state
            affect_state = self.snns['affective'].evaluate_affective_state(affect_input)
            
            # Apply modulation to SNNs
            for snn_name, snn in self.snns.items():
                if snn_name != 'affective' and hasattr(snn, 'set_affective_state'):
                    try:
                        snn.set_affective_state(affect_state)
                    except Exception as e:
                        print(f"Error applying affective modulation to {snn_name}: {e}")
    
    def _enhanced_log_progress(self, iteration: int, metrics: Dict):
        """Enhanced logging with additional SNN metrics."""
        # Calculate recent metrics
        recent_accuracy = np.mean(self.training_stats['accuracy_history'][-100:]) if self.training_stats['accuracy_history'] else 0.0
        
        print(f"Iteration {iteration}:")
        print(f"  - Recent accuracy: {recent_accuracy:.3f}")
        print(f"  - Current difficulty: {self.curriculum.get_difficulty_level():.2f}")
        print(f"  - Last reward: {metrics['combined_reward']:.3f}")
        
        # Add enhanced metrics
        if 'reasoning_accuracy' in metrics:
            print(f"  - Reasoning accuracy: {metrics['reasoning_accuracy']:.3f}")
        if 'decision_confidence' in metrics:
            print(f"  - Decision confidence: {metrics['decision_confidence']:.3f}")
        if 'memory_utilization' in metrics:
            print(f"  - Memory utilization: {metrics['memory_utilization']:.3f}")
        
        print(f"  - Task type: {metrics['task_type']}")
        
    def emphasize_component(self, component_name: str):
        """Emphasize a specific component in the training process."""
        if component_name not in self.snns:
            print(f"Warning: Cannot emphasize {component_name} - component not found")
            return
            
        print(f"Emphasizing {component_name} SNN component")
        
        # Simple way to emphasize - adjust parameters based on component type
        if component_name == 'statistical' and hasattr(self.snns['statistical'], 'snn') and hasattr(self.snns['statistical'].snn, 'set_learning_rate'):
            self.snns['statistical'].snn.set_learning_rate(0.02)  # Double default
            
        elif component_name == 'metacognitive' and hasattr(self.curriculum, 'set_adaptation_rate'):
            self.curriculum.set_adaptation_rate(0.2)  # Increase adaptation rate
            
        elif component_name == 'memory' and 'memory' in self.snns:
            # Increase memory retention
            if hasattr(self.snns['memory'], 'set_retention'):
                self.snns['memory'].set_retention(0.9)
                
        elif component_name == 'reasoning' and 'reasoning' in self.snns:
            # Increase logical weight
            if hasattr(self.snns['reasoning'], 'set_logical_weight'):
                self.snns['reasoning'].set_logical_weight(2.0)
                
        # More component-specific emphasis could be added


def create_trainer(snns=None, symbolic_engine=None, vsa_engine=None, use_real_snns=False):
    """
    Factory function to create a trainer instance.
    
    Args:
        snns: Pre-created SNN components (optional)
        symbolic_engine: Symbolic engine for symbolic reasoning
        vsa_engine: VSA engine for vector symbolic operations
        use_real_snns: Whether to use real SNNs (only used if snns is None)
        
    Returns:
        An instance of EnhancedAbsoluteZeroTrainer
    """
    # Create the enhanced trainer
    return EnhancedAbsoluteZeroTrainer(
        snns=snns,
        symbolic_engine=symbolic_engine,
        vsa_engine=vsa_engine,
        use_real_snns=use_real_snns
    ) 