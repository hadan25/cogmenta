import time
import numpy as np
from typing import Dict
from .task_generator import CogmentaTaskGenerator
from .reward_calculator import RewardCalculator
from .self_play import CogmentaSelfPlay
from .curriculum_manager import VSACurriculum

class AbsoluteZeroTrainer:
    """Main orchestrator for Absolute Zero training approach"""
    
    def __init__(self, snns, symbolic_engine, vsa_engine):
        self.snns = snns
        self.symbolic_engine = symbolic_engine
        self.vsa_engine = vsa_engine
        
        # Initialize components
        self.task_generator = CogmentaTaskGenerator(symbolic_engine, vsa_engine)
        self.reward_calculator = RewardCalculator()
        self.self_play = CogmentaSelfPlay(
            snns=snns,
            task_generator=self.task_generator,
            reward_calculator=self.reward_calculator
        )
        self.curriculum = VSACurriculum(vsa_engine)
        
        self.training_stats = {
            'iterations': 0,
            'total_reward': 0,
            'accuracy_history': []
        }
    
    def train(self, iterations: int = 1000, log_interval: int = 100):
        """Main training loop"""
        print(f"Starting Absolute Zero training for {iterations} iterations...")
        
        for i in range(iterations):
            # 1. Generate and validate task novelty
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
                
            # 2. Execute self-play iteration
            metrics = self.self_play.self_play_iteration(task)
            
            # 3. Update curriculum with results
            self.curriculum.add_task_result(task, metrics)
            
            # 4. Track statistics
            self.training_stats['iterations'] += 1
            self.training_stats['total_reward'] += metrics['combined_reward']
            self.training_stats['accuracy_history'].append(metrics['accuracy'])
            
            # 5. Affective modulation
            if hasattr(self.snns['affective'], 'influence_processing'):
                affect_state = self.snns['affective'].evaluate_affective_state({
                    'sentiment': metrics['accuracy'] - 0.5,  # Convert to sentiment
                    'intensity': abs(metrics['combined_reward'])
                })
                
                # Apply affective influence to learning
                self.snns['affective'].influence_processing(self.snns['statistical'])
            
            # 6. Logging
            if i % log_interval == 0:
                self._log_progress(i, metrics)
    
    def _log_progress(self, iteration: int, metrics: Dict):
        """Log training progress"""
        recent_accuracy = np.mean(self.training_stats['accuracy_history'][-100:]) if self.training_stats['accuracy_history'] else 0.0
        print(f"Iteration {iteration}:")
        print(f"  - Recent accuracy: {recent_accuracy:.3f}")
        print(f"  - Current difficulty: {self.curriculum.get_difficulty_level():.2f}")
        print(f"  - Last reward: {metrics['combined_reward']:.3f}")
        print(f"  - Task type: {metrics['task_type']}")