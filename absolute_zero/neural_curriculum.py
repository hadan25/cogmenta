import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .curriculum_manager import VSACurriculum

class NeuralCurriculum(VSACurriculum):
    """
    Enhanced curriculum manager that uses SNN state information to
    adaptively adjust task difficulty and selection for optimal learning.
    """
    
    def __init__(self, vsa_engine=None, snns=None):
        """
        Initialize the neural curriculum manager.
        
        Args:
            vsa_engine: Optional Vector Symbolic Architecture engine
            snns: Dictionary of SNN adapters
        """
        super().__init__(vsa_engine)
        
        self.snns = snns or {}
        
        # Performance tracking by SNN
        self.task_performance_by_snn = defaultdict(lambda: defaultdict(list))
        
        # Complexity tracking by task type - use defaultdict to avoid KeyError
        self.complexity_by_task_type = defaultdict(lambda: 1.0)
        
        # Initialize default task types
        default_task_types = [
            'pattern', 'classification', 'association', 'perceptual', 
            'reasoning', 'decision', 'memory', 'metacognitive', 'sequence'
        ]
        for task_type in default_task_types:
            self.complexity_by_task_type[task_type] = 1.0
        
        # Learning readiness tracking
        self.snn_readiness = {snn_name: 0.5 for snn_name in self.snns}
        
        # Concept tracking
        self.concept_complexity = defaultdict(lambda: 1.0)
        self.concept_performance = defaultdict(list)
        
        # Curriculum strategies
        self.strategies = [
            'progressive',     # Gradually increase complexity
            'oscillating',     # Alternate between easy and hard tasks
            'spaced',          # Space out task repetition
            'interleaved',     # Mix different task types
            'adaptive'         # Adapt based on performance
        ]
        self.current_strategy = 'adaptive'
        
        # Task selection history
        self.task_history_by_type = defaultdict(list)
        self.last_task_types = []
    
    def select_next_task_type(self) -> str:
        """
        Intelligently select the next task type based on curriculum strategy.
        
        Returns:
            Selected task type
        """
        # Get available task types (assuming they're the keys in complexity_by_task_type)
        available_types = list(self.complexity_by_task_type.keys())
        if not available_types:
            available_types = ['pattern', 'classification', 'association']
        
        # Apply current strategy
        if self.current_strategy == 'progressive':
            # Select the type with the lowest complexity
            return min(available_types, key=lambda t: self.complexity_by_task_type[t])
            
        elif self.current_strategy == 'oscillating':
            # Alternate between easy and hard tasks
            if len(self.last_task_types) % 2 == 0:
                # Select an easy task
                return min(available_types, key=lambda t: self.complexity_by_task_type[t])
            else:
                # Select a challenging task
                return max(available_types, key=lambda t: self.complexity_by_task_type[t])
                
        elif self.current_strategy == 'spaced':
            # Avoid recently used task types
            recent_types = set(self.last_task_types[-3:])
            candidates = [t for t in available_types if t not in recent_types]
            
            if candidates:
                return np.random.choice(candidates)
            
        elif self.current_strategy == 'interleaved':
            # Ensure we interleave different task types
            if self.last_task_types:
                last_type = self.last_task_types[-1]
                candidates = [t for t in available_types if t != last_type]
                
                if candidates:
                    return np.random.choice(candidates)
        
        # Default adaptive strategy or fallback
        # Select the task type with the lowest performance
        avg_performance = {}
        for task_type, performance in self.task_performance_by_snn.items():
            if task_type in available_types:
                # Average performance across all SNNs
                all_scores = []
                for snn_name, scores in performance.items():
                    if scores:
                        all_scores.extend(scores[-10:])  # Use recent scores
                
                if all_scores:
                    avg_performance[task_type] = sum(all_scores) / len(all_scores)
                else:
                    avg_performance[task_type] = 0.5  # Default
        
        # If we have performance data, select lowest performing task type
        if avg_performance:
            # Add some exploration with probability 0.2
            if np.random.random() < 0.2:
                return np.random.choice(available_types)
            
            # Select lowest performing type
            return min(avg_performance.keys(), key=lambda t: avg_performance[t])
        
        # Fallback to random selection
        return np.random.choice(available_types)
    
    def adjust_complexity_for(self, task_type: str, performance: float) -> float:
        """
        Adjust complexity for a specific task type based on performance.
        
        Args:
            task_type: Task type
            performance: Performance on task (0.0 to 1.0)
            
        Returns:
            New complexity
        """
        # Use defaultdict to get current complexity (defaults to 1.0 if not found)
        current_complexity = self.complexity_by_task_type[task_type]
        
        # Adjust based on performance
        if performance > 0.8:
            # Doing well, increase complexity
            new_complexity = current_complexity * 1.1
        elif performance < 0.3:
            # Struggling, decrease complexity
            new_complexity = current_complexity * 0.9
        else:
            # Maintain current complexity
            new_complexity = current_complexity
        
        # Ensure complexity stays within reasonable bounds
        new_complexity = max(0.5, min(10.0, new_complexity))
        
        # Update complexity
        self.complexity_by_task_type[task_type] = new_complexity
        
        return new_complexity
    
    def decrease_complexity_for(self, snn_name: str) -> None:
        """
        Decrease complexity for tasks relevant to specified SNN.
        
        Args:
            snn_name: Name of SNN
        """
        # Find task types relevant to this SNN
        relevant_task_types = self._get_relevant_task_types(snn_name)
        
        # Decrease complexity for all relevant task types
        for task_type in relevant_task_types:
            current = self.complexity_by_task_type[task_type]
            self.complexity_by_task_type[task_type] = max(0.5, current * 0.9)
    
    def increase_complexity_for(self, snn_name: str) -> None:
        """
        Increase complexity for tasks relevant to specified SNN.
        
        Args:
            snn_name: Name of SNN
        """
        # Find task types relevant to this SNN
        relevant_task_types = self._get_relevant_task_types(snn_name)
        
        # Increase complexity for all relevant task types
        for task_type in relevant_task_types:
            current = self.complexity_by_task_type[task_type]
            self.complexity_by_task_type[task_type] = min(10.0, current * 1.1)
    
    def _get_relevant_task_types(self, snn_name: str) -> List[str]:
        """
        Get task types relevant to specified SNN.
        
        Args:
            snn_name: Name of SNN
            
        Returns:
            List of relevant task types
        """
        if snn_name == 'statistical':
            return ['pattern', 'classification', 'association']
        elif snn_name == 'perceptual':
            return ['classification', 'perceptual']
        elif snn_name == 'reasoning':
            return ['reasoning', 'pattern', 'classification']
        elif snn_name == 'decision':
            return ['decision']
        elif snn_name == 'memory':
            return ['memory', 'association']
        elif snn_name == 'metacognitive':
            return ['metacognitive']
        elif snn_name == 'affective':
            return ['decision', 'metacognitive']
        else:
            return []
    
    def adjust_curriculum_by_snn_state(self) -> Dict[str, Any]:
        """
        Adjust curriculum based on state of all SNNs.
        
        Returns:
            Dictionary with curriculum adjustments
        """
        adjustments = {}
        
        # Check each SNN's learning state
        for snn_name, snn in self.snns.items():
            if hasattr(snn, 'get_learning_state'):
                state = snn.get_learning_state()
                
                # Update readiness
                self.snn_readiness[snn_name] = state.get('readiness', 0.5)
                
                # Adjust curriculum based on SNN state
                if state.get('confidence', 0.0) < 0.3 or state.get('accuracy', 0.0) < 0.3:
                    # SNN is struggling, decrease complexity
                    self.decrease_complexity_for(snn_name)
                    adjustments[snn_name] = 'decreased_complexity'
                elif state.get('confidence', 0.0) > 0.8 and state.get('accuracy', 0.0) > 0.8:
                    # SNN is doing well, increase complexity
                    self.increase_complexity_for(snn_name)
                    adjustments[snn_name] = 'increased_complexity'
                else:
                    # Maintain current complexity
                    adjustments[snn_name] = 'maintained_complexity'
        
        # Update overall curriculum strategy based on SNN states
        self._update_curriculum_strategy()
        
        # Include current strategy in adjustments
        adjustments['current_strategy'] = self.current_strategy
        
        return adjustments
    
    def _update_curriculum_strategy(self) -> None:
        """Update curriculum strategy based on overall learning state"""
        # Calculate average readiness across all SNNs
        readiness_values = list(self.snn_readiness.values())
        avg_readiness = sum(readiness_values) / len(readiness_values) if readiness_values else 0.5
        
        # Adjust strategy based on readiness
        if avg_readiness < 0.3:
            # Low readiness - use progressive approach to build foundations
            self.current_strategy = 'progressive'
        elif avg_readiness < 0.6:
            # Medium readiness - use spaced approach to reinforce learning
            self.current_strategy = 'spaced'
        elif avg_readiness < 0.8:
            # Good readiness - use interleaved approach for transfer learning
            self.current_strategy = 'interleaved'
        else:
            # High readiness - use adaptive approach for optimization
            self.current_strategy = 'adaptive'
    
    def get_complexity_for_task(self, task_type: str) -> float:
        """
        Get complexity for a specific task type.
        
        Args:
            task_type: Task type
            
        Returns:
            Complexity value
        """
        return self.complexity_by_task_type[task_type]
    
    def update_after_task(self, task: Dict, metrics: Dict) -> Dict:
        """
        Update curriculum after task completion.
        
        Args:
            task: Task dictionary
            metrics: Performance metrics
            
        Returns:
            Dictionary with curriculum updates
        """
        task_type = task.get('type', 'unknown')
        concept = task.get('metadata', {}).get('concept', 'unknown')
        
        # Extract overall performance
        performance = metrics.get('combined_reward', metrics.get('reward', 0.0))
        
        # Update performance tracking
        for snn_name, snn_reward in metrics.get('individual_rewards', {}).items():
            self.task_performance_by_snn[task_type][snn_name].append(snn_reward)
            
            # Keep history manageable
            if len(self.task_performance_by_snn[task_type][snn_name]) > 100:
                self.task_performance_by_snn[task_type][snn_name] = \
                    self.task_performance_by_snn[task_type][snn_name][-100:]
        
        # Update concept tracking
        self.concept_performance[concept].append(performance)
        if len(self.concept_performance[concept]) > 50:
            self.concept_performance[concept] = self.concept_performance[concept][-50:]
        
        # Adjust concept complexity
        avg_concept_perf = sum(self.concept_performance[concept]) / len(self.concept_performance[concept])
        if avg_concept_perf > 0.8:
            # Doing well on this concept, increase complexity
            self.concept_complexity[concept] = min(10.0, self.concept_complexity[concept] * 1.05)
        elif avg_concept_perf < 0.4:
            # Struggling with this concept, decrease complexity
            self.concept_complexity[concept] = max(0.5, self.concept_complexity[concept] * 0.95)
        
        # Adjust complexity for task type
        new_complexity = self.adjust_complexity_for(task_type, performance)
        
        # Update task history
        self.task_history_by_type[task_type].append({
            'task_id': task.get('id', ''),
            'complexity': task.get('complexity', 1.0),
            'performance': performance,
            'concept': concept
        })
        
        # Keep history manageable
        if len(self.task_history_by_type[task_type]) > 100:
            self.task_history_by_type[task_type] = self.task_history_by_type[task_type][-100:]
        
        # Update last task types
        self.last_task_types.append(task_type)
        if len(self.last_task_types) > 20:
            self.last_task_types = self.last_task_types[-20:]
        
        # Return updated curriculum state
        return {
            'task_type': task_type,
            'concept': concept,
            'performance': performance,
            'new_complexity': new_complexity,
            'strategy': self.current_strategy
        }
    
    def get_curriculum_state(self) -> Dict:
        """
        Get current state of the curriculum.
        
        Returns:
            Dictionary with curriculum state
        """
        # Calculate average performance by task type
        avg_performance = {}
        for task_type, snn_perf in self.task_performance_by_snn.items():
            # Flatten all scores from all SNNs
            all_scores = []
            for scores in snn_perf.values():
                all_scores.extend(scores[-20:] if len(scores) > 20 else scores)
            
            if all_scores:
                avg_performance[task_type] = sum(all_scores) / len(all_scores)
            else:
                avg_performance[task_type] = 0.0
        
        # Calculate average performance by concept
        concept_perf = {}
        for concept, scores in self.concept_performance.items():
            if scores:
                concept_perf[concept] = sum(scores) / len(scores)
        
        # Determine top performing and struggling areas
        top_performing = {}
        struggling = {}
        
        if avg_performance:
            top_task_types = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:3]
            bottom_task_types = sorted(avg_performance.items(), key=lambda x: x[1])[:3]
            
            top_performing['task_types'] = [t[0] for t in top_task_types]
            struggling['task_types'] = [t[0] for t in bottom_task_types]
        
        if concept_perf:
            top_concepts = sorted(concept_perf.items(), key=lambda x: x[1], reverse=True)[:3]
            bottom_concepts = sorted(concept_perf.items(), key=lambda x: x[1])[:3]
            
            top_performing['concepts'] = [c[0] for c in top_concepts]
            struggling['concepts'] = [c[0] for c in bottom_concepts]
        
        # Build the state dictionary
        state = {
            'complexity_by_task_type': dict(self.complexity_by_task_type),
            'concept_complexity': dict(self.concept_complexity),
            'current_strategy': self.current_strategy,
            'performance': {
                'by_task_type': avg_performance,
                'by_concept': concept_perf
            },
            'top_performing': top_performing,
            'struggling': struggling,
            'snn_readiness': dict(self.snn_readiness)
        }
        
        return state
    
    def update(self, task_type: str, reward: float) -> float:
        """
        Update the curriculum based on task performance (override from parent).
        
        Args:
            task_type: The type of task
            reward: The reward received
            
        Returns:
            Updated difficulty level
        """
        # Create a simple metrics dictionary for compatibility
        metrics = {
            'combined_reward': reward,
            'individual_rewards': {
                'statistical': reward
            }
        }
        
        # Create a simple task dictionary for compatibility
        task = {
            'type': task_type,
            'metadata': {
                'concept': task_type
            }
        }
        
        # Update using the enhanced method
        update_result = self.update_after_task(task, metrics)
        
        # Return the new difficulty level for compatibility
        return update_result['new_complexity'] 