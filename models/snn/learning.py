# cogmenta_core/models/snn/learning.py
import numpy as np
import time
import random
from collections import deque

class LearningModule:
    """
    Dynamic learning module for spiking neural networks.
    
    Implements various learning algorithms:
    - Spike-Timing-Dependent Plasticity (STDP)
    - Hebbian learning ("neurons that fire together, wire together")
    - Reinforcement learning with feedback
    - Homeostatic plasticity for stability
    """
    
    def __init__(self, snn, learning_rate=0.01):
        self.snn = snn  # Reference to the spiking neural network
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.stdp_window = 20  # ms, temporal window for STDP
        self.hebbian_rate = 0.02  # Rate for Hebbian learning
        self.homeostatic_rate = 0.005  # Rate for homeostatic adjustments
        
        # Activity tracking
        self.neuron_activity = np.zeros(snn.neuron_count)
        self.target_activity = 0.1  # Target average activity level
        
        # Spike timing tracking (for STDP)
        self.last_spike_times = np.full(snn.neuron_count, -1000.0)  # Initialize with large negative values
        
        # Plasticity factors (control learning for different neurons)
        self.plasticity_factors = np.random.uniform(0.8, 1.2, snn.neuron_count)
        
        # Learning history for analysis
        self.learning_events = deque(maxlen=1000)
        
        # Feedback history for reinforcement learning
        self.feedback_history = deque(maxlen=100)
        
        print(f"[Learning] Neural learning module initialized (learning rate={learning_rate})")
    
    def apply_stdp(self, pre_neuron, post_neuron, current_time):
        """
        Apply Spike-Timing-Dependent Plasticity between two neurons
        
        Args:
            pre_neuron: Index of presynaptic neuron
            post_neuron: Index of postsynaptic neuron
            current_time: Current simulation time
            
        Returns:
            Weight change amount
        """
        # Get the spike times
        pre_spike_time = self.last_spike_times[pre_neuron]
        post_spike_time = self.last_spike_times[post_neuron]
        
        # Calculate time differences
        time_diff = post_spike_time - pre_spike_time
        
        # Apply STDP learning rule
        if abs(time_diff) > self.stdp_window:
            # Spikes too far apart in time
            return 0.0
            
        if time_diff > 0:
            # Post-synaptic neuron fired after pre-synaptic -> strengthen connection
            weight_change = self.learning_rate * np.exp(-time_diff / 10.0)
        else:
            # Post-synaptic neuron fired before pre-synaptic -> weaken connection
            weight_change = -self.learning_rate * np.exp(time_diff / 10.0)
            
        # Apply plasticity factor
        weight_change *= self.plasticity_factors[post_neuron]
        
        # Record learning event
        self.learning_events.append({
            'type': 'stdp',
            'pre': pre_neuron,
            'post': post_neuron,
            'time_diff': time_diff,
            'weight_change': weight_change,
            'timestamp': current_time
        })
        
        return weight_change
    
    def apply_hebbian_learning(self, active_neurons, strength=1.0):
        """
        Apply Hebbian learning to strengthen connections between co-active neurons
        
        Args:
            active_neurons: List of currently active neuron indices
            strength: Learning strength multiplier
            
        Returns:
            Number of connections modified
        """
        if len(active_neurons) < 2:
            return 0  # Need at least 2 neurons to form connections
        
        connections_modified = 0
        
        # Strengthen connections between all pairs of active neurons
        for i in range(len(active_neurons)):
            for j in range(i+1, len(active_neurons)):
                pre = active_neurons[i]
                post = active_neurons[j]
                
                # Calculate weight increase based on Hebbian rule
                weight_increase = self.hebbian_rate * strength * self.plasticity_factors[post]
                
                # Update weight in both directions (symmetric)
                old_weight = self.snn.synaptic_weights[pre, post]
                self.snn.synaptic_weights[pre, post] = min(1.0, old_weight + weight_increase)
                self.snn.synaptic_weights[post, pre] = min(1.0, old_weight + weight_increase)
                
                connections_modified += 1
                
                # Record learning event
                self.learning_events.append({
                    'type': 'hebbian',
                    'pre': pre,
                    'post': post,
                    'weight_increase': weight_increase,
                    'new_weight': self.snn.synaptic_weights[pre, post],
                    'timestamp': time.time()
                })
        
        return connections_modified
    
    def apply_homeostatic_plasticity(self):
        """
        Apply homeostatic plasticity to maintain stability
        This prevents runaway excitation or silencing of neurons
        
        Returns:
            Number of neurons adjusted
        """
        # Calculate activity deviation from target
        activity_deviation = self.neuron_activity - self.target_activity
        
        # Adjust neurons that are too active or too silent
        adjustments_needed = np.abs(activity_deviation) > 0.05
        neurons_to_adjust = np.where(adjustments_needed)[0]
        
        if len(neurons_to_adjust) == 0:
            return 0
        
        for neuron_idx in neurons_to_adjust:
            # If too active, decrease incoming weights
            if activity_deviation[neuron_idx] > 0:
                # Find incoming connections
                incoming = self.snn.synaptic_weights[:, neuron_idx]
                # Scale down weights
                scale_factor = 1.0 - (self.homeostatic_rate * activity_deviation[neuron_idx])
                self.snn.synaptic_weights[:, neuron_idx] = incoming * scale_factor
            # If too silent, increase incoming weights
            else:
                # Find incoming connections
                incoming = self.snn.synaptic_weights[:, neuron_idx]
                # Scale up weights
                scale_factor = 1.0 + (self.homeostatic_rate * abs(activity_deviation[neuron_idx]))
                self.snn.synaptic_weights[:, neuron_idx] = incoming * scale_factor
            
            # Record homeostatic adjustment
            self.learning_events.append({
                'type': 'homeostatic',
                'neuron': neuron_idx,
                'activity': self.neuron_activity[neuron_idx],
                'deviation': activity_deviation[neuron_idx],
                'timestamp': time.time()
            })
        
        return len(neurons_to_adjust)
    
    def process_feedback(self, feedback_value, active_neurons=None):
        """
        Process feedback (reinforcement) to adjust neuron weights
        
        Args:
            feedback_value: Feedback value (-1 to 1, negative is punishment, positive is reward)
            active_neurons: List of neurons that were active during the action receiving feedback
            
        Returns:
            Number of weights adjusted
        """
        # Store feedback in history
        self.feedback_history.append({
            'value': feedback_value,
            'timestamp': time.time()
        })
        
        # Default to neurons with recent activity if not specified
        if active_neurons is None:
            active_threshold = 0.3
            active_neurons = np.where(self.neuron_activity > active_threshold)[0]
        
        if len(active_neurons) == 0:
            return 0
        
        # Determine weight adjustment based on feedback
        if feedback_value > 0:
            # Positive feedback - reinforce connections between active neurons
            adjustment_factor = feedback_value * self.learning_rate * 2.0
            connections_adjusted = self.apply_hebbian_learning(active_neurons, strength=adjustment_factor)
            
            # Also slightly increase excitability of these neurons
            for neuron in active_neurons:
                self.plasticity_factors[neuron] = min(1.5, self.plasticity_factors[neuron] * (1.0 + 0.01 * feedback_value))
        else:
            # Negative feedback - weaken connections between active neurons
            adjustment_factor = abs(feedback_value) * self.learning_rate
            connections_adjusted = 0
            
            # Weaken connections between all pairs of active neurons
            for i in range(len(active_neurons)):
                for j in range(len(active_neurons)):
                    if i != j:
                        pre = active_neurons[i]
                        post = active_neurons[j]
                        
                        # Decrease weight
                        old_weight = self.snn.synaptic_weights[pre, post]
                        new_weight = max(0.0, old_weight - adjustment_factor)
                        self.snn.synaptic_weights[pre, post] = new_weight
                        
                        connections_adjusted += 1
            
            # Also slightly decrease excitability of these neurons
            for neuron in active_neurons:
                self.plasticity_factors[neuron] = max(0.5, self.plasticity_factors[neuron] * (1.0 + 0.01 * feedback_value))
        
        # Record learning event
        self.learning_events.append({
            'type': 'reinforcement',
            'feedback': feedback_value,
            'active_neurons': len(active_neurons),
            'connections_adjusted': connections_adjusted,
            'timestamp': time.time()
        })
        
        return connections_adjusted
    
    def update_activity_trace(self, spike_data):
        """
        Update neuron activity trace based on recent spikes
        
        Args:
            spike_data: List of (neuron_idx, spike_strength) tuples
            
        Returns:
            Updated neuron activity array
        """
        # Apply decay to all activity
        self.neuron_activity *= 0.9  # 10% decay
        
        # Update activity for neurons that spiked
        current_time = time.time()
        for neuron_idx, spike_strength in spike_data:
            # Update last spike time for STDP
            self.last_spike_times[neuron_idx] = current_time
            
            # Increase activity trace
            self.neuron_activity[neuron_idx] = min(1.0, self.neuron_activity[neuron_idx] + (0.3 * spike_strength))
        
        return self.neuron_activity
    
    def train_pattern(self, input_pattern, target_pattern, epochs=10):
        """
        Train the network to associate an input pattern with a target pattern
        
        Args:
            input_pattern: Input activation pattern
            target_pattern: Target activation pattern
            epochs: Number of training iterations
            
        Returns:
            Training statistics
        """
        stats = {
            'epochs': epochs,
            'errors': [],
            'final_error': 0.0
        }
        
        for epoch in range(epochs):
            # Activate input pattern
            self.snn.apply_activation_pattern(input_pattern)
            
            # Run simulation to get output pattern
            spike_patterns = self.snn.simulate_spiking(input_pattern)
            
            # Get final output activation
            output_pattern = self.snn.get_current_activation()
            
            # Calculate error
            error = np.mean(np.abs(output_pattern - target_pattern))
            stats['errors'].append(error)
            
            # Identify active neurons in input and target
            input_active = np.where(input_pattern > 0.5)[0]
            target_active = np.where(target_pattern > 0.5)[0]
            
            # Strengthen connections from input to target neurons
            for in_neuron in input_active:
                for out_neuron in target_active:
                    # Increase weight based on error
                    weight_increase = self.learning_rate * (1.0 - error)
                    old_weight = self.snn.synaptic_weights[in_neuron, out_neuron]
                    self.snn.synaptic_weights[in_neuron, out_neuron] = min(1.0, old_weight + weight_increase)
            
            # Apply homeostatic plasticity occasionally
            if epoch % 5 == 0:
                self.apply_homeostatic_plasticity()
        
        # Final error
        stats['final_error'] = stats['errors'][-1]
        
        return stats
    
    def get_learning_stats(self):
        """
        Get statistics about recent learning events
        
        Returns:
            Dictionary of learning statistics
        """
        stats = {
            'total_events': len(self.learning_events),
            'event_types': {},
            'recent_events': list(self.learning_events)[-10:] if self.learning_events else [],
            'average_feedback': sum(f['value'] for f in self.feedback_history) / len(self.feedback_history) if self.feedback_history else 0.0,
            'feedback_count': len(self.feedback_history)
        }
        
        # Count event types
        for event in self.learning_events:
            event_type = event['type']
            if event_type not in stats['event_types']:
                stats['event_types'][event_type] = 0
            stats['event_types'][event_type] += 1
        
        return stats