# cogmenta_core/models/snn/metacognitive_snn.py
import numpy as np
import random
import time
import math
from collections import defaultdict, deque
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from models.snn.enhanced_snn import EnhancedSpikingCore

class MetacognitiveSNN(EnhancedSpikingCore):
    """
    Metacognitive Spiking Neural Network for system self-monitoring and regulation.
    
    This component is specialized for:
    - Monitoring system processes and states across components
    - Detecting contradictions, inconsistencies, and reasoning failures
    - Evaluating confidence and uncertainty across different subsystems
    - Regulating cognitive processes (attention, strategy selection)
    - Suggesting adaptations based on system performance
    - Learning from previous reasoning episodes
    
    Works alongside other specialized SNNs and the symbolic system to provide
    metacognitive capabilities for the overall architecture.
    """
    
    def __init__(self, neuron_count=1200, meta_regions=None, learning_rate=0.01, topology_type="small_world", vector_dim=300, bidirectional_processor=None):
        """
        Initialize the metacognitive SNN.
        
        Args:
            neuron_count: Number of neurons in the network
            meta_regions: Custom region definitions (optional)
            learning_rate: Base learning rate for plasticity
            topology_type: Type of network topology
            vector_dim: Dimension of the vector space for token embeddings
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
        """
        # Initialize parent class
        super().__init__(
            neuron_count=neuron_count,
            topology_type=topology_type,
            model_type="metacognitive",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )
        
        # Store learning rate
        self.learning_rate = learning_rate
        
        # Metacognitive state tracking
        self.metacognitive_state = {
            'system_confidence': 0.5,
            'detected_contradictions': [],
            'uncertainty_estimates': {},
            'attention_focus': None,
            'reasoning_quality': 0.5,
            'self_regulation_signals': {},
            'metacognitive_alerts': []
        }
        
        # Initialize specialized metacognitive regions using parent's region system
        self._init_metacognitive_regions(meta_regions)
        
        # Knowledge about component states and interactions
        self.component_state_history = {}  # Tracks states of other components
        self.component_performance = {}  # Tracks performance metrics
        
        # Specialized memory for metacognition
        self.reasoning_episodes = deque(maxlen=200)  # Previous reasoning episodes
        self.detected_patterns = {}  # Patterns in reasoning and failures
        self.strategy_evaluations = {}  # Effectiveness of different strategies
        
        # Integration quality tracking
        self.phi_history = deque(maxlen=50)  # Track integration level (Φ) over time
        self.current_phi = 0.1  # Current integration level
        
        # Define input and output layers if not already defined by parent
        if not hasattr(self, 'input_layer'):
            self.input_layer = list(range(int(neuron_count * 0.1)))  # First 10% of neurons
        if not hasattr(self, 'output_layer'):
            self.output_layer = list(range(neuron_count - int(neuron_count * 0.1), neuron_count))  # Last 10%
        
        # Training data for metacognitive processing
        self.training_data = []  # (input_pattern, target_output) pairs
        
        # Region-specific decoders for interpreting output patterns
        self._init_decoders()
        
        print(f"[MetaSNN] Initialized Metacognitive SNN with {neuron_count} neurons")
    
    def _init_decoders(self):
        """Initialize neural decoders for interpreting output patterns."""
        # Each decoder maps ranges of output neuron activations to specific metacognitive outputs
        self.decoders = {
            # Confidence decoder (maps output neuron activation to confidence levels)
            'confidence': {
                'neurons': self.regions['confidence_estimation']['neurons'][-20:],  # Last 20 neurons of confidence region
                'thresholds': {  # Activation thresholds for different confidence levels
                    'high': 0.7,
                    'medium': 0.4,
                    'low': 0.2
                },
                'ranges': {  # Maps activation ranges to specific values
                    (0.0, 0.2): 0.1,  # Very low confidence
                    (0.2, 0.4): 0.3,
                    (0.4, 0.6): 0.5,
                    (0.6, 0.8): 0.7,
                    (0.8, 1.0): 0.9   # Very high confidence
                }
            },
            
            # Issue detection decoder
            'issues': {
                'neurons': self.regions['contradiction_detection']['neurons'][-30:],
                'thresholds': {
                    'detection': 0.6  # Threshold for detecting an issue
                },
                'issue_types': {  # Maps neuron indices to issue types
                    0: 'low_integration',
                    1: 'subsystem_imbalance',
                    2: 'contradictions',
                    3: 'processing_loop',
                    4: 'logical_inconsistency',
                    5: 'shallow_reasoning'
                }
            },
            
            # Attention decoder
            'attention': {
                'neurons': self.regions['attention_regulation']['neurons'][-10:],
                'thresholds': {
                    'focus_shift': 0.6  # Threshold for triggering focus shift
                }
            },
            
            # Strategy decoder
            'strategy': {
                'neurons': self.regions['strategy_selection']['neurons'][-20:],
                'thresholds': {
                    'adjustment': 0.65  # Threshold for suggesting strategy adjustment
                },
                'strategies': {  # Maps neuron indices to strategies
                    0: 'increase_recurrent_loops',
                    1: 'change_reasoning_approach',
                    2: 'verify_outputs',
                    3: 'increase_integration',
                    4: 'boost_subsystem_interaction'
                }
            }
        }
    
    def process_input(self, input_pattern, timesteps=10):
        """
        Process an input pattern through the SNN.
        
        Args:
            input_pattern: Input activation pattern
            timesteps: Number of simulation steps
            
        Returns:
            Dictionary with processing results including output pattern
        """
        # Reset membrane potentials
        self.membrane_potentials = np.zeros(self.neuron_count)
        
        # Apply input pattern to input layer
        self._apply_input_pattern(input_pattern)
        
        # Simulate network activity
        spike_patterns = self._simulate_network(timesteps)
        
        # Get final output pattern
        output_pattern = self._get_output_pattern()
        
        # Update region activations
        self._update_region_activations(spike_patterns)
        
        return {
            'output_pattern': output_pattern,
            'spikes': spike_patterns,
            'region_activations': {r: info['activation'] for r, info in self.regions.items()}
        }
    
    def _apply_input_pattern(self, pattern):
        """
        Apply input pattern to the network's input layer.
        
        Args:
            pattern: Input activation pattern
        """
        # Ensure pattern has the right dimensionality
        pattern_len = len(pattern)
        input_len = len(self.input_layer)
        
        # Resize pattern if needed
        if pattern_len != input_len:
            if pattern_len < input_len:
                # Pad with zeros
                pattern = np.pad(pattern, (0, input_len - pattern_len))
            else:
                # Truncate
                pattern = pattern[:input_len]
        
        # Apply pattern to input layer neurons
        for i, neuron_idx in enumerate(self.input_layer):
            self.membrane_potentials[neuron_idx] = pattern[i]
    
    def _get_output_pattern(self):
        """
        Get the current output pattern from the output layer.
        
        Returns:
            Output activation pattern
        """
        return np.array([self.membrane_potentials[i] for i in self.output_layer])
    
    def _simulate_network(self, steps=10):
        """
        Simulate network activity for a given number of steps.
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            List of spike events per step
        """
        # Record spikes for each step
        spikes = []
        
        # Current simulation time
        current_time = time.time()
        
        for t in range(steps):
            # Neurons that spike this step
            spiking_neurons = np.where(
                self.membrane_potentials >= self.spike_thresholds
            )[0]
            
            if len(spiking_neurons) > 0:
                # Record spike times for STDP
                self.last_spike_times[spiking_neurons] = current_time + 0.001 * t
                
                # Record spikes
                step_spikes = [(int(i), float(self.membrane_potentials[i])) for i in spiking_neurons]
                spikes.append(step_spikes)
                
                # Reset membrane potential of spiking neurons
                self.membrane_potentials[spiking_neurons] = 0.0
                
                # Propagate to connected neurons (using sparse matrix multiplication)
                # Extract columns of synaptic weights corresponding to spiking neurons
                post_weights = self.synaptic_weights[:, spiking_neurons]
                # Sum across columns to get total input to each neuron
                delta_potentials = np.array(post_weights.sum(axis=1)).flatten()
                # Update membrane potentials
                self.membrane_potentials += delta_potentials
            else:
                # No spikes this step
                spikes.append([])
            
            # Apply decay
            self.membrane_potentials *= self.decay_rate
            
            # Apply STDP learning for neurons that spiked
            if len(spiking_neurons) > 1:
                self._apply_stdp(spiking_neurons, current_time + 0.001 * t)
        
        return spikes
    
    def _apply_stdp(self, spiking_neurons, current_time):
        """
        Apply Spike-Timing-Dependent Plasticity to update weights.
        
        Args:
            spiking_neurons: Indices of neurons that spiked
            current_time: Current simulation time
        """
        for post in spiking_neurons:
            # Update incoming connections to this neuron
            for pre in range(self.neuron_count):
                if pre != post:  # Skip self-connections
                    pre_spike_time = self.last_spike_times[pre]
                    time_diff = current_time - pre_spike_time
                    
                    # Skip if pre neuron hasn't spiked recently or time difference is too large
                    if pre_spike_time <= 0 or abs(time_diff) > self.stdp_window:
                        continue
                    
                    # Calculate weight change
                    if time_diff > 0:  # Post spiked after pre (potentiation)
                        weight_change = self.learning_rate * np.exp(-time_diff / (self.stdp_window / 2))
                    else:  # Post spiked before pre (depression)
                        weight_change = -self.learning_rate * np.exp(time_diff / (self.stdp_window / 2))
                    
                    # Apply weight change
                    if self.synaptic_weights[pre, post] != 0:  # Only update existing connections
                        new_weight = max(0, min(1.0, self.synaptic_weights[pre, post] + weight_change))
                        self.synaptic_weights[pre, post] = new_weight
    
    def _update_region_activations(self, spikes):
        """
        Update region activations based on spike activity.
        
        Args:
            spikes: List of spike events per timestep
        """
        # Count spikes per neuron
        spike_counts = defaultdict(int)
        for step_spikes in spikes:
            for neuron_idx, _ in step_spikes:
                spike_counts[neuron_idx] += 1
        
        # Update region activations
        for region_name, region in self.regions.items():
            region_neurons = region['neurons']
            if not region_neurons:
                continue
                
            # Calculate average activation in region
            total_spikes = sum(spike_counts.get(n, 0) for n in region_neurons)
            if region_neurons:
                activation = min(1.0, total_spikes / (len(region_neurons) * 0.5))
                # Apply smoothing with previous activation
                region['activation'] = 0.3 * region['activation'] + 0.7 * activation
            else:
                # Apply decay if no neurons
                region['activation'] *= 0.8
    
    def encode_system_state(self, system_state, component_states=None):
        """
        Enhanced encoding of system state with temporal dynamics.
        
        Args:
            system_state: Dictionary with system state information
            component_states: Optional component state information
            
        Returns:
            Encoded input pattern
        """
        # Initialize or update history if not already present
        if not hasattr(self, 'encoded_state_history'):
            self.encoded_state_history = deque(maxlen=5)  # Store last 5 states
            
        # Create input pattern of appropriate size
        input_size = len(self.input_layer)
        
        # Allocate portions of the input pattern
        current_state_size = int(input_size * 0.7)  # 70% for current state
        history_size = input_size - current_state_size  # 30% for history
        
        # Create pattern for current state
        current_pattern = np.zeros(current_state_size)
        
        # Encoding sections for current state
        sections = {
            'phi': (0, int(0.05 * current_state_size)),  # Integration level
            'recurrent_loops': (int(0.05 * current_state_size), int(0.1 * current_state_size)),
            'subsystems': (int(0.1 * current_state_size), int(0.3 * current_state_size)),
            'component_states': (int(0.3 * current_state_size), int(0.5 * current_state_size)),
            'contradictions': (int(0.5 * current_state_size), int(0.55 * current_state_size)),
            'relative_changes': (int(0.55 * current_state_size), current_state_size)  # New section for relative changes
        }
        
        # 1. Encode phi value (integration)
        if system_state and 'phi' in system_state:
            phi = system_state['phi']
            start, end = sections['phi']
            # Use population coding for phi (multiple neurons with bell curve activation)
            phi_range = end - start
            center = start + int(phi * phi_range)
            width = max(1, phi_range // 10)
            for i in range(start, end):
                # Gaussian-like activation centered on phi value
                distance = abs(i - center)
                current_pattern[i] = np.exp(-0.5 * (distance / width) ** 2)
        
        # 2. Encode recurrent loops with population coding
        if system_state and 'recurrent_loops' in system_state:
            loops = system_state['recurrent_loops']
            start, end = sections['recurrent_loops']
            loops_range = end - start
            # Normalize loops to 0-1 range (assuming max reasonable value is 15)
            norm_loops = min(1.0, loops / 15.0)
            center = start + int(norm_loops * loops_range)
            width = max(1, loops_range // 8)
            for i in range(start, end):
                distance = abs(i - center)
                current_pattern[i] = np.exp(-0.5 * (distance / width) ** 2)
        
        # 3. Encode subsystem activities with dedicated neurons per subsystem
        if system_state and 'subsystem_activities' in system_state:
            activities = system_state['subsystem_activities']
            start, end = sections['subsystems']
            subsystem_range = end - start
            
            # Allocate neurons for each subsystem
            neurons_per_subsystem = subsystem_range // (len(activities) + 1)
            
            # Encode each subsystem activity
            index = start
            for subsystem, activity in activities.items():
                subsystem_start = index
                subsystem_end = index + neurons_per_subsystem
                
                # Population coding for activity level
                center = subsystem_start + int(activity * neurons_per_subsystem)
                width = max(1, neurons_per_subsystem // 4)
                for i in range(subsystem_start, subsystem_end):
                    if i < end:  # Ensure we stay within bounds
                        distance = abs(i - center)
                        current_pattern[i] = np.exp(-0.5 * (distance / width) ** 2)
                
                index += neurons_per_subsystem
        
        # 4. Encode component states with relationship encoding
        if component_states:
            start, end = sections['component_states']
            comp_range = end - start
            
            # First pass: encode absolute activations
            components = list(component_states.keys())
            neurons_per_component = comp_range // (2 * len(components))
            
            index = start
            for component in components:
                if component in component_states and hasattr(component_states[component], 'get'):
                    activation = component_states[component].get('activation', 0)
                    component_start = index
                    component_end = index + neurons_per_component
                    
                    # Population coding for component activation
                    center = component_start + int(activation * neurons_per_component)
                    width = max(1, neurons_per_component // 4)
                    for i in range(component_start, component_end):
                        if i < end:
                            distance = abs(i - center)
                            current_pattern[i] = np.exp(-0.5 * (distance / width) ** 2)
                    
                    index += neurons_per_component
            
            # Second pass: encode relationships between components
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components):
                    if i < j and index < end:  # Only encode each pair once
                        # Calculate correlation between component activations
                        if (comp1 in component_states and comp2 in component_states and
                            hasattr(component_states[comp1], 'get') and 
                            hasattr(component_states[comp2], 'get')):
                            
                            act1 = component_states[comp1].get('activation', 0)
                            act2 = component_states[comp2].get('activation', 0)
                            
                            # Simple measure of relationship (difference)
                            relationship = 1.0 - abs(act1 - act2)
                            
                            # Encode relationship (higher for components with similar activity)
                            current_pattern[index] = relationship
                            index += 1
        
        # 5. Encode presence of contradictions with severity
        start, end = sections['contradictions']
        if system_state and 'contradictions' in system_state:
            contradictions = system_state['contradictions']
            if isinstance(contradictions, bool):
                # Simple presence/absence
                current_pattern[start:end] = 0.8 if contradictions else 0.0
            elif isinstance(contradictions, (list, tuple)):
                # Encode number and severity of contradictions
                count = len(contradictions)
                severity = sum(c.get('severity', 0.5) for c in contradictions) / max(1, count)
                
                # Spread encoding across neurons
                current_pattern[start] = min(1.0, count / 5.0)  # Normalize count (assume max 5)
                if end > start + 1:
                    current_pattern[start+1] = severity
        
        # 6. Encode relative changes from previous state
        start, end = sections['relative_changes']
        if self.encoded_state_history and 'phi' in system_state:
            # Get previous phi value
            prev_state = self.encoded_state_history[0]
            if len(prev_state) >= sections['phi'][1]:
                # Extract previous phi from encoded pattern (approximate)
                prev_phi_section = prev_state[sections['phi'][0]:sections['phi'][1]]
                if np.any(prev_phi_section):
                    prev_phi_idx = np.argmax(prev_phi_section)
                    prev_phi = prev_phi_idx / (sections['phi'][1] - sections['phi'][0])
                    
                    # Calculate change
                    current_phi = system_state['phi']
                    phi_change = current_phi - prev_phi
                    
                    # Encode change direction and magnitude
                    change_point = start + (end - start) // 2
                    if phi_change > 0:
                        # Increasing phi (activation to the right of center)
                        magnitude = min(1.0, abs(phi_change) * 5)  # Scale for visibility
                        for i in range(change_point, min(change_point + int(magnitude * (end - change_point)), end)):
                            current_pattern[i] = 0.8
                    else:
                        # Decreasing phi (activation to the left of center)
                        magnitude = min(1.0, abs(phi_change) * 5)
                        for i in range(max(start, change_point - int(magnitude * (change_point - start))), change_point):
                            current_pattern[i] = 0.8
        
        # Now encode historical information
        history_pattern = np.zeros(history_size)
        if self.encoded_state_history:
            # Encode temporal sequence of previous states
            history_per_state = history_size // len(self.encoded_state_history)
            
            for i, hist_state in enumerate(self.encoded_state_history):
                # Calculate temporal decay factor (older states have less influence)
                decay = 1.0 - (i / len(self.encoded_state_history))
                
                # Extract key features from historical state (compression)
                if len(hist_state) > 0:
                    # Compress history by averaging sections
                    compressed = []
                    section_size = len(hist_state) // 5  # Divide into 5 sections
                    for j in range(5):
                        section_start = j * section_size
                        section_end = (j+1) * section_size if j < 4 else len(hist_state)
                        section_avg = np.mean(hist_state[section_start:section_end])
                        compressed.append(section_avg)
                    
                    # Place compressed history in appropriate section
                    start_idx = i * history_per_state
                    for j, value in enumerate(compressed):
                        if j < history_per_state:
                            history_pattern[start_idx + j] = value * decay
        
        # Combine current state and history
        full_pattern = np.zeros(input_size)
        full_pattern[:current_state_size] = current_pattern
        full_pattern[current_state_size:] = history_pattern
        
        # Save current state to history
        self.encoded_state_history.appendleft(current_pattern)
        
        return full_pattern
    
    def decode_metacognitive_output(self, output_pattern, spike_patterns=None):
        """
        Decode metacognitive output based on spike patterns rather than
        just final membrane potentials, capturing temporal dynamics.
        
        Args:
            output_pattern: Neural output pattern (final membrane potentials)
            spike_patterns: Spike patterns over time from simulation
            
        Returns:
            Dictionary with decoded metacognitive assessments
        """
        # Initialize decoded outputs
        decoded = {
            'confidence': 0.5,  # Default mid-level confidence
            'issues': [],
            'attention_focus': None,
            'strategy_adjustments': {}
        }
        
        # If no spike patterns provided, fall back to using output_pattern
        if not spike_patterns:
            # Use original decoding logic based on membrane potentials
            confidence_neurons = self.decoders['confidence']['neurons']
            confidence_activations = [
                output_pattern[self.output_layer.index(n)] if n in self.output_layer else 0.0
                for n in confidence_neurons
            ]
            
            if confidence_activations:
                # Use mean activation of confidence neurons
                mean_activation = sum(confidence_activations) / len(confidence_activations)
                
                # Map to confidence level using ranges
                conf_level = 0.5  # Default
                for (lower, upper), value in self.decoders['confidence']['ranges'].items():
                    if lower <= mean_activation < upper:
                        conf_level = value
                        break
                
                decoded['confidence'] = conf_level
        else:
            # Use spike patterns for more sophisticated decoding
            
            # 1. Calculate spike rates for output neurons
            output_spike_counts = defaultdict(int)
            total_timesteps = len(spike_patterns)
            
            for step_spikes in spike_patterns:
                for neuron_idx, _ in step_spikes:
                    if neuron_idx in self.output_layer:
                        output_spike_counts[neuron_idx] += 1
            
            # 2. Analyze temporal patterns of spikes
            # Identify sequential activation patterns
            sequential_patterns = self._identify_sequential_patterns(spike_patterns)
            
            # 3. Calculate coherence of output activation
            coherence = self._calculate_output_coherence(spike_patterns)
            
            # 4. Decode confidence using spike rates
            confidence_neurons = self.decoders['confidence']['neurons']
            if confidence_neurons:
                # Calculate spike rates for confidence neurons
                confidence_rates = [output_spike_counts.get(n, 0) / max(1, total_timesteps) 
                                for n in confidence_neurons]
                
                # Weight by temporal position (later spikes more significant)
                temporal_weights = []
                for n in confidence_neurons:
                    # Find when this neuron spiked
                    spike_times = []
                    for t, step_spikes in enumerate(spike_patterns):
                        if any(idx == n for idx, _ in step_spikes):
                            spike_times.append(t)
                    
                    # Calculate temporal weight (more weight to later spikes)
                    if spike_times:
                        avg_time = sum(spike_times) / len(spike_times)
                        # Normalize to 0-1 range
                        temporal_weight = avg_time / max(1, total_timesteps - 1)
                    else:
                        temporal_weight = 0.5  # Default for no spikes
                    
                    temporal_weights.append(temporal_weight)
                
                # Combine spike rates and temporal weights
                if confidence_rates:
                    # Higher rate and later timing = higher confidence
                    confidence_scores = [rate * (0.5 + 0.5 * weight) 
                                        for rate, weight in zip(confidence_rates, temporal_weights)]
                    
                    # Normalize to get final confidence
                    max_possible_rate = 1.0  # Maximum expected rate
                    confidence = min(1.0, sum(confidence_scores) / 
                                (len(confidence_scores) * max_possible_rate))
                    
                    # Apply coherence factor (more coherent = more confident)
                    confidence = confidence * (0.7 + 0.3 * coherence)
                    
                    decoded['confidence'] = confidence
            
            # 5. Decode detected issues using spike patterns
            issue_neurons = self.decoders['issues']['neurons']
            issue_threshold = self.decoders['issues']['thresholds']['detection']
            
            for i, neuron_idx in enumerate(issue_neurons):
                # Calculate spike rate for this neuron
                spike_rate = output_spike_counts.get(neuron_idx, 0) / max(1, total_timesteps)
                
                # Check for specific temporal pattern of activation
                temporal_signature = False
                for pattern in sequential_patterns:
                    if neuron_idx in pattern:
                        temporal_signature = True
                        break
                
                # Combine rate and temporal information
                activation_strength = spike_rate
                if temporal_signature:
                    activation_strength *= 1.5  # Boost if part of temporal pattern
                
                # If activation exceeds threshold, issue is detected
                if activation_strength >= issue_threshold:
                    # Map to issue type
                    issue_idx = i % len(self.decoders['issues']['issue_types'])
                    issue_type = self.decoders['issues']['issue_types'].get(issue_idx, 'unknown_issue')
                    
                    # Calculate severity based on activation level
                    severity = (activation_strength - issue_threshold) / (1.0 - issue_threshold)
                    severity = max(0.3, min(0.9, 0.5 + severity))
                    
                    decoded['issues'].append({
                        'type': issue_type,
                        'severity': severity,
                        'description': f"Detected {issue_type.replace('_', ' ')}",
                        'temporal_signature': temporal_signature
                    })
            
            # 6. Decode attention focus
            attention_neurons = self.decoders['attention']['neurons']
            attention_threshold = self.decoders['attention']['thresholds']['focus_shift']
            
            # Find neuron with highest sustained activation
            max_sustained_activation = 0.0
            max_idx = -1
            
            for i, neuron_idx in enumerate(attention_neurons):
                # Calculate sustained activation (not just spike count but consistency)
                spike_times = []
                for t, step_spikes in enumerate(spike_patterns):
                    if any(idx == neuron_idx for idx, _ in step_spikes):
                        spike_times.append(t)
                
                # Calculate activation sustainability
                if spike_times:
                    # More spikes and more consistent timing = higher sustainability
                    spike_count = len(spike_times)
                    time_variance = np.var(spike_times) if len(spike_times) > 1 else total_timesteps
                    
                    # Normalize variance (lower is better)
                    norm_variance = max(0.1, 1.0 - (time_variance / total_timesteps))
                    
                    # Combine count and consistency
                    sustained_activation = (spike_count / total_timesteps) * norm_variance
                    
                    if sustained_activation > max_sustained_activation:
                        max_sustained_activation = sustained_activation
                        max_idx = i
            
            # Set attention focus if sustained activation is above threshold
            if max_sustained_activation >= attention_threshold:
                # Determine focus target
                if decoded['issues']:
                    # Focus on most severe issue
                    decoded['attention_focus'] = decoded['issues'][0]['type']
                else:
                    # Focus on appropriate subsystem based on neuron index
                    subsystems = ['neural', 'symbolic', 'vector', 'reasoning', 'metacognitive']
                    if max_idx < len(subsystems):
                        decoded['attention_focus'] = subsystems[max_idx]
                        
                        # Additional focus parameters based on activation pattern
                        decoded['attention_parameters'] = {
                            'intensity': max_sustained_activation,
                            'duration': 0.5 + 0.5 * max_sustained_activation  # Higher activation = longer focus
                        }
            
            # 7. Decode strategy adjustments from spike patterns
            strategy_neurons = self.decoders['strategy']['neurons']
            strategy_threshold = self.decoders['strategy']['thresholds']['adjustment']
            
            # Look for co-activation patterns in strategy neurons
            coactivation_groups = self._identify_coactivation_groups(spike_patterns, strategy_neurons)
            
            for i, neuron_idx in enumerate(strategy_neurons):
                # Calculate spike rate
                spike_rate = output_spike_counts.get(neuron_idx, 0) / max(1, total_timesteps)
                
                # Check if part of co-activation group
                in_coactivation = False
                for group in coactivation_groups:
                    if neuron_idx in group:
                        in_coactivation = True
                        break
                
                # Neuron activation is more significant if part of co-activation group
                activation_strength = spike_rate
                if in_coactivation:
                    activation_strength *= 1.3  # Boost if part of co-activation
                
                # If activation exceeds threshold, strategy adjustment is suggested
                if activation_strength >= strategy_threshold:
                    # Map to strategy type
                    strategy_idx = i % len(self.decoders['strategy']['strategies'])
                    strategy_type = self.decoders['strategy']['strategies'].get(strategy_idx)
                    
                    if strategy_type:
                        decoded['strategy_adjustments'][strategy_type] = True
                        
                        # Add parameter details for strategy
                        decoded['strategy_parameters'] = decoded.get('strategy_parameters', {})
                        decoded['strategy_parameters'][strategy_type] = {
                            'strength': activation_strength,
                            'confidence': activation_strength * coherence
                        }
        
        return decoded

    def _identify_sequential_patterns(self, spike_patterns):
        """
        Identify sequential activation patterns in spike data.
        
        Args:
            spike_patterns: Spike patterns over time
            
        Returns:
            List of detected sequential patterns
        """
        sequential_patterns = []
        
        # Track neuron activation order
        neuron_activation_times = defaultdict(list)
        
        # Record first spike time for each neuron
        for t, step_spikes in enumerate(spike_patterns):
            for neuron_idx, _ in step_spikes:
                neuron_activation_times[neuron_idx].append(t)
        
        # Filter to output neurons only
        output_activation_times = {n: times for n, times in neuron_activation_times.items() 
                                if n in self.output_layer}
        
        # Find neurons that consistently activate in sequence
        for n1 in output_activation_times:
            for n2 in output_activation_times:
                if n1 != n2:
                    # Check if n1 consistently activates before n2
                    consistent_sequence = True
                    n1_times = output_activation_times[n1]
                    n2_times = output_activation_times[n2]
                    
                    # Need minimum number of activations to establish pattern
                    if len(n1_times) >= 2 and len(n2_times) >= 2:
                        # Check timing relationships
                        for t1 in n1_times:
                            # Find closest n2 activation after t1
                            future_n2 = [t for t in n2_times if t > t1]
                            if not future_n2:
                                consistent_sequence = False
                                break
                        
                        # If consistent sequence found, add to patterns
                        if consistent_sequence:
                            sequential_patterns.append((n1, n2))
        
        return sequential_patterns

    def _calculate_output_coherence(self, spike_patterns):
        """
        Calculate coherence of output activation patterns.
        Higher coherence indicates more organized, less random firing.
        
        Args:
            spike_patterns: Spike patterns over time
            
        Returns:
            Coherence score (0-1)
        """
        # Extract output layer spikes
        output_spikes = []
        for t, step_spikes in enumerate(spike_patterns):
            step_output_spikes = [n for n, _ in step_spikes if n in self.output_layer]
            output_spikes.append(step_output_spikes)
        
        # Calculate various coherence metrics
        
        # 1. Temporal consistency (less random timing)
        spike_counts_per_step = [len(spikes) for spikes in output_spikes]
        if not spike_counts_per_step:
            return 0.5  # Default medium coherence
        
        temporal_variance = np.var(spike_counts_per_step) / max(1, np.mean(spike_counts_per_step))
        temporal_consistency = 1.0 / (1.0 + temporal_variance)  # Lower variance = higher consistency
        
        # 2. Spatial consistency (same groups of neurons fire together)
        neuron_co_activations = defaultdict(int)
        total_timesteps = len(output_spikes)
        
        # Count co-activations
        for step_spikes in output_spikes:
            # For each pair of neurons that spike together
            for i, n1 in enumerate(step_spikes):
                for n2 in step_spikes[i+1:]:
                    pair = tuple(sorted([n1, n2]))
                    neuron_co_activations[pair] += 1
        
        # Calculate co-activation consistency
        if neuron_co_activations:
            coactivation_values = list(neuron_co_activations.values())
            # Higher mean and lower variance = more consistent co-activation
            coactivation_mean = np.mean(coactivation_values) / total_timesteps
            coactivation_variance = np.var(coactivation_values) / max(1, np.mean(coactivation_values))
            spatial_consistency = coactivation_mean / (1.0 + coactivation_variance)
        else:
            spatial_consistency = 0.5  # Default
        
        # Combine metrics for overall coherence
        coherence = 0.5 * temporal_consistency + 0.5 * spatial_consistency
        
        return min(1.0, coherence)

    def _identify_coactivation_groups(self, spike_patterns, target_neurons):
        """
        Identify groups of neurons that consistently activate together.
        
        Args:
            spike_patterns: Spike patterns over time
            target_neurons: List of neurons to analyze for coactivation
            
        Returns:
            List of coactivation groups
        """
        coactivation_groups = []
        
        # Count how often each pair of neurons fire together
        coactivation_counts = defaultdict(int)
        
        for step_spikes in spike_patterns:
            # Get neurons from target list that spiked this step
            spiking_targets = [n for n, _ in step_spikes if n in target_neurons]
            
            # Count coactivations for each pair
            for i, n1 in enumerate(spiking_targets):
                for n2 in spiking_targets[i+1:]:
                    pair = tuple(sorted([n1, n2]))
                    coactivation_counts[pair] += 1
        
        # Build coactivation graph
        coactivation_graph = defaultdict(set)
        total_steps = len(spike_patterns)
        
        # Add edges for pairs that coactivate frequently
        for (n1, n2), count in coactivation_counts.items():
            if count / total_steps > 0.3:  # Coactivate in at least 30% of timesteps
                coactivation_graph[n1].add(n2)
                coactivation_graph[n2].add(n1)
        
        # Find connected components (coactivation groups)
        visited = set()
        
        for neuron in target_neurons:
            if neuron not in visited and neuron in coactivation_graph:
                # Start a new group
                group = set()
                queue = [neuron]
                
                # Breadth-first search to find connected neurons
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        
                        # Add neighbors
                        for neighbor in coactivation_graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                # Add group if it has at least 2 neurons
                if len(group) >= 2:
                    coactivation_groups.append(group)
        
        return coactivation_groups
    
    def monitor_system_state(self, 
                          system_state: Dict, 
                          component_states: Dict = None, 
                          recent_processing: Dict = None):
        """
        Monitor the current state of the system and components.
        
        Args:
            system_state: Overall system state information (phi, integration, etc.)
            component_states: States of individual components
            recent_processing: Information about recent processing steps
            
        Returns:
            Metacognitive analysis of the system state
        """
        # Update component state history
        if component_states:
            timestamp = time.time()
            for component, state in component_states.items():
                if component not in self.component_state_history:
                    self.component_state_history[component] = deque(maxlen=20)
                self.component_state_history[component].append({
                    'timestamp': timestamp,
                    'state': state
                })
        
        # Extract key information from system state
        phi_value = system_state.get('phi', 0.1)
        self.current_phi = phi_value
        self.phi_history.append(phi_value)
        
        # Encode system state as neural input pattern
        monitoring_pattern = self.encode_system_state(system_state, component_states)
        
        # Process the monitoring pattern through the SNN
        metacognitive_results = self.process_input(monitoring_pattern, timesteps=10)
        
        # Decode the metacognitive outputs from the resulting neural activity pattern
        metacognitive_assessments = self.decode_metacognitive_output(
            metacognitive_results['output_pattern'], 
            metacognitive_results['spikes']
        )
        
        # Update metacognitive state based on decoded neural output
        self._update_metacognitive_state_from_neural_output(metacognitive_assessments)
        
        # Generate metacognitive signals from state
        signals = self._generate_metacognitive_signals_from_state()
        
        # Check for contradictions in recent processing
        if recent_processing and 'contradictions' in recent_processing:
            contradictions = recent_processing['contradictions']
            if contradictions:
                self.metacognitive_state['detected_contradictions'] = contradictions
        
        return {
            'metacognitive_state': self.metacognitive_state,
            'phi_value': phi_value,
            'phi_trend': self._calculate_phi_trend(),
            'detected_issues': metacognitive_assessments['issues'],
            'confidence_estimate': metacognitive_assessments['confidence'],
            'metacognitive_signals': signals,
            'neural_processing': {
                'pattern': monitoring_pattern,
                'spikes': len(metacognitive_results['spikes']),
                'region_activations': metacognitive_results['region_activations']
            }
        }
    
    def _update_metacognitive_state_from_neural_output(self, metacognitive_assessments):
        """
        Update metacognitive state based on decoded neural output.
        
        Args:
            metacognitive_assessments: Decoded metacognitive assessments
        """
        # Update confidence
        self.metacognitive_state['system_confidence'] = metacognitive_assessments['confidence']
        
        # Update detected issues
        if metacognitive_assessments['issues']:
            # Store detected issues that weren't previously detected
            current_issues = {issue['type'] for issue in self.metacognitive_state.get('detected_issues', [])}
            new_issues = [
                issue for issue in metacognitive_assessments['issues']
                if issue['type'] not in current_issues
            ]
            
            # Update with new issues
            self.metacognitive_state['detected_issues'] = \
                self.metacognitive_state.get('detected_issues', []) + new_issues
        
        # Update attention focus
        if metacognitive_assessments['attention_focus']:
            self.metacognitive_state['attention_focus'] = metacognitive_assessments['attention_focus']
        
        # Update strategy adjustments
        self.metacognitive_state['strategy_adjustments'] = metacognitive_assessments['strategy_adjustments']
    
    def _generate_metacognitive_signals_from_state(self):
        """
        Generate metacognitive signals based on updated state.
        
        Returns:
            Dictionary with metacognitive signals
        """
        signals = {}
        
        # Generate strategy signals
        if self.metacognitive_state.get('strategy_adjustments'):
            signals['strategy_adjustments'] = self.metacognitive_state['strategy_adjustments']
        
        # Generate attention signals
        if self.metacognitive_state.get('attention_focus'):
            attention_signals = {
                'focus_on': self.metacognitive_state['attention_focus']
            }
            
            # If contradictions detected, focus attention on resolving them
            if self.metacognitive_state.get('detected_contradictions'):
                attention_signals['resolve_contradictions'] = True
                
            signals['attention_allocation'] = attention_signals
        
        # Generate learning signals
        learning_signals = {}
        
        # Learning from issues and experiences
        if self.metacognitive_state.get('detected_contradictions'):
            learning_signals['learn_from_contradictions'] = True
        
        if len(self.reasoning_episodes) > 10:
            # Enough episodes to extract patterns
            learning_signals['extract_reasoning_patterns'] = True
        
        if learning_signals:
            signals['learning_adjustments'] = learning_signals
        
        return signals
    
    def evaluate_reasoning(self, reasoning_trace, expected_outcome=None):
        """
        Evaluate the quality of a reasoning process and suggest improvements.
        
        Args:
            reasoning_trace: Trace of the reasoning steps
            expected_outcome: Optional expected outcome for comparison
            
        Returns:
            Evaluation of reasoning quality and improvement suggestions
        """
        # Extract key features from reasoning trace
        trace_features = self._extract_reasoning_trace_features(reasoning_trace)
        
        # Encode reasoning trace as neural input pattern
        eval_pattern = self._encode_reasoning_trace(trace_features)
        
        # Process the encoded pattern through the SNN
        reasoning_results = self.process_input(eval_pattern, timesteps=10)
        
        # Decode the evaluation results
        reasoning_assessment = self._decode_reasoning_evaluation(
            reasoning_results['output_pattern'],
            reasoning_results['spikes']
        )
        
        # Update metacognitive state
        self.metacognitive_state['reasoning_quality'] = reasoning_assessment['quality_score']
        
        # Store this reasoning episode for learning
        self._store_reasoning_episode(reasoning_trace, trace_features, 
                                    reasoning_assessment['quality_score'], 
                                    reasoning_assessment['issues'])
    
    def _encode_reasoning_trace(self, trace_features):
        """
        Encode reasoning trace features into neural activation pattern.
        
        Args:
            trace_features: Dictionary of reasoning trace features
            
        Returns:
            Encoded neural pattern for input layer
        """
        # Create input pattern of appropriate size
        input_size = len(self.input_layer)
        pattern = np.zeros(input_size)
        
        # Encoding sections
        sections = {
            'depth': (0, 2),                 # Reasoning depth
            'coherence': (2, 4),             # Reasoning coherence
            'logical_consistency': (4, 6),   # Logical consistency
            'recurrence': (6, 8),            # Recurrence/cyclic patterns
            'steps_count': (8, 10),          # Number of reasoning steps
            'step_types': (10, input_size)   # Types of reasoning steps
        }
        
        # 1. Encode reasoning depth
        if 'depth' in trace_features:
            start, end = sections['depth']
            depth = trace_features['depth']
            pattern[start:end] = depth
        
        # 2. Encode coherence
        if 'coherence' in trace_features:
            start, end = sections['coherence']
            coherence = trace_features['coherence']
            pattern[start:end] = coherence
        
        # 3. Encode logical consistency
        if 'logical_consistency' in trace_features:
            start, end = sections['logical_consistency']
            consistency = trace_features['logical_consistency']
            pattern[start:end] = consistency
        
        # 4. Encode recurrence
        if 'recurrence' in trace_features:
            start, end = sections['recurrence']
            recurrence = trace_features['recurrence']
            pattern[start:end] = recurrence
        
        # 5. Encode step count
        if 'steps_count' in trace_features:
            start, end = sections['steps_count']
            steps = trace_features['steps_count']
            # Normalize to 0-1 range (assuming 20 steps is maximum)
            normalized_steps = min(1.0, steps / 20.0)
            pattern[start:end] = normalized_steps
        
        # 6. Encode step type distribution
        if 'step_types' in trace_features and trace_features['step_types']:
            start, end = sections['step_types']
            step_types = trace_features['step_types']
            total_steps = sum(step_types.values())
            
            if total_steps > 0:
                # Sort step types by frequency
                sorted_types = sorted(step_types.items(), key=lambda x: x[1], reverse=True)
                
                # Encode top step types
                max_types = min(len(sorted_types), end - start)
                for i in range(max_types):
                    _, count = sorted_types[i]
                    idx = start + i
                    pattern[idx] = count / total_steps
        
        return pattern
    
    def _decode_reasoning_evaluation(self, output_pattern, spike_patterns):
        """
        Decode neural output pattern into reasoning evaluation.
        
        Args:
            output_pattern: Neural output pattern
            spike_patterns: Spike patterns from simulation
            
        Returns:
            Dictionary with reasoning evaluation
        """
        # Extract spikes from reasoning evaluation region
        eval_region_neurons = self.regions['reasoning_evaluation']['neurons']
        eval_spikes = 0
        
        for step_spikes in spike_patterns:
            for neuron_idx, _ in step_spikes:
                if neuron_idx in eval_region_neurons:
                    eval_spikes += 1
        
        # Calculate quality score based on output pattern
        # Higher activations in output layer indicate better quality
        output_activations = [
            output_pattern[i] for i in range(len(output_pattern))
            if self.output_layer[i] in eval_region_neurons
        ]
        
        quality_score = 0.5  # Default middle score
        if output_activations:
            # Use weighted average of output activations
            quality_score = sum(output_activations) / len(output_activations)
        
        # Detect reasoning issues
        issue_neurons = self.regions['reasoning_evaluation']['neurons'][-15:]  # Last 15 neurons for issues
        issues = []
        
        # Map output neurons to issue types
        issue_types = {
            0: 'shallow_reasoning',
            1: 'low_coherence',
            2: 'logical_inconsistency',
            3: 'circular_reasoning',
            4: 'insufficient_steps',
            5: 'logic_jump'
        }
        
        for i, neuron_idx in enumerate(issue_neurons):
            if neuron_idx in self.output_layer:
                # Get index in output pattern
                output_idx = self.output_layer.index(neuron_idx)
                if output_idx < len(output_pattern):
                    activation = output_pattern[output_idx]
                    
                    # If activation exceeds threshold, issue is detected
                    if activation >= 0.6:  # Threshold for issue detection
                        # Map to issue type
                        issue_idx = i % len(issue_types)
                        issue_type = issue_types.get(issue_idx, 'unknown_issue')
                        
                        # Calculate severity based on activation level
                        severity = (activation - 0.6) / 0.4  # Normalize to 0-1 range
                        severity = max(0.3, min(0.9, 0.5 + severity))
                        
                        issues.append({
                            'type': issue_type,
                            'severity': severity,
                            'description': self._get_issue_description(issue_type)
                        })
        
        # Generate improvement suggestions based on detected issues
        suggestions = self._generate_suggestions_from_issues(issues, quality_score)
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'improvement_suggestions': suggestions,
            'metacognitive_evaluation': {
                'coherence': quality_score * 0.8 + 0.1,  # Adjusted based on quality
                'depth': quality_score * 0.7 + 0.15,
                'logical_consistency': quality_score * 0.9 + 0.05
            }
        }
    
    def _get_issue_description(self, issue_type):
        """Get description for an issue type."""
        descriptions = {
            'shallow_reasoning': 'Reasoning lacks sufficient depth',
            'low_coherence': 'Reasoning steps lack proper connections',
            'logical_inconsistency': 'Reasoning contains potential contradictions',
            'circular_reasoning': 'Reasoning appears to cycle through the same operations',
            'insufficient_steps': 'Reasoning process has very few steps',
            'logic_jump': 'Steps have missing logical connections'
        }
        return descriptions.get(issue_type, f"Issue with {issue_type.replace('_', ' ')}")
    
    def _generate_suggestions_from_issues(self, issues, quality_score):
        """Generate improvement suggestions based on detected issues."""
        suggestions = []
        
        # Generate specific suggestions based on issue types
        for issue in issues:
            issue_type = issue.get('type', '')
            
            if issue_type == 'shallow_reasoning':
                suggestions.append({
                    'target': 'depth',
                    'action': 'increase',
                    'description': 'Consider exploring more implications of each step',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'low_coherence':
                suggestions.append({
                    'target': 'coherence',
                    'action': 'improve',
                    'description': 'Explicitly connect each step to previous reasoning',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'logical_inconsistency':
                suggestions.append({
                    'target': 'consistency',
                    'action': 'verify',
                    'description': 'Review reasoning for potential contradictions',
                    'priority': 0.9  # Always high priority
                })
                
            elif issue_type == 'circular_reasoning':
                suggestions.append({
                    'target': 'approach',
                    'action': 'change',
                    'description': 'Try alternative reasoning strategies to avoid loops',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'insufficient_steps':
                suggestions.append({
                    'target': 'elaboration',
                    'action': 'increase',
                    'description': 'Break reasoning into smaller, more detailed steps',
                    'priority': issue.get('severity', 0.5)
                })
        
        # General quality-based suggestions
        if quality_score < 0.4:
            suggestions.append({
                'target': 'overall',
                'action': 'redesign',
                'description': 'Consider a completely different approach to the problem',
                'priority': 0.8
            })
        elif quality_score < 0.7:
            suggestions.append({
                'target': 'recurrent_processing',
                'action': 'increase',
                'description': 'Increase recurrent processing loops for deeper analysis',
                'priority': 0.6
            })
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return suggestions
    
    def _extract_reasoning_trace_features(self, reasoning_trace):
        """Extract key features from a reasoning trace."""
        features = {
            'steps_count': 0,
            'depth': 0.5,  # Default middle value
            'coherence': 0.5,
            'logical_consistency': 0.5,
            'recurrence': 0.0,
            'step_types': defaultdict(int)
        }
        
        if not reasoning_trace:
            return features
            
        # Count reasoning steps
        if isinstance(reasoning_trace, list):
            features['steps_count'] = len(reasoning_trace)
            
            # Track types of reasoning steps
            for step in reasoning_trace:
                step_type = step.get('type', 'unknown')
                features['step_types'][step_type] += 1
                
            # Calculate recurrence (repeated operations)
            if features['steps_count'] > 1:
                operations = [step.get('operation') for step in reasoning_trace 
                             if 'operation' in step]
                if operations:
                    # Count unique operations
                    unique_ops = len(set(operations))
                    # Recurrence = repeated operations / total operations
                    features['recurrence'] = 1.0 - (unique_ops / len(operations))
            
            # Calculate depth - based on max depth of any reasoning chain
            max_depth = 0
            for step in reasoning_trace:
                depth = step.get('depth', 0)
                max_depth = max(max_depth, depth)
            # Normalize depth to 0-1 range (assuming max reasonable depth is 10)
            features['depth'] = min(1.0, max_depth / 10.0)
            
            # Calculate coherence - based on step connections
            if features['steps_count'] > 1:
                connected_steps = 0
                for i, step in enumerate(reasoning_trace[1:], 1):
                    # Check if step references previous steps
                    if 'depends_on' in step and step['depends_on']:
                        connected_steps += 1
                # Coherence = connected steps / total possible connections
                features['coherence'] = connected_steps / (features['steps_count'] - 1)
            
            # Calculate logical consistency - check for contradictions
            contradictions = 0
            conclusions = set()
            for step in reasoning_trace:
                if 'conclusion' in step:
                    conclusion = step['conclusion']
                    # Check if negation of conclusion already exists
                    neg_conclusion = f"not_{conclusion}" if not conclusion.startswith("not_") else conclusion[4:]
                    if neg_conclusion in conclusions:
                        contradictions += 1
                    conclusions.add(conclusion)
            # Logical consistency decreases with contradictions
            features['logical_consistency'] = max(0.0, 1.0 - (contradictions * 0.2))
        
        return features
    
    def train_from_heuristic_examples(self, num_examples=100):
        """
        Train the metacognitive SNN using heuristic-generated examples.
        This bridges the gap between heuristic logic and neural computation.
        
        Args:
            num_examples: Number of training examples to generate
            
        Returns:
            Training statistics
        """
        # Generate synthetic training data
        training_data = self._generate_training_examples(num_examples)
        
        # Track training performance
        training_stats = {
            'epochs': 10,
            'error_history': [],
            'final_error': 0.0
        }
        
        # Train the network for multiple epochs
        for epoch in range(training_stats['epochs']):
            print(f"[MetaSNN] Training epoch {epoch+1}/{training_stats['epochs']}")
            epoch_error = 0.0
            
            # Shuffle training data
            random.shuffle(training_data)
            
            # Process each training example
            for input_pattern, target_output in training_data:
                # Process the input pattern
                result = self.process_input(input_pattern, timesteps=8)
                output_pattern = result['output_pattern']
                
                # Calculate error
                error = np.mean((output_pattern - target_output) ** 2)
                epoch_error += error
                
                # Apply learning - make adjustments to synaptic weights
                self._train_on_example(input_pattern, target_output, result['spikes'])
            
            # Record average error for this epoch
            avg_error = epoch_error / len(training_data)
            training_stats['error_history'].append(avg_error)
            print(f"[MetaSNN] Epoch {epoch+1} average error: {avg_error:.4f}")
        
        # Record final error
        training_stats['final_error'] = training_stats['error_history'][-1]
        
        # Store a subset of training data for future reference
        self.training_data = random.sample(training_data, min(20, len(training_data)))
        
        return training_stats
    
    def _generate_training_examples(self, num_examples):
        """
        Generate synthetic training examples using heuristic logic.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of (input_pattern, target_output) pairs
        """
        examples = []
        
        # Generate system monitoring examples
        for _ in range(num_examples // 2):
            # Generate random system state
            system_state = self._generate_random_system_state()
            
            # Encode system state as input pattern
            input_pattern = self.encode_system_state(system_state)
            
            # Generate target output using heuristic logic
            issues = self._detect_system_issues_heuristic(system_state)
            confidence = self._estimate_system_confidence_heuristic(system_state, issues)
            strategy_signals = self._generate_strategy_signals_heuristic(issues, confidence, system_state)
            
            # Encode target output
            target_output = self._encode_target_output(confidence, issues, strategy_signals)
            
            examples.append((input_pattern, target_output))
        
        # Generate reasoning evaluation examples
        for _ in range(num_examples // 2):
            # Generate random reasoning trace
            trace_features = self._generate_random_reasoning_trace()
            
            # Encode reasoning trace as input pattern
            input_pattern = self._encode_reasoning_trace(trace_features)
            
            # Generate target output using heuristic logic
            reasoning_issues = self._detect_reasoning_issues_heuristic(trace_features)
            quality_score = self._calculate_reasoning_quality_heuristic(trace_features, reasoning_issues)
            suggestions = self._generate_reasoning_suggestions_heuristic(reasoning_issues, quality_score)
            
            # Encode target output
            target_output = self._encode_target_output_reasoning(quality_score, reasoning_issues, suggestions)
            
            examples.append((input_pattern, target_output))
        
        return examples
    
    def _generate_random_system_state(self):
        """Generate random system state for training."""
        # Random phi value (integration measure)
        phi = random.uniform(0.1, 0.9)
        
        # Random recurrent loops
        recurrent_loops = random.randint(1, 10)
        
        # Random subsystem activities
        subsystem_activities = {
            'neural': random.uniform(0.1, 0.9),
            'symbolic': random.uniform(0.1, 0.9),
            'vector': random.uniform(0.1, 0.9),
            'reasoning': random.uniform(0.1, 0.9),
            'metacognitive': random.uniform(0.1, 0.9)
        }
        
        # Random presence of contradictions
        contradictions = random.choice([True, False, False])  # Less likely to have contradictions
        
        return {
            'phi': phi,
            'recurrent_loops': recurrent_loops,
            'subsystem_activities': subsystem_activities,
            'contradictions': contradictions
        }
    
    def _generate_random_reasoning_trace(self):
        """Generate random reasoning trace features for training."""
        # Generate random features with realistic correlations
        depth = random.uniform(0.1, 0.9)
        coherence = random.uniform(0.1, 0.9)
        logical_consistency = random.uniform(0.1, 0.9)
        
        # More depth often correlates with more steps
        steps_factor = 0.7 * depth + 0.3 * random.uniform(0, 1)
        steps_count = int(5 + steps_factor * 15)  # 5 to 20 steps
        
        # Recurrence tends to be inversely related to coherence
        recurrence = random.uniform(0, 0.3) + (1 - coherence) * 0.5
        recurrence = min(1.0, recurrence)
        
        # Generate random step types
        step_types = defaultdict(int)
        possible_types = ['deduction', 'induction', 'abduction', 'analogy', 'calculation']
        for _ in range(steps_count):
            step_type = random.choice(possible_types)
            step_types[step_type] += 1
        
        return {
            'steps_count': steps_count,
            'depth': depth,
            'coherence': coherence,
            'logical_consistency': logical_consistency,
            'recurrence': recurrence,
            'step_types': step_types
        }
    
    def _detect_system_issues_heuristic(self, system_state):
        """Detect system issues using heuristic logic (for training)."""
        issues = []
        
        # Check for low integration (phi)
        if system_state.get('phi', 0) < 0.3:
            issues.append({
                'type': 'low_integration',
                'severity': 0.7,
                'description': 'System has low integration level (phi)',
                'phi_value': system_state.get('phi', 0)
            })
        
        # Check for imbalanced subsystem activity
        if system_state and 'subsystem_activities' in system_state:
            activities = system_state['subsystem_activities']
            if activities:
                max_activity = max(activities.values())
                min_activity = min(activities.values())
                if max_activity > 0.7 and min_activity < 0.2:
                    # Large disparity between highest and lowest activity
                    issues.append({
                        'type': 'subsystem_imbalance',
                        'severity': 0.6,
                        'description': 'Large disparity between subsystem activities',
                        'max_activity': max_activity,
                        'min_activity': min_activity
                    })
        
        # Check for contradictions
        if system_state.get('contradictions', False):
            issues.append({
                'type': 'contradictions',
                'severity': 0.8,
                'description': 'Contradictions detected in processing',
                'count': random.randint(1, 3)  # Random count for training
            })
        
        return issues
    
    def _estimate_system_confidence_heuristic(self, system_state, issues):
        """Estimate system confidence using heuristic logic (for training)."""
        # Start with a base confidence of 0.7
        confidence = 0.7
        
        # Adjust based on phi value (integration)
        if 'phi' in system_state:
            phi = system_state['phi']
            # Higher phi -> higher confidence
            confidence += phi * 0.3  # Max +0.3 for phi=1.0
        
        # Reduce confidence based on detected issues
        for issue in issues:
            severity = issue.get('severity', 0.5)
            # More severe issues reduce confidence more
            confidence -= severity * 0.1  # Each issue can reduce by up to 0.1
        
        # Ensure confidence is in [0, 1] range
        confidence = max(0.1, min(1.0, confidence))
        
        return confidence
    
    def _generate_strategy_signals_heuristic(self, issues, confidence, system_state):
        """Generate strategy signals using heuristic logic (for training)."""
        strategy_signals = {}
        
        # Generate strategy adjustments based on issues and confidence
        # Very low confidence - suggest verification or additional processing
        if confidence < 0.3:
            strategy_signals['verify_outputs'] = True
            strategy_signals['increase_symbolic_processing'] = True
        
        # Low integration - suggest ways to increase integration
        if system_state.get('phi', 0) < 0.3:
            strategy_signals['increase_integration'] = True
            strategy_signals['boost_subsystem_interaction'] = True
        
        # Issue-specific strategies
        for issue in issues:
            issue_type = issue.get('type')
            
            if issue_type == 'low_integration':
                strategy_signals['increase_recurrent_loops'] = True
                
            elif issue_type == 'subsystem_imbalance':
                strategy_signals['balance_subsystems'] = True
                
            elif issue_type == 'contradictions':
                strategy_signals['resolve_contradictions'] = True
                strategy_signals['verify_outputs'] = True
        
        return strategy_signals
    
    def _detect_reasoning_issues_heuristic(self, trace_features):
        """Detect reasoning issues using heuristic logic (for training)."""
        issues = []
        
        # Issue: Shallow reasoning
        if trace_features.get('depth', 0) < 0.3:
            issues.append({
                'type': 'shallow_reasoning',
                'severity': 0.7,
                'description': 'Reasoning lacks sufficient depth',
                'depth_value': trace_features.get('depth', 0)
            })
        
        # Issue: Poor coherence
        if trace_features.get('coherence', 1.0) < 0.4:
            issues.append({
                'type': 'low_coherence',
                'severity': 0.6,
                'description': 'Reasoning steps lack proper connections',
                'coherence_value': trace_features.get('coherence', 0)
            })
        
        # Issue: Logical inconsistencies
        if trace_features.get('logical_consistency', 1.0) < 0.7:
            issues.append({
                'type': 'logical_inconsistency',
                'severity': 0.8,
                'description': 'Reasoning contains potential contradictions',
                'consistency_value': trace_features.get('logical_consistency', 0)
            })
        
        # Issue: Circular reasoning (high recurrence)
        if trace_features.get('recurrence', 0.0) > 0.6:
            issues.append({
                'type': 'circular_reasoning',
                'severity': 0.7,
                'description': 'Reasoning appears to cycle through the same operations',
                'recurrence_value': trace_features.get('recurrence', 0)
            })
        
        # Issue: Very few steps
        if trace_features.get('steps_count', 0) < 3:
            issues.append({
                'type': 'insufficient_steps',
                'severity': 0.5,
                'description': 'Reasoning process has very few steps',
                'steps_count': trace_features.get('steps_count', 0)
            })
        
        return issues
    
    def _calculate_reasoning_quality_heuristic(self, trace_features, reasoning_issues):
        """Calculate reasoning quality using heuristic logic (for training)."""
        # Base quality score
        quality = 0.7  # Start with above average
        
        # Adjust based on trace features
        if 'depth' in trace_features:
            quality += trace_features['depth'] * 0.1  # Depth adds up to 0.1
            
        if 'coherence' in trace_features:
            quality += trace_features['coherence'] * 0.1  # Coherence adds up to 0.1
            
        if 'logical_consistency' in trace_features:
            quality += trace_features['logical_consistency'] * 0.2  # Consistency adds up to 0.2
        
        # Penalize for issues
        for issue in reasoning_issues:
            severity = issue.get('severity', 0.5)
            quality -= severity * 0.15  # Each issue can reduce quality by up to 0.15
        
        # Ensure quality is in [0, 1] range
        quality = max(0.1, min(1.0, quality))
        
        return quality
    
    def _generate_reasoning_suggestions_heuristic(self, reasoning_issues, quality_score):
        """Generate reasoning suggestions using heuristic logic (for training)."""
        suggestions = []
        
        # Generate specific suggestions based on issue types
        for issue in reasoning_issues:
            issue_type = issue.get('type', '')
            
            if issue_type == 'shallow_reasoning':
                suggestions.append({
                    'target': 'depth',
                    'action': 'increase',
                    'description': 'Consider exploring more implications of each step',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'low_coherence':
                suggestions.append({
                    'target': 'coherence',
                    'action': 'improve',
                    'description': 'Explicitly connect each step to previous reasoning',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'logical_inconsistency':
                suggestions.append({
                    'target': 'consistency',
                    'action': 'verify',
                    'description': 'Review reasoning for potential contradictions',
                    'priority': 0.9  # Always high priority
                })
                
            elif issue_type == 'circular_reasoning':
                suggestions.append({
                    'target': 'approach',
                    'action': 'change',
                    'description': 'Try alternative reasoning strategies to avoid loops',
                    'priority': issue.get('severity', 0.5)
                })
                
            elif issue_type == 'insufficient_steps':
                suggestions.append({
                    'target': 'elaboration',
                    'action': 'increase',
                    'description': 'Break reasoning into smaller, more detailed steps',
                    'priority': issue.get('severity', 0.5)
                })
        
        # General quality-based suggestions
        if quality_score < 0.4:
            suggestions.append({
                'target': 'overall',
                'action': 'redesign',
                'description': 'Consider a completely different approach to the problem',
                'priority': 0.8
            })
        elif quality_score < 0.7:
            suggestions.append({
                'target': 'recurrent_processing',
                'action': 'increase',
                'description': 'Increase recurrent processing loops for deeper analysis',
                'priority': 0.6
            })
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return suggestions
    
    def _encode_target_output(self, confidence, issues, strategy_signals):
        """
        Encode target output pattern for system monitoring.
        
        Args:
            confidence: System confidence level
            issues: Detected issues
            strategy_signals: Strategy adjustment signals
            
        Returns:
            Target output pattern
        """
        # Create output pattern of appropriate size
        output_size = len(self.output_layer)
        target_output = np.zeros(output_size)
        
        # Encoding sections
        sections = {
            'confidence': (0, output_size // 5),                  # Confidence level
            'issues': (output_size // 5, 2 * output_size // 5),  # Detected issues
            'attention': (2 * output_size // 5, 3 * output_size // 5),  # Attention focus
            'strategy': (3 * output_size // 5, output_size)       # Strategy adjustments
        }
        
        # 1. Encode confidence level
        start, end = sections['confidence']
        confidence_range = end - start
        # Map confidence to positions in output layer
        confidence_pos = int(start + confidence * confidence_range)
        # Set a small window of neurons around this position
        window_size = max(1, confidence_range // 10)
        for i in range(max(start, confidence_pos - window_size), 
                      min(end, confidence_pos + window_size + 1)):
            # Gaussian-like activation centered on confidence_pos
            dist = abs(i - confidence_pos)
            target_output[i] = np.exp(-0.5 * (dist / (window_size / 2)) ** 2)
        
        # 2. Encode detected issues
        start, end = sections['issues']
        issue_range = end - start
        
        # Map issue types to positions in output layer
        issue_positions = {
            'low_integration': start,
            'subsystem_imbalance': start + issue_range // 6,
            'contradictions': start + 2 * issue_range // 6,
            'processing_loop': start + 3 * issue_range // 6,
            'logical_inconsistency': start + 4 * issue_range // 6,
            'shallow_reasoning': start + 5 * issue_range // 6
        }
        
        # Set activations for detected issues
        for issue in issues:
            issue_type = issue.get('type')
            if issue_type in issue_positions:
                pos = issue_positions[issue_type]
                severity = issue.get('severity', 0.5)
                
                # Set activation based on severity
                for i in range(pos, pos + max(1, issue_range // 36)):
                    if i < end:
                        target_output[i] = severity
        
        # 3. Encode attention focus
        start, end = sections['attention']
        attention_range = end - start
        
        # Determine attention focus based on issues
        attention_focus = None
        if issues:
            # Focus on most severe issue
            most_severe = max(issues, key=lambda x: x.get('severity', 0))
            attention_focus = most_severe.get('type')
        
        # Set activation for attention focus
        if attention_focus:
            attention_pos = start
            if attention_focus == 'low_integration':
                attention_pos = start
            elif attention_focus == 'subsystem_imbalance':
                attention_pos = start + attention_range // 5
            elif attention_focus == 'contradictions':
                attention_pos = start + 2 * attention_range // 5
            elif attention_focus == 'processing_loop':
                attention_pos = start + 3 * attention_range // 5
            elif attention_focus == 'logical_inconsistency':
                attention_pos = start + 4 * attention_range // 5
            
            # Set attention activation
            for i in range(attention_pos, attention_pos + max(1, attention_range // 25)):
                if i < end:
                    target_output[i] = 0.9  # Strong activation for attention focus
        
        # 4. Encode strategy adjustments
        start, end = sections['strategy']
        strategy_range = end - start
        
        # Map strategies to positions in output layer
        strategy_positions = {
            'increase_recurrent_loops': start,
            'change_reasoning_approach': start + strategy_range // 6,
            'verify_outputs': start + 2 * strategy_range // 6,
            'increase_integration': start + 3 * strategy_range // 6,
            'boost_subsystem_interaction': start + 4 * strategy_range // 6,
            'resolve_contradictions': start + 5 * strategy_range // 6
        }
        
        # Set activations for suggested strategies
        for strategy, suggested in strategy_signals.items():
            if suggested and strategy in strategy_positions:
                pos = strategy_positions[strategy]
                
                # Set activation for this strategy
                for i in range(pos, pos + max(1, strategy_range // 36)):
                    if i < end:
                        target_output[i] = 0.9  # Strong activation for suggested strategies
        
        return target_output
    
    def _encode_target_output_reasoning(self, quality_score, issues, suggestions):
        """
        Encode target output pattern for reasoning evaluation.
        
        Args:
            quality_score: Reasoning quality score
            issues: Detected reasoning issues
            suggestions: Improvement suggestions
            
        Returns:
            Target output pattern
        """
        # Create output pattern of appropriate size
        output_size = len(self.output_layer)
        target_output = np.zeros(output_size)
        
        # Encoding sections
        sections = {
            'quality': (0, output_size // 4),                  # Quality score
            'issues': (output_size // 4, 2 * output_size // 4),  # Detected issues
            'suggestions': (2 * output_size // 4, output_size)    # Improvement suggestions
        }
        
        # 1. Encode quality score
        start, end = sections['quality']
        quality_range = end - start
        # Map quality to positions in output layer
        quality_pos = int(start + quality_score * quality_range)
        # Set a small window of neurons around this position
        window_size = max(1, quality_range // 10)
        for i in range(max(start, quality_pos - window_size), 
                      min(end, quality_pos + window_size + 1)):
            # Gaussian-like activation centered on quality_pos
            dist = abs(i - quality_pos)
            target_output[i] = np.exp(-0.5 * (dist / (window_size / 2)) ** 2)
        
        # 2. Encode detected issues
        start, end = sections['issues']
        issue_range = end - start
        
        # Map issue types to positions in output layer
        issue_positions = {
            'shallow_reasoning': start,
            'low_coherence': start + issue_range // 6,
            'logical_inconsistency': start + 2 * issue_range // 6,
            'circular_reasoning': start + 3 * issue_range // 6,
            'insufficient_steps': start + 4 * issue_range // 6,
            'logic_jump': start + 5 * issue_range // 6
        }
        
        # Set activations for detected issues
        for issue in issues:
            issue_type = issue.get('type')
            if issue_type in issue_positions:
                pos = issue_positions[issue_type]
                severity = issue.get('severity', 0.5)
                
                # Set activation based on severity
                for i in range(pos, pos + max(1, issue_range // 36)):
                    if i < end:
                        target_output[i] = severity
        
        # 3. Encode improvement suggestions
        start, end = sections['suggestions']
        suggestion_range = end - start
        
        # Map suggestion targets to positions in output layer
        suggestion_positions = {
            'depth': start,
            'coherence': start + suggestion_range // 7,
            'consistency': start + 2 * suggestion_range // 7,
            'approach': start + 3 * suggestion_range // 7,
            'elaboration': start + 4 * suggestion_range // 7,
            'overall': start + 5 * suggestion_range // 7,
            'recurrent_processing': start + 6 * suggestion_range // 7
        }
        
        # Set activations for suggestions
        for suggestion in suggestions:
            target = suggestion.get('target')
            if target in suggestion_positions:
                pos = suggestion_positions[target]
                priority = suggestion.get('priority', 0.5)
                
                # Set activation based on priority
                for i in range(pos, pos + max(1, suggestion_range // 42)):
                    if i < end:
                        target_output[i] = priority
        
        return target_output
    
    def _train_on_example(self, input_pattern, target_output, spike_patterns):
        """
        Train the network on a single example using parent's plasticity mechanisms
        with enhanced eligibility traces and error signals.
        
        Args:
            input_pattern: Input activation pattern
            target_output: Target output pattern
            spike_patterns: Spike patterns from simulation
        """
        # Calculate error between target and actual output
        output_pattern = self._get_output_pattern()
        error = target_output - output_pattern
        
        # Convert error to compatible format for the learning mechanism
        error_signal = np.zeros(self.neuron_count)
        for i, output_idx in enumerate(self.output_layer):
            if i < len(error):
                error_signal[output_idx] = error[i]
        
        # Extract which neurons were active during the simulation
        active_neurons = set()
        neuron_spike_times = defaultdict(list)
        
        # Record spike times for each neuron
        for t, step_spikes in enumerate(spike_patterns):
            for neuron_idx, _ in step_spikes:
                active_neurons.add(neuron_idx)
                neuron_spike_times[neuron_idx].append(t)
        
        # Calculate eligibility traces based on causal relationships
        eligibility_traces = np.zeros((self.neuron_count, self.neuron_count))
        
        # For each output neuron with an error signal
        for output_idx in self.output_layer:
            if abs(error_signal[output_idx]) > 0.01:  # Only adjust significant errors
                # Find neurons that caused this output to fire (or prevented it)
                output_times = neuron_spike_times.get(output_idx, [])
                
                # For neurons that fired before this output
                for input_idx in active_neurons:
                    if input_idx != output_idx:  # Skip self-connections
                        input_times = neuron_spike_times.get(input_idx, [])
                        
                        # Calculate causal effect (how many times input preceded output)
                        causal_effect = 0
                        for in_time in input_times:
                            for out_time in output_times:
                                # If input spike could have caused output spike (within STDP window)
                                if 0 < out_time - in_time <= 3:  # 3 timesteps window
                                    causal_effect += 1
                        
                        # Set eligibility trace based on causal effect
                        if causal_effect > 0:
                            # Normalize by number of input spikes
                            eligibility_traces[input_idx, output_idx] = min(1.0, causal_effect / len(input_times))
        
        # Apply weight updates based on eligibility traces and error signals
        for pre in range(self.neuron_count):
            for post in self.output_layer:
                if eligibility_traces[pre, post] > 0:
                    # Calculate weight update based on error and eligibility
                    # Direction: increase weights if error is positive (output should be higher)
                    # decrease weights if error is negative (output should be lower)
                    weight_change = self.learning_rate * error_signal[post] * eligibility_traces[pre, post]
                    
                    # Apply weight change with constraints
                    current_weight = self.synaptic_weights[pre, post]
                    new_weight = max(0.0, min(1.0, current_weight + weight_change))
                    
                    # Only update if there's a meaningful change
                    if abs(new_weight - current_weight) > 0.001:
                        self.synaptic_weights[pre, post] = new_weight
        
        # Apply STDP for all active neurons (delegate to parent's STDP mechanism)
        for t, step_spikes in enumerate(spike_patterns):
            if len(step_spikes) > 1:
                # Extract spiking neurons at this timestep
                spiking_indices = [idx for idx, _ in step_spikes]
                # Apply parent's STDP mechanism
                self._apply_stdp(spiking_indices, time.time() + 0.001 * t)
        
        # Apply homeostatic plasticity if available
        if hasattr(self, '_apply_homeostatic') and callable(getattr(self, '_apply_homeostatic')):
            self._apply_homeostatic()

    def _store_reasoning_episode(self, reasoning_trace, trace_features, quality_score, reasoning_issues):
        """Store reasoning episode for future learning."""
        episode = {
            'timestamp': time.time(),
            'features': trace_features,
            'quality': quality_score,
            'issues': reasoning_issues,
            'trace_summary': self._summarize_reasoning_trace(reasoning_trace)
        }
        
        self.reasoning_episodes.append(episode)
        
        # Update pattern detection if we have enough episodes
        if len(self.reasoning_episodes) >= 5:
            self._update_reasoning_patterns()

    def _summarize_reasoning_trace(self, reasoning_trace):
        """Create a compact summary of a reasoning trace."""
        if not reasoning_trace:
            return {}
            
        # Extract key information
        summary = {
            'steps': len(reasoning_trace) if isinstance(reasoning_trace, list) else 0,
            'operations': [],
            'conclusion': None
        }
        
        if isinstance(reasoning_trace, list):
            # Extract unique operations
            operations = [step.get('operation') for step in reasoning_trace if 'operation' in step]
            summary['operations'] = list(set(operations))
            
            # Extract final conclusion if available
            if reasoning_trace and 'conclusion' in reasoning_trace[-1]:
                summary['conclusion'] = reasoning_trace[-1]['conclusion']
        
        return summary

    def _update_reasoning_patterns(self):
        """Analyze reasoning episodes to identify patterns."""
        # Skip if too few episodes
        if len(self.reasoning_episodes) < 5:
            return
        
        # Look for patterns in successful reasoning
        successful_episodes = [e for e in self.reasoning_episodes if e.get('quality', 0) > 0.7]
        
        # Look for patterns in problematic reasoning
        problematic_episodes = [e for e in self.reasoning_episodes if e.get('quality', 0) < 0.5]
        
        # Extract common features in successful reasoning
        success_patterns = self._extract_common_patterns(successful_episodes)
        
        # Extract common issues in problematic reasoning
        problem_patterns = self._extract_common_patterns(problematic_episodes, field='issues')
        
        # Update detected patterns
        self.detected_patterns = {
            'success_patterns': success_patterns,
            'problem_patterns': problem_patterns,
            'last_updated': time.time()
        }

    def _extract_common_patterns(self, episodes, field='features'):
        """Extract common patterns from a list of reasoning episodes."""
        if not episodes:
            return []
            
        patterns = []
        
        if field == 'features':
            # Aggregate feature statistics
            feature_values = defaultdict(list)
            
            for episode in episodes:
                features = episode.get('features', {})
                for feature, value in features.items():
                    feature_values[feature].append(value)
            
            # Calculate average and standard deviation for each feature
            feature_stats = {}
            for feature, values in feature_values.items():
                if values:
                    avg = sum(values) / len(values)
                    std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                    feature_stats[feature] = {'avg': avg, 'std': std}
            
            # Create patterns based on notable feature statistics
            for feature, stats in feature_stats.items():
                if stats['std'] < 0.2:  # Low variability indicates a pattern
                    patterns.append({
                        'type': 'feature',
                        'feature': feature,
                        'avg_value': stats['avg'],
                        'consistency': 1.0 - stats['std']
                    })
        
        elif field == 'issues':
            # Count issue occurrences
            issue_counts = defaultdict(int)
            total_episodes = len(episodes)
            
            for episode in episodes:
                issues = episode.get('issues', [])
                # Track unique issue types per episode
                seen_types = set()
                for issue in issues:
                    issue_type = issue.get('type', 'unknown')
                    if issue_type not in seen_types:
                        issue_counts[issue_type] += 1
                        seen_types.add(issue_type)
            
            # Create patterns for frequent issues
            for issue_type, count in issue_counts.items():
                frequency = count / total_episodes
                if frequency > 0.4:  # Issue appears in at least 40% of problematic episodes
                    patterns.append({
                        'type': 'issue',
                        'issue_type': issue_type,
                        'frequency': frequency
                    })
        
        return patterns

    def _calculate_phi_trend(self):
        """Calculate trend in phi (integration) values."""
        if not self.phi_history or len(self.phi_history) < 2:
            return {
                'direction': 'stable',
                'magnitude': 0.0
            }
        
        # Get recent phi values
        recent_phis = list(self.phi_history)
        
        # Calculate simple linear trend
        if len(recent_phis) >= 5:
            # Use last 5 values for trend
            recent_phis = recent_phis[-5:]
        
        # Calculate slope
        x = list(range(len(recent_phis)))
        y = recent_phis
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_xx = sum(x_i ** 2 for x_i in x)
        
        # Calculate slope
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        except ZeroDivisionError:
            slope = 0
        
        # Determine direction and magnitude
        direction = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        magnitude = abs(slope)
        
        return {
            'direction': direction,
            'magnitude': magnitude
        }

    def adjust_system_strategy(self, current_system_state, processing_history=None):
        """
        Recommend strategy adjustments based on system state.
        
        Args:
            current_system_state: Dictionary with system state info
            processing_history: Optional history of recent processing
            
        Returns:
            Strategy adjustment recommendations
        """
        # First monitor system to update metacognitive state
        monitoring_result = self.monitor_system_state(
            current_system_state, 
            recent_processing=processing_history
        )
        
        # Get current metacognitive signals
        signals = monitoring_result.get('metacognitive_signals', {})
        
        # Prepare strategy adjustments
        adjustments = {
            'subsystem_weights': {},
            'processing_parameters': {},
            'attention_allocation': {},
            'confidence_threshold': None
        }
        
        # 1. Adjust subsystem weights based on metacognitive state
        if 'attention_allocation' in signals:
            attention_signals = signals['attention_allocation']
            
            # Focus more on specific subsystem if needed
            focus_target = attention_signals.get('focus_on')
            if focus_target:
                # Increase weight for focus target
                adjustments['subsystem_weights'][focus_target] = 1.5
                
                # Slightly decrease others to maintain total influence
                other_systems = [
                    system for system in current_system_state.get('subsystem_activities', {}).keys()
                    if system != focus_target
                ]
                for system in other_systems:
                    adjustments['subsystem_weights'][system] = 0.8
        
        # 2. Adjust processing parameters
        if 'strategy_adjustments' in signals:
            strategy_signals = signals['strategy_adjustments']
            
            if strategy_signals.get('increase_recurrent_loops', False):
                # Increase recurrent processing
                adjustments['processing_parameters']['max_recurrent_loops'] = 7  # Increased from default
                
            if strategy_signals.get('change_reasoning_approach', False):
                # Change inference approach
                adjustments['processing_parameters']['inference_strategy'] = 'mixed'  # Use mixed strategy instead of default
                
            if strategy_signals.get('verify_outputs', False):
                # Lower confidence threshold for extra verification
                adjustments['confidence_threshold'] = 0.8  # Higher threshold for accepting outputs
        
        # 3. Special adjustment: Integration boost
        if self.current_phi < 0.4:
            # Low integration, increase interactions between components
            adjustments['processing_parameters']['component_interaction'] = 1.5  # Boost interactions
            adjustments['processing_parameters']['force_cross_component_feedback'] = True
        
        # 4. Learning-based adjustments
        if 'learning_adjustments' in signals:
            learning_signals = signals['learning_adjustments']
            
            if learning_signals.get('extract_reasoning_patterns', False):
                # Recommend applying learned patterns
                adjustments['processing_parameters']['apply_learned_patterns'] = True
                
                # Include successful patterns if available
                if 'success_patterns' in self.detected_patterns:
                    adjustments['learned_patterns'] = self.detected_patterns['success_patterns']
        
        return adjustments

    def get_metacognitive_state_summary(self):
        """
        Get a summary of the current metacognitive state.
        
        Returns:
            Dictionary with summary of metacognitive state
        """
        summary = {
            'confidence': self.metacognitive_state.get('system_confidence', 0.5),
            'reasoning_quality': self.metacognitive_state.get('reasoning_quality', 0.5),
            'attention_focus': self.metacognitive_state.get('attention_focus'),
            'integration_level': self.current_phi,
            'integration_trend': self._calculate_phi_trend(),
            'detected_issues': len(self.metacognitive_state.get('detected_issues', [])),
            'active_regions': []
        }
        
        # Add active metacognitive regions
        active_regions = []
        for region_name, region_info in self.regions.items():
            activation = region_info.get('activation', 0)
            if activation > 0.4:  # Only include significantly active regions
                active_regions.append({
                    'name': region_name,
                    'activation': activation
                })
        
        summary['active_regions'] = sorted(active_regions, key=lambda x: x['activation'], reverse=True)
        
        return summary

    def train_monitoring(self, system_state, issues, learn_rate=0.01):
        """
        Train the metacognitive SNN to monitor system state and detect issues.
        
        Args:
            system_state: Dictionary containing system state metrics
            issues: List of detected issues
            learn_rate: Learning rate for this training instance
            
        Returns:
            Dictionary containing training results
        """
        # Encode system state for neural processing
        input_pattern = self._encode_system_state_for_training(system_state)
        
        # Simulate network response
        spike_patterns = self._simulate_network(steps=15)
        
        # Determine expected output based on issues and system state
        target_confidence = self._calculate_target_confidence(system_state, issues)
        target_output = self._encode_target_output_monitoring(target_confidence, issues)
        
        # Train on this example
        training_error = self._train_on_monitoring_example(input_pattern, target_output, spike_patterns, learn_rate)
        
        # Get current network output
        current_output = self._get_output_pattern()
        
        # Decode monitoring results
        detected_issues = self._decode_issues_from_output(current_output)
        confidence = self._decode_confidence_from_output(current_output)
        
        # Return training results
        return {
            'success': True,
            'confidence': confidence,
            'detected_issues': detected_issues,
            'training_error': training_error
        }
    
    def _encode_system_state_for_training(self, system_state):
        """Encode system state into neural activation pattern for training"""
        # Initialize activation pattern
        activation = np.zeros(self.neuron_count)
        
        # Extract key metrics
        confidence = system_state.get('confidence', 0.5)
        error_rate = system_state.get('error_rate', 0.0)
        processing_time = system_state.get('processing_time', 1.0)
        
        # Map to specific neuron groups in the monitoring region
        monitoring_neurons = self.regions['monitoring']['neurons']
        
        # Distribute values across neuron groups (simplified encoding)
        group_size = len(monitoring_neurons) // 3
        
        # Confidence neurons (first third)
        confidence_neurons = monitoring_neurons[:group_size]
        for i, n in enumerate(confidence_neurons):
            if i < int(confidence * group_size):
                activation[n] = 0.8 + 0.2 * random.random()
        
        # Error rate neurons (second third)
        error_neurons = monitoring_neurons[group_size:2*group_size]
        for i, n in enumerate(error_neurons):
            if i < int(error_rate * group_size):
                activation[n] = 0.8 + 0.2 * random.random()
        
        # Processing time neurons (normalization to 0-1 range, last third)
        time_neurons = monitoring_neurons[2*group_size:]
        norm_time = min(1.0, processing_time / 2.0)  # Normalize to [0,1]
        for i, n in enumerate(time_neurons):
            if i < int(norm_time * len(time_neurons)):
                activation[n] = 0.8 + 0.2 * random.random()
        
        return activation
    
    def _calculate_target_confidence(self, system_state, issues):
        """Calculate target confidence based on system state and issues"""
        # Base confidence from system state
        base_confidence = system_state.get('confidence', 0.5)
        
        # Adjust based on error rate and number of issues
        error_rate = system_state.get('error_rate', 0.0)
        issue_penalty = len(issues) * 0.05
        
        # Calculate adjusted confidence
        adjusted_confidence = base_confidence * (1.0 - error_rate) - issue_penalty
        
        # Ensure within valid range [0.1, 0.9]
        return max(0.1, min(0.9, adjusted_confidence))
    
    def _encode_target_output_monitoring(self, confidence, issues):
        """Encode target output pattern for monitoring training"""
        # Initialize target output
        target_output = np.zeros(self.neuron_count)
        
        # Encode confidence in confidence estimation region
        confidence_neurons = self.regions['confidence_estimation']['neurons']
        conf_group_size = len(confidence_neurons) // 3
        
        # Low confidence neurons
        if confidence < 0.33:
            for n in confidence_neurons[:conf_group_size]:
                target_output[n] = 0.8 + 0.2 * random.random()
        # Medium confidence neurons
        elif confidence < 0.66:
            for n in confidence_neurons[conf_group_size:2*conf_group_size]:
                target_output[n] = 0.8 + 0.2 * random.random()
        # High confidence neurons
        else:
            for n in confidence_neurons[2*conf_group_size:]:
                target_output[n] = 0.8 + 0.2 * random.random()
        
        # Encode issues in contradiction detection region
        if issues:
            contradiction_neurons = self.regions['contradiction_detection']['neurons']
            neurons_per_issue = max(1, len(contradiction_neurons) // 10)
            
            for i, issue in enumerate(issues[:10]):  # Limit to 10 issues
                issue_idx = hash(issue) % 10  # Map issue to index 0-9
                start_idx = issue_idx * neurons_per_issue
                end_idx = start_idx + neurons_per_issue
                
                # Ensure within range
                start_idx = min(start_idx, len(contradiction_neurons) - neurons_per_issue)
                end_idx = min(end_idx, len(contradiction_neurons))
                
                for n in contradiction_neurons[start_idx:end_idx]:
                    target_output[n] = 0.8 + 0.2 * random.random()
        
        return target_output
    
    def _train_on_monitoring_example(self, input_pattern, target_output, spike_patterns, learn_rate):
        """Train network on a monitoring example"""
        # Apply input pattern
        self._apply_input_pattern(input_pattern)
        
        # Get current output
        current_output = self._get_output_pattern()
        
        # Calculate error (ensure arrays have the same shape)
        output_error = np.zeros_like(current_output)
        min_len = min(len(target_output), len(current_output))
        output_error[:min_len] = target_output[:min_len] - current_output[:min_len]
        
        # Adjust weights based on error and spike patterns
        total_error = 0.0
        
        # Extract active neurons from spike patterns
        active_neurons = set()
        for time_step, spikes in enumerate(spike_patterns):
            for neuron, _ in spikes:
                active_neurons.add(neuron)
        
        # Adjust weights for active neurons that contributed to output
        for pre in active_neurons:
            for post in range(self.neuron_count):
                # Skip if no connection or post neuron is out of range
                if self.synaptic_weights[pre, post] == 0:
                    continue
                
                # Calculate weight adjustment (simplified)
                if post in self.output_layer and post < len(output_error):
                    # For output neurons, adjust based on output error
                    error = output_error[post]
                    total_error += abs(error)
                    
                    # Adjust weight
                    delta_w = learn_rate * error * 0.1
                    
                    # Update weight (in sparse matrix)
                    self.synaptic_weights[pre, post] += delta_w
        
        # Return average error
        return total_error / max(1, len(self.output_layer))
    
    def _decode_issues_from_output(self, output_pattern):
        """Decode detected issues from output pattern"""
        # Get contradiction detection neurons
        contradiction_neurons = self.regions['contradiction_detection']['neurons']
        neurons_per_issue = max(1, len(contradiction_neurons) // 10)
        
        # Check activation patterns for each potential issue
        detected_issues = []
        
        for issue_idx in range(10):  # Check all 10 possible issues
            start_idx = issue_idx * neurons_per_issue
            end_idx = start_idx + neurons_per_issue
            
            # Ensure within range
            start_idx = min(start_idx, len(contradiction_neurons) - neurons_per_issue)
            end_idx = min(end_idx, len(contradiction_neurons))
            
            # Calculate average activation for this issue's neurons
            issue_neurons = contradiction_neurons[start_idx:end_idx]
            
            # Safely access output_pattern with bounds checking
            valid_activations = []
            for n in issue_neurons:
                if 0 <= n < len(output_pattern):
                    valid_activations.append(output_pattern[n])
                
            # Calculate average activation if we have valid neurons
            if valid_activations:
                activation = sum(valid_activations) / len(valid_activations)
                
                # If activation exceeds threshold, add issue
                if activation > 0.5:
                    detected_issues.append(f"issue_{issue_idx}")
        
        return detected_issues
    
    def _decode_confidence_from_output(self, output_pattern):
        """Extract confidence estimate from neural output pattern"""
        conf_neurons = self.decoders['confidence']['neurons']
        confidence_activations = output_pattern[conf_neurons]
        
        # Calculate average activation
        if len(confidence_activations) > 0:
            confidence = np.mean(confidence_activations)
        else:
            confidence = 0.5  # Default confidence value
            
        return confidence
    
    def state_dict(self):
        """
        Returns a dictionary containing the whole state of the module.
        This method enables compatibility with PyTorch's save functionality.
        
        Returns:
            dict: A dictionary containing model parameters and persistent buffers.
        """
        # Create a state dictionary with all the necessary components
        state = {}
        
        # Add neural network parameters
        state['membrane_potentials'] = self.membrane_potentials.copy() if hasattr(self, 'membrane_potentials') else None
        state['spike_thresholds'] = self.spike_thresholds.copy() if hasattr(self, 'spike_thresholds') else None
        
        # Handle sparse synaptic weights
        if hasattr(self, 'synaptic_weights'):
            if hasattr(self.synaptic_weights, 'toarray'):
                # For sparse matrices
                state['synaptic_weights_data'] = self.synaptic_weights.data
                state['synaptic_weights_indices'] = self.synaptic_weights.indices
                state['synaptic_weights_indptr'] = self.synaptic_weights.indptr
                state['synaptic_weights_shape'] = self.synaptic_weights.shape
            else:
                # For dense matrices
                state['synaptic_weights'] = self.synaptic_weights.copy()
        
        # Add metacognitive state tracking
        if hasattr(self, 'metacognitive_state'):
            state['metacognitive_state'] = self.metacognitive_state.copy()
        
        # Add region definitions (convert sets to lists for serialization)
        if hasattr(self, 'regions'):
            serializable_regions = {}
            for region_name, region_data in self.regions.items():
                # Create a new dict for this region
                region_copy = {}
                for k, v in region_data.items():
                    if isinstance(v, set):
                        region_copy[k] = list(v)
                    else:
                        region_copy[k] = v
                serializable_regions[region_name] = region_copy
            state['regions'] = serializable_regions
        
        # Add region connections
        if hasattr(self, 'region_connections'):
            state['region_connections'] = self.region_connections.copy()
        
        # Add decoders
        if hasattr(self, 'decoders'):
            state['decoders'] = self.decoders.copy()
        
        # Add learning parameters
        state['learning_rate'] = self.learning_rate if hasattr(self, 'learning_rate') else 0.01
        state['plasticity_rates'] = self.plasticity_rates.copy() if hasattr(self, 'plasticity_rates') else None
        state['hebbian_rate'] = self.hebbian_rate if hasattr(self, 'hebbian_rate') else 0.008
        
        # Add component state and performance tracking
        if hasattr(self, 'component_state_history'):
            # Convert defaultdict to regular dict and deque to list
            state['component_state_history'] = {
                k: list(v) for k, v in self.component_state_history.items()
            } if self.component_state_history else {}
        
        if hasattr(self, 'component_performance'):
            state['component_performance'] = self.component_performance.copy()
        
        # Add reasoning episodes (convert deque to list)
        if hasattr(self, 'reasoning_episodes'):
            state['reasoning_episodes'] = list(self.reasoning_episodes)
        
        # Add other attributes
        if hasattr(self, 'detected_patterns'):
            state['detected_patterns'] = self.detected_patterns.copy()
        
        if hasattr(self, 'strategy_evaluations'):
            state['strategy_evaluations'] = self.strategy_evaluations.copy()
            
        return state
    
    def load_state_dict(self, state_dict):
        """
        Copies parameters and buffers from state_dict into this module.
        This method enables compatibility with PyTorch's load functionality.
        
        Args:
            state_dict (dict): A dictionary containing parameters and buffers.
        """
        # Load neural network parameters
        if 'membrane_potentials' in state_dict and state_dict['membrane_potentials'] is not None:
            self.membrane_potentials = state_dict['membrane_potentials'].copy()
            
        if 'spike_thresholds' in state_dict and state_dict['spike_thresholds'] is not None:
            self.spike_thresholds = state_dict['spike_thresholds'].copy()
        
        # Load synaptic weights, handling both sparse and dense formats
        if 'synaptic_weights' in state_dict and state_dict['synaptic_weights'] is not None:
            # Dense matrix format
            self.synaptic_weights = state_dict['synaptic_weights'].copy()
        elif all(k in state_dict for k in ['synaptic_weights_data', 'synaptic_weights_indices', 'synaptic_weights_indptr', 'synaptic_weights_shape']):
            # Sparse matrix format
            from scipy import sparse
            self.synaptic_weights = sparse.csr_matrix(
                (state_dict['synaptic_weights_data'],
                 state_dict['synaptic_weights_indices'],
                 state_dict['synaptic_weights_indptr']),
                shape=state_dict['synaptic_weights_shape']
            )
        
        # Load metacognitive state
        if 'metacognitive_state' in state_dict:
            self.metacognitive_state = state_dict['metacognitive_state'].copy()
        
        # Load regions, converting lists back to sets where needed
        if 'regions' in state_dict:
            for region_name, region_data in state_dict['regions'].items():
                if region_name not in self.regions:
                    self.regions[region_name] = {}
                    
                for k, v in region_data.items():
                    # Convert lists back to sets if needed
                    if k in self.regions[region_name] and isinstance(self.regions[region_name][k], set):
                        self.regions[region_name][k] = set(v)
                    else:
                        self.regions[region_name][k] = v
        
        # Load region connections
        if 'region_connections' in state_dict:
            self.region_connections = state_dict['region_connections'].copy()
        
        # Load decoders
        if 'decoders' in state_dict:
            self.decoders = state_dict['decoders'].copy()
        
        # Load learning parameters
        if 'learning_rate' in state_dict:
            self.learning_rate = state_dict['learning_rate']
            
        if 'plasticity_rates' in state_dict and state_dict['plasticity_rates'] is not None:
            self.plasticity_rates = state_dict['plasticity_rates'].copy()
            
        if 'hebbian_rate' in state_dict:
            self.hebbian_rate = state_dict['hebbian_rate']
        
        # Load component state history, converting dict of lists back to defaultdict of deques
        if 'component_state_history' in state_dict:
            from collections import defaultdict, deque
            self.component_state_history = defaultdict(lambda: deque(maxlen=50))
            for k, v in state_dict['component_state_history'].items():
                self.component_state_history[k] = deque(v, maxlen=50)
        
        # Load component performance
        if 'component_performance' in state_dict:
            self.component_performance = state_dict['component_performance'].copy()
        
        # Load reasoning episodes, converting list back to deque
        if 'reasoning_episodes' in state_dict:
            from collections import deque
            max_len = self.reasoning_episodes.maxlen if hasattr(self, 'reasoning_episodes') else 200
            self.reasoning_episodes = deque(state_dict['reasoning_episodes'], maxlen=max_len)
        
        # Load other attributes
        if 'detected_patterns' in state_dict:
            self.detected_patterns = state_dict['detected_patterns'].copy()
            
        if 'strategy_evaluations' in state_dict:
            self.strategy_evaluations = state_dict['strategy_evaluations'].copy()
        
        return self

    def _init_metacognitive_regions(self, custom_regions=None):
        """
        Initialize neural regions specialized for metacognitive functions.
        
        Args:
            custom_regions: Optional dictionary of custom region definitions
        """
        total_neurons = self.neuron_count
        
        # Default metacognitive regions if none provided
        default_regions = {
            # Monitors overall system state and integration
            'monitoring': {
                'start': 0,
                'end': int(0.15 * total_neurons),
                'activation': 0.0
            },
            # Detects contradictions and inconsistencies
            'contradiction_detection': {
                'start': int(0.15 * total_neurons),
                'end': int(0.30 * total_neurons),
                'activation': 0.0
            },
            # Estimates confidence and uncertainty
            'confidence_estimation': {
                'start': int(0.30 * total_neurons),
                'end': int(0.45 * total_neurons),
                'activation': 0.0
            },
            # Controls attention allocation across components
            'attention_regulation': {
                'start': int(0.45 * total_neurons),
                'end': int(0.60 * total_neurons),
                'activation': 0.0
            },
            # Evaluates reasoning quality
            'reasoning_evaluation': {
                'start': int(0.60 * total_neurons),
                'end': int(0.75 * total_neurons),
                'activation': 0.0
            },
            # Selects and adjusts cognitive strategies
            'strategy_selection': {
                'start': int(0.75 * total_neurons),
                'end': int(0.90 * total_neurons),
                'activation': 0.0
            },
            # Learns from experience and metacognitive patterns
            'metalearning': {
                'start': int(0.90 * total_neurons),
                'end': total_neurons,
                'activation': 0.0
            }
        }
        
        # Use custom regions if provided, otherwise use defaults
        regions_def = custom_regions if custom_regions else default_regions
        
        # Create or update regions in parent's region system
        for region_name, region_info in regions_def.items():
            # Create list of neuron indices for this region
            start = region_info['start']
            end = region_info['end']
            neurons = list(range(start, end))
            
            # Add or update region in parent class regions
            if not hasattr(self, 'regions'):
                self.regions = {}
                
            self.regions[region_name] = {
                'neurons': neurons,
                'activation': 0.0
            }
        
        # Set up region connectivity (which regions connect to which)
        self.region_connections = {
            'monitoring': ['contradiction_detection', 'confidence_estimation', 'attention_regulation'],
            'contradiction_detection': ['reasoning_evaluation', 'confidence_estimation', 'strategy_selection'],
            'confidence_estimation': ['attention_regulation', 'strategy_selection', 'reasoning_evaluation'],
            'attention_regulation': ['monitoring', 'strategy_selection'],
            'reasoning_evaluation': ['metalearning', 'strategy_selection', 'confidence_estimation'],
            'strategy_selection': ['metalearning', 'attention_regulation'],
            'metalearning': ['strategy_selection', 'reasoning_evaluation', 'monitoring']
        }
        
        # Connect regions using parent synaptic weights
        self._connect_metacognitive_regions()

    def _connect_metacognitive_regions(self):
        """
        Create connections between metacognitive regions using parent synaptic weights.
        """
        connection_count = 0
        
        # For each region connection defined in self.region_connections
        for source_region, target_regions in self.region_connections.items():
            if source_region not in self.regions:
                continue
                
            source_neurons = self.regions[source_region]['neurons']
            
            # Connect to each target region
            for target_region in target_regions:
                if target_region not in self.regions:
                    continue
                    
                target_neurons = self.regions[target_region]['neurons']
                
                # Create connections with higher density between related regions
                connection_density = 0.3  # Higher density for metacognitive connections
                
                # Calculate number of connections to create
                if hasattr(self, 'synaptic_weights') and isinstance(self.synaptic_weights, sparse.spmatrix):
                    # Using sparse matrix
                    num_connections = int(len(source_neurons) * len(target_neurons) * connection_density)
                    num_connections = min(num_connections, 1000)  # Limit to reasonable number
                    
                    for _ in range(num_connections):
                        # Select random neuron from source and target
                        source_idx = random.choice(source_neurons)
                        target_idx = random.choice(target_neurons)
                        
                        # Set weight in sparse matrix
                        if source_idx != target_idx:  # Avoid self-connections
                            # Higher initial weight for metacognitive pathways
                            weight = 0.3 + 0.2 * random.random()
                            self.synaptic_weights[source_idx, target_idx] = weight
                            connection_count += 1
        
        print(f"[MetaSNN] Created {connection_count} connections between metacognitive regions")

    def process_text_input(self, text_input, timesteps=20):
        """
        Process text input using the standardized bidirectional processor.
        
        Args:
            text_input: Text input for metacognitive processing
            timesteps: Number of timesteps for spike patterns
            
        Returns:
            Processed spike patterns for metacognitive processing
        """
        # Check if we have bidirectional processor
        if hasattr(self, 'bidirectional_processor') and self.bidirectional_processor is not None:
            try:
                # Call parent method to get spike patterns
                spike_patterns = super().process_text_input(
                    text_input, 
                    timesteps=timesteps, 
                    add_special_tokens=True
                )
                
                # Process with metacognitive analysis if needed
                return spike_patterns
            except Exception as e:
                print(f"[MetacognitiveSNN] Error processing text with bidirectional processor: {e}")
                # Fall back to legacy encoding
        
        # Legacy encoding if bidirectional processor is not available or error occurs
        # Use simplified encoding for text input
        # Convert text to activation pattern
        activation = self._encode_text_input_legacy(text_input)
        # Simulate spiking to get patterns
        spike_patterns = self._simulate_network(timesteps)
        
        return spike_patterns
    
    def generate_text_output(self, spike_patterns, max_length=100):
        """
        Generate text output from metacognitive spike patterns using the standardized bidirectional processor.
        
        Args:
            spike_patterns: Spike patterns from metacognitive processing
            max_length: Maximum length of generated text
            
        Returns:
            Generated text with metacognitive information
        """
        # Check if we have bidirectional processor
        if hasattr(self, 'bidirectional_processor') and self.bidirectional_processor is not None:
            try:
                # Use the parent's implementation for standardized text generation
                text_output = super().generate_text_output(
                    spike_patterns,
                    max_length=max_length,
                    remove_special_tokens=True
                )
                return text_output
            except Exception as e:
                print(f"[MetacognitiveSNN] Error generating text with bidirectional processor: {e}")
                # Fall back to legacy approach
        
        # Legacy approach for text generation
        # Decode output pattern from spike patterns
        output_pattern = self._get_output_pattern()
        # Decode metacognitive assessment from output
        metacognitive_assessment = self.decode_metacognitive_output(output_pattern, spike_patterns)
        
        # Generate text based on metacognitive assessment
        if metacognitive_assessment:
            confidence = metacognitive_assessment.get('confidence', 0.5)
            issues = metacognitive_assessment.get('detected_issues', [])
            suggestions = metacognitive_assessment.get('suggestions', [])
            
            # Format output text
            output_text = f"Metacognitive assessment (confidence: {confidence:.2f}):\n"
            
            if issues:
                output_text += "Detected issues:\n"
                for issue in issues[:3]:  # Limit to top 3 issues
                    output_text += f"- {issue.get('description', 'Unknown issue')}\n"
            
            if suggestions:
                output_text += "Suggestions:\n"
                for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
                    output_text += f"- {suggestion.get('description', 'Unknown suggestion')}\n"
            
            return output_text
        
        return "No significant metacognitive assessment available"
    
    def _encode_text_input_legacy(self, text_input):
        """
        Legacy method to encode text input into neural activation pattern.
        
        Args:
            text_input: Input text to process
            
        Returns:
            Neural activation pattern
        """
        # Create blank activation
        activation = np.zeros(self.neuron_count)
        
        # Simple word-based encoding
        words = text_input.lower().split()
        
        # Get input layer neurons
        input_neurons = self.input_layer if hasattr(self, 'input_layer') else list(range(int(self.neuron_count * 0.1)))
        
        if input_neurons and words:
            neurons_per_word = max(1, len(input_neurons) // (len(words) + 1))
            
            for i, word in enumerate(words):
                start_idx = (i * neurons_per_word) % len(input_neurons)
                end_idx = min(start_idx + neurons_per_word, len(input_neurons))
                
                # Set activation for these neurons
                for j in range(start_idx, end_idx):
                    neuron_idx = input_neurons[j]
                    # Decay factor for position
                    position_factor = 1.0 - (0.01 * i)
                    activation[neuron_idx] = 0.7 * position_factor
        
        return activation