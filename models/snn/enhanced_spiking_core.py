# cogmenta_core/models/snn/enhanced_spiking_core.py
import re
import random
import numpy as np
import math
import time
from collections import defaultdict, deque
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from cognitive.thought_tracer import *

class EnhancedSpikingCore:
    """
    Enhanced Spiking Neural Network core that incorporates principles from
    Integrated Information Theory (IIT) and Recurrent Processing Theory.
    
    This updated version integrates with NetworkTopology and LearningModule
    for advanced neural assemblies and dynamic learning capabilities.
    
    Optimized for large-scale networks with:
    - Sparse connectivity
    - Modular organization
    - Efficient computation
    """
    
    def __init__(self, neuron_count=1000, connection_density=0.1, region_density=0.6):
        """
        Initialize the SNN with configurable parameters.
        
        Args:
            neuron_count: Total number of neurons
            connection_density: Fraction of possible connections to create (0-1)
            region_density: Connection density within regions (higher than global)
        """
        self.neuron_count = neuron_count
        self.connection_density = min(1.0, max(0.01, connection_density))
        self.region_density = min(1.0, max(0.1, region_density))
        self.spike_threshold = 0.5
        self.decay_rate = 0.9  # Membrane potential decay
        self.membrane_potentials = np.zeros(neuron_count)
        self.abductive_memory = []
        
        # Initialize regions (with scaled neuron counts)
        self._init_regions()
        
        # Initialize weights and concept mappings
        self.synaptic_weights = self._init_sparse_weights()
        self.concept_mappings = self._init_concept_mappings()
        
        # Initialize advanced components
        self._init_learning_module()
        self._init_network_topology()
        
        # Track spikes over time (using sparse representation)
        self.spike_history = defaultdict(lambda: deque(maxlen=100))
        
        # Recurrent activation tracking
        self.recurrent_activations = {}
        self.recurrent_loops = 0
        
        # Integration metrics (IIT-inspired)
        self.phi = 0.0              # Overall integration measure
        self.differentiation = 0.0  # Information differentiation
        self.integration = 0.0      # Information integration
        
        # Store emergent patterns
        self.emergent_patterns = []
        
        # Active neurons cache (for optimization)
        self.active_neurons_cache = set()
        
        print(f"[SNN] Enhanced Spiking Neural Network initialized with {neuron_count} neurons")
        print(f"[SNN] Connection density: {connection_density:.2f}, Region density: {region_density:.2f}")
    
    def _init_regions(self):
        """Initialize neural regions with scaled neuron counts"""
        # Calculate neurons per region (approximately)
        region_size = max(20, self.neuron_count // 6)  # At least 20 neurons per region
        
        # Define region ratios (perception should be larger, etc.)
        region_ratios = {
            'perception': 1.5,
            'working_memory': 1.2,
            'conceptual': 1.0,
            'prediction': 0.8,
            'metacognition': 0.7,
            'action': 0.5
        }
        
        # Calculate actual neurons per region based on ratios
        total_ratio = sum(region_ratios.values())
        neurons_allocated = 0
        
        # Initialize regions dictionary
        self.regions = {}
        
        # Allocate neurons to regions
        for i, (region_name, ratio) in enumerate(region_ratios.items()):
            # Calculate region size
            if i == len(region_ratios) - 1:
                # Last region gets remaining neurons to ensure total equals neuron_count
                r_size = self.neuron_count - neurons_allocated
            else:
                r_size = int((ratio / total_ratio) * self.neuron_count)
            
            # Define region
            self.regions[region_name] = {
                'neurons': list(range(neurons_allocated, neurons_allocated + r_size)),
                'activation': 0.0,
                'connections': [],  # Will be populated below
                'recurrent': region_name != 'action'  # All regions except 'action' are recurrent
            }
            
            # Update allocated count
            neurons_allocated += r_size
        
        # Define connections between regions (brain-inspired architecture)
        connections = {
            'perception': ['working_memory', 'conceptual'],
            'working_memory': ['perception', 'conceptual', 'prediction'],
            'conceptual': ['working_memory', 'prediction', 'metacognition'],
            'prediction': ['working_memory', 'conceptual', 'action'],
            'metacognition': ['conceptual', 'prediction', 'action'],
            'action': ['prediction', 'metacognition']
        }
        
        # Add connections to regions
        for region_name, connected_regions in connections.items():
            if region_name in self.regions:
                self.regions[region_name]['connections'] = connected_regions
                
        print(f"[SNN] Neural regions initialized with {neurons_allocated} neurons")
        for region_name, region in self.regions.items():
            print(f"[SNN]   {region_name}: {len(region['neurons'])} neurons")
    
    def _init_sparse_weights(self) -> sparse.csr_matrix:
        """Initialize sparse synaptic weights"""
        # Calculate expected number of connections
        expected_connections = int(self.neuron_count * self.neuron_count * self.connection_density)
        
        # Create data structures for sparse matrix (COO format initially)
        rows = []
        cols = []
        data = []
        
        # Create region-specific connections with higher density
        for region_name, region in self.regions.items():
            # Within-region connections (higher density)
            region_neurons = region['neurons']
            within_connections = int(len(region_neurons) * len(region_neurons) * self.region_density)
            
            # Generate random connections within region
            for _ in range(within_connections):
                i = random.choice(region_neurons)
                j = random.choice(region_neurons)
                # Small random weight
                weight = np.random.uniform(-0.1, 0.1)
                rows.append(i)
                cols.append(j)
                data.append(weight)
            
            # Between-region connections (standard density)
            for connected_region in region['connections']:
                if connected_region in self.regions:
                    target_neurons = self.regions[connected_region]['neurons']
                    # Calculate number of connections based on regions sizes
                    between_connections = int(
                        len(region_neurons) * 
                        len(target_neurons) * 
                        self.connection_density
                    )
                    
                    # Generate random connections between regions
                    for _ in range(between_connections):
                        i = random.choice(region_neurons)
                        j = random.choice(target_neurons)
                        # Small random weight
                        weight = np.random.uniform(-0.1, 0.1)
                        rows.append(i)
                        cols.append(j)
                        data.append(weight)
        
        # Create sparse matrix in CSR format (efficient for operations)
        weights = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.neuron_count, self.neuron_count)
        )
        
        print(f"[SNN] Sparse synaptic weights initialized with {len(data)} connections")
        print(f"[SNN] Sparsity: {1.0 - (len(data) / (self.neuron_count * self.neuron_count)):.4f}")
        
        return weights
    
    def _init_concept_mappings(self):
        """Initialize concept-to-neuron mappings with scaled sizes"""
        # Calculate neurons per concept (scaled with network size)
        concept_size = max(10, min(self.neuron_count // 50, 100))
        
        # Get available neurons from conceptual region
        conceptual_neurons = self.regions['conceptual']['neurons'].copy()
        
        # Ensure we have enough neurons
        if len(conceptual_neurons) < concept_size * 6:  # 6 basic concepts
            # Not enough neurons, use a reduced set or neurons from other regions
            concept_size = max(5, len(conceptual_neurons) // 6)
            
            # If still not enough, add some neurons from working memory
            if len(conceptual_neurons) < concept_size * 6:
                additional_neurons = self.regions['working_memory']['neurons'][:concept_size * 3]
                conceptual_neurons.extend(additional_neurons)
        
        # Shuffle neurons for random assignment
        random.shuffle(conceptual_neurons)
        
        # Map concepts to groups of neurons
        concepts = {}
        concept_list = ["trust", "distrust", "fear", "like", "hate", "avoid"]
        
        # Assign neurons to concepts
        start_idx = 0
        for concept in concept_list:
            end_idx = start_idx + concept_size
            if end_idx <= len(conceptual_neurons):
                concepts[concept] = conceptual_neurons[start_idx:end_idx]
                start_idx = end_idx
        
        return concepts
    
    def _init_learning_module(self):
        """Initialize the learning module"""
        try:
            # Try to import the LearningModule
            from models.snn.learning import LearningModule
            self.learning = LearningModule(self)
            print("[SNN] Learning module initialized")
        except ImportError:
            self.learning = None
            print("[SNN] Learning module not available")
    
    def _init_network_topology(self):
        """Initialize the neural network topology"""
        try:
            # Try to import the NetworkTopology
            from models.snn.network_topology import NetworkTopology
            self.topology = NetworkTopology(self.neuron_count)
            print("[SNN] Network topology initialized")
        except ImportError:
            self.topology = None
            print("[SNN] Network topology not available")
    
    def _calculate_region_activations(self, spikes_over_time):
        """Calculate region activations based on spike patterns"""
        # Initialize region activations
        for region_name in self.regions:
            self.regions[region_name]['activation'] = 0.0
            
        # Count spikes per neuron
        spike_counts = defaultdict(int)
        for time_step in spikes_over_time:
            for neuron_idx, _ in time_step:
                spike_counts[neuron_idx] += 1
                
        # Calculate activation for each region
        for region_name, region in self.regions.items():
            region_neurons = region['neurons']
            if region_neurons:
                # Calculate average spike count for neurons in this region
                region_spike_count = sum(spike_counts.get(n, 0) for n in region_neurons) 
                activation = min(1.0, region_spike_count / (len(region_neurons) * 0.7))
                region['activation'] = activation
    
    def _propagate_activation(self, initial_region, strength=0.8, max_steps=5):
        """
        Propagate activation through the neural network using recurrent processing.
        Implementation of Recurrent Processing Theory for consciousness emergence.
        
        Args:
            initial_region: Region to initially activate
            strength: Initial activation strength
            max_steps: Maximum propagation steps
        
        Returns:
            Dictionary of region activations
        """
        # Set initial activation
        if initial_region in self.regions:
            self.regions[initial_region]['activation'] = strength
        
        # Reset recurrent loop counter
        self.recurrent_loops = 0
        
        # If we have a network topology, use it for propagation
        if self.topology:
            # First, activate the corresponding assembly in the topology
            region_to_assembly = {
                'perception': 'visual',
                'working_memory': 'episodic',
                'conceptual': 'concepts',
                'prediction': 'planning',
                'metacognition': 'reflection',
                'action': 'action'
            }
            
            if initial_region in region_to_assembly:
                assembly_name = region_to_assembly[initial_region]
                if assembly_name in self.topology.assemblies:
                    self.topology.activate_assembly(assembly_name, strength)
            
            # Propagate through the topology
            self.topology.propagate_activation(steps=max_steps)
            
            # Update recurrent loops
            self.recurrent_loops = max_steps
            
            # Update region activations based on topology
            for region_name, region in self.regions.items():
                if region_name in region_to_assembly:
                    assembly_name = region_to_assembly[region_name]
                    if assembly_name in self.topology.assemblies:
                        region['activation'] = self.topology.assemblies[assembly_name]['activation']
            
            # Get active assemblies for recurrent history
            active_assemblies = self.topology.get_active_assemblies()
            self.recurrent_activations = {step: {assembly: activation for assembly, activation in active_assemblies}
                                        for step in range(max_steps)}
        else:
            # Activation propagation using basic regions
            for step in range(max_steps):
                # Increment recurrent loop counter
                self.recurrent_loops += 1
                
                # Store previous activations for recurrent analysis
                previous_activations = {r: self.regions[r]['activation'] for r in self.regions}
                
                # Propagate activations
                new_activations = {}
                for region_name, region in self.regions.items():
                    # Start with decay of current activation
                    new_act = region['activation'] * 0.8  # Decay factor
                    
                    # Add incoming activations from connected regions
                    for connected_region in region['connections']:
                        if connected_region in self.regions:
                            # Connection strength decreases with distance in the network
                            connection_strength = 0.6
                            new_act += self.regions[connected_region]['activation'] * connection_strength
                    
                    # Add recurrent activation if enabled
                    if region['recurrent'] and region['activation'] > 0.1:
                        recurrent_boost = region['activation'] * 0.3  # Recurrent strength
                        new_act += recurrent_boost
                        
                    # Apply sigmoid-like activation function to keep values in [0,1]
                    new_act = 1.0 / (1.0 + math.exp(-new_act + 0.5))
                    
                    new_activations[region_name] = new_act
                
                # Update all activations
                for region_name, activation in new_activations.items():
                    self.regions[region_name]['activation'] = activation
                    
                    # Record in spike history (simplified)
                    if activation > 0.5:
                        self.spike_history[region_name].append({
                            'time': time.time(),
                            'activation': activation,
                            'step': step
                        })
                
                # Store recurrent activation for this step
                self.recurrent_activations[step] = new_activations.copy()
                
                # Check for stable state (little change from previous step)
                total_change = sum(abs(new_activations[r] - previous_activations[r]) for r in new_activations)
                if total_change < 0.1:
                    print(f"[SNN] Activation stabilized after {step+1} steps")
                    break
        
        # Calculate IIT metrics after propagation
        self._calculate_iit_metrics()
        
        return {r: self.regions[r]['activation'] for r in self.regions}
    
    def _calculate_iit_metrics(self):
        """
        Calculate IIT-inspired metrics for the current network state.
        - Phi (Φ): Overall integration measure
        - Differentiation: How varied the activations are
        - Integration: How interconnected the active regions are
        """
        # Get all activations
        activations = [self.regions[r]['activation'] for r in self.regions]
        
        # Calculate differentiation (variance of activations)
        mean_activation = sum(activations) / len(activations)
        variance = sum((a - mean_activation) ** 2 for a in activations) / len(activations)
        self.differentiation = math.sqrt(variance)
        
        # Calculate integration (based on recurrent connections and co-activation)
        total_connections = 0
        active_connections = 0
        
        for region_name, region in self.regions.items():
            if region['activation'] > 0.3:  # Consider regions with significant activation
                for connected in region['connections']:
                    if connected in self.regions:
                        total_connections += 1
                        if self.regions[connected]['activation'] > 0.3:
                            active_connections += 1
        
        self.integration = active_connections / max(total_connections, 1)
        
        # Calculate Phi (simplified version of IIT's Φ)
        # Phi is high when the system is both differentiated and integrated
        self.phi = self.differentiation * self.integration * mean_activation
        
        print(f"[SNN] IIT Metrics - Φ: {self.phi:.4f}, Diff: {self.differentiation:.4f}, Int: {self.integration:.4f}")
        
    def process_input(self, text, trace_id=None):
        """
        Process text input through the spiking neural network
        
        Args:
            text: Input text to process
            trace_id: Optional ID for thought tracing
            
        Returns:
            Processing results including activations and IIT metrics
        """
        # If we have a thought_trace available (passed from bridge)
        if trace_id and hasattr(self, 'thought_trace'):
            # Add initial snn step
            self.thought_trace.add_step(
                trace_id,
                "EnhancedSpikingCore",
                "neural_input",
                {"text": text, "component": "snn"}
            )
            
        # Vectorize input
        input_vector = self._vectorize_input(text)
        
        # Trace the input vector
        if trace_id and hasattr(self, 'thought_trace'):
            self.thought_trace.add_step(
                trace_id,
                "EnhancedSpikingCore",
                "vector_representation",
                {"operation": "Input vectorization", 
                 "active_indices": input_vector.nonzero()[0].tolist()}
            )
        
        # Run simulation 
        spike_patterns = self.simulate_spiking(input_vector, trace_id=trace_id)

        # If learning module is available, update activity trace
        if self.learning:
            # Convert spike patterns to format expected by learning module
            flat_spikes = []
            for time_step in spike_patterns:
                for neuron_idx, spike_strength in time_step:
                    flat_spikes.append((neuron_idx, spike_strength))
            
            self.learning.update_activity_trace(flat_spikes)
        
        # Calculate region activations from spike patterns
        self._calculate_region_activations(spike_patterns)
        
        # Determine primary activation region based on input content
        text_lower = text.lower()
        emotion_words = ['like', 'hate', 'fear', 'trust', 'happy', 'sad', 'angry']
        concept_words = ['knows', 'believes', 'thinks', 'understands', 'remembers']
        
        emotion_count = sum(1 for w in text_lower.split() if w in emotion_words)
        concept_count = sum(1 for w in text_lower.split() if w in concept_words)
        
        # Determine initial region for activation propagation
        if emotion_count > concept_count:
            initial_region = 'conceptual'
            activation_strength = 0.7 + (min(emotion_count, 3) * 0.1)
        else:
            initial_region = 'perception'
            activation_strength = 0.6 + (min(concept_count, 4) * 0.1)
            
        # Propagate activation through regions
        region_activations = self._propagate_activation(initial_region, activation_strength)
        
        # Identify emergent patterns
        emergent_pattern = self._identify_emergent_pattern()
        if emergent_pattern:
            self.emergent_patterns.append(emergent_pattern)
            print(f"[SNN] Identified emergent pattern: {emergent_pattern['name']}")
            
            # If we have a learning module, reinforce this pattern
            if self.learning and emergent_pattern['name'] in ['global_ignition', 'metacognitive_reflection']:
                # These are valuable patterns - provide positive feedback
                active_neurons = []
                if emergent_pattern['name'] == 'global_ignition':
                    active_neurons = self.regions['working_memory']['neurons'] + self.regions['conceptual']['neurons']
                elif emergent_pattern['name'] == 'metacognitive_reflection':
                    active_neurons = self.regions['metacognition']['neurons'] + self.regions['conceptual']['neurons']
                
                # Apply reinforcement learning
                if active_neurons:
                    self.learning.process_feedback(0.5, active_neurons)  # Moderate positive feedback
        
        # If network topology is available, get its state
        topology_state = None
        if self.topology:
            topology_state = self.topology.get_topology_state()
        
        # Track emergence of consciousness-like patterns
        if emergent_pattern and trace_id and hasattr(self, 'thought_trace'):
            self.thought_trace.add_step(
                trace_id,
                "EnhancedSpikingCore",
                "emergent_pattern",
                {"pattern": emergent_pattern['name'], "strength": emergent_pattern['strength']}
            )

        # Return combined results
        return {
            'membrane_potentials': self.membrane_potentials.copy(),
            'region_activations': region_activations,
            'phi': self.phi,
            'differentiation': self.differentiation,
            'integration': self.integration,
            'initial_region': initial_region,
            'activation_strength': activation_strength,
            'emergent_pattern': emergent_pattern,
            'topology_state': topology_state
        }
    
    def _identify_emergent_pattern(self):
        """
        Identify emergent patterns in activation dynamics.
        These represent higher-level "consciousness-like" properties.
        """
        # Check for "Global Ignition" pattern (GWT concept)
        # Occurs when activation rapidly spreads to multiple regions
        if (self.regions['working_memory']['activation'] > 0.7 and
            self.regions['conceptual']['activation'] > 0.6 and
            self.regions['metacognition']['activation'] > 0.5):
            return {
                'name': 'global_ignition',
                'description': 'Global workspace activation pattern',
                'strength': self.regions['working_memory']['activation'],
                'timestamp': time.time()
            }
            
        # Check for "Metacognitive Reflection" pattern
        if (self.regions['metacognition']['activation'] > 0.7 and
            self.regions['conceptual']['activation'] > 0.5 and
            self.phi > 0.5):
            return {
                'name': 'metacognitive_reflection',
                'description': 'Self-reflective processing pattern',
                'strength': self.regions['metacognition']['activation'] * self.phi,
                'timestamp': time.time()
            }
            
        # Check for "Prediction-Perception Loop" pattern (predictive processing)
        if (self.regions['prediction']['activation'] > 0.6 and
            self.regions['perception']['activation'] > 0.5 and
            len(self.recurrent_activations) >= 3):  # At least 3 recurrent steps
            return {
                'name': 'prediction_perception_loop',
                'description': 'Predictive processing pattern',
                'strength': self.regions['prediction']['activation'],
                'timestamp': time.time()
            }
            
        # No recognized pattern
        return None
    
    # Methods adapted from original SpikingCore - optimized for scalability
    
    def _vectorize_input(self, text_input):
        """Convert text input to neural activation pattern"""
        # Very simple vectorization - just for demonstration
        # In a real system, this would use word embeddings or other NLP techniques
        activation = np.zeros(self.neuron_count)
        
        # Simple keyword activation
        keywords = {
            "trust": 0.9,
            "distrust": 0.8,
            "fear": 0.7,
            "like": 0.6,
            "hate": 0.8,
            "avoid": 0.7,
            "no": 0.9,
            "not": 0.8,
            "never": 0.8,
            "none": 0.7,
            "nobody": 0.9
        }
        
        # Activate neurons based on words in input
        text_lower = text_input.lower()
        for keyword, strength in keywords.items():
            if keyword in text_lower:
                # Activate the corresponding neurons
                if keyword in self.concept_mappings:
                    for neuron_idx in self.concept_mappings[keyword]:
                        activation[neuron_idx] = strength
                        
        return activation
        
    def simulate_spiking(self, input_vector, time_steps=5, trace_id=None):
        """
        Simulate network activity over multiple time steps
        
        Args:
            input_vector: Initial input activation
            time_steps: Number of simulation steps
            trace_id: Optional ID for thought tracing
            
        Returns:
            List of spike events over time
        """
        spikes_over_time = []
        
        # Initialize membrane potentials with input
        self.membrane_potentials += input_vector
        
        # Reset active neurons cache
        self.active_neurons_cache = set()
        
        # Run simulation for multiple time steps
        for t in range(time_steps):
            # Determine which neurons spike
            spiking_neurons = self.membrane_potentials > self.spike_threshold
            spiking_indices = np.where(spiking_neurons)[0]
            
            # If no neurons are spiking, we can skip some computation
            if len(spiking_indices) == 0:
                spikes_over_time.append([])
                self.membrane_potentials *= self.decay_rate  # Apply decay
                continue
                
            # Record spikes
            current_spikes = [(i, self.membrane_potentials[i]) 
                             for i in spiking_indices]
            spikes_over_time.append(current_spikes)
            
            # Update active neurons cache
            self.active_neurons_cache.update(spiking_indices)
            
            # Reset membrane potential of spiking neurons
            self.membrane_potentials[spiking_neurons] = 0
            
            # Create sparse representation of spike vector
            rows = spiking_indices
            data = np.ones(len(rows))
            spike_vector = sparse.csr_matrix(
                (data, (rows, np.zeros(len(rows)))),
                shape=(self.neuron_count, 1)
            )
            
            # Efficient sparse matrix multiplication
            # Only compute effect of spiking neurons on their postsynaptic targets
            if len(spiking_indices) > 0:
                # Extract columns of synaptic weights corresponding to spiking neurons
                post_weights = self.synaptic_weights[:, spiking_indices]
                # Sum across columns to get total input to each neuron
                delta_potentials = np.array(post_weights.sum(axis=1)).flatten()
                # Update membrane potentials
                self.membrane_potentials += delta_potentials
            
            # Apply decay
            self.membrane_potentials *= self.decay_rate
            
            # If learning module is available, apply STDP learning
            if self.learning and len(current_spikes) > 1:
                # Apply Hebbian learning to co-active neurons
                active_neurons = [idx for idx, _ in current_spikes]
                if len(active_neurons) >= 2:
                    self.learning.apply_hebbian_learning(active_neurons, strength=0.5)
            
            # Trace intermediate activation if available
            if trace_id and hasattr(self, 'thought_trace') and t % 2 == 0:  # Only trace every other step
                self.thought_trace.add_step(
                    trace_id,
                    "EnhancedSpikingCore",
                    f"simulation_step_{t}",
                    {
                        "active_neurons": len(spiking_indices),
                        "step": t,
                        "max_potential": float(self.membrane_potentials.max())
                    }
                )
        
        return spikes_over_time
    
    def apply_activation_pattern(self, pattern):
        """Apply an activation pattern directly to membrane potentials"""
        if len(pattern) == self.neuron_count:
            self.membrane_potentials = np.array(pattern)
            return True
        return False
    
    def get_current_activation(self):
        """Get the current network activation pattern"""
        return (self.membrane_potentials > 0.3).astype(float)
        
    def _detect_active_concepts(self, spikes_over_time):
        """Detect which concepts were activated based on spike patterns"""
        # Count spikes per neuron
        spike_counts = defaultdict(int)
        for time_step in spikes_over_time:
            for neuron_idx, _ in time_step:
                spike_counts[neuron_idx] += 1
                
        # Determine which concepts were active
        active_concepts = {}
        for concept, neurons in self.concept_mappings.items():
            # Calculate activation level for this concept
            concept_activity = sum(spike_counts.get(n, 0) for n in neurons) / max(len(neurons), 1)
            if concept_activity > 0.3:  # Threshold for concept activation
                active_concepts[concept] = concept_activity
                
        return active_concepts
    
    def abductive_reasoning(self, observation):
        """
        Generate hypotheses through abductive reasoning with IIT-enhanced processing.
        
        Args:
            observation: Input text to reason about
            
        Returns:
            List of hypotheses
        """
        print(f"[SNN] Enhanced abductive reasoning triggered for: '{observation}'")
        
        # First process input to get network into appropriate state
        result = self.process_input(observation)
        
        # Extract entities from observation using simple regex
        entities = self._extract_entities(observation)
        print(f"[SNN] Extracted entities: {entities}")
        
        # Get active concepts based on neural activation
        active_concepts = self._detect_active_concepts_from_regions()
        print(f"[SNN] Active concepts from regions: {active_concepts}")
        
        # Number of hypotheses based on integration level (phi)
        if self.phi < 0.3:
            # Low integration - fewer, more random hypotheses
            num_hypotheses = random.randint(1, 2)
        elif self.phi < 0.6:
            # Medium integration - moderate number of hypotheses
            num_hypotheses = random.randint(2, 4)
        else:
            # High integration - more hypotheses with better quality
            num_hypotheses = random.randint(3, 5)
        
        # Generate hypotheses
        hypotheses = []
            
        # If "no one" pattern is detected
        if "no_one" in entities or "nobody" in entities:
            # Generate hypotheses about distrusting everyone
            if "trust" in active_concepts:
                subject = entities.get("subject", "x")
                hypotheses.append(f"trusts_nobody({subject})")
                hypotheses.append(f"distrusts_everybody({subject})")
        else:
            # Generate standard relation hypotheses
            subject = entities.get("subject", "x")
            object_entity = entities.get("object", "y")
            
            # Sort concepts by activation level (descending)
            sorted_concepts = sorted(active_concepts.items(), key=lambda x: x[1], reverse=True)
            
            # Generate hypotheses for top concepts
            for concept, _ in sorted_concepts[:num_hypotheses]:
                if concept == "trust":
                    hypotheses.append(f"trusts({subject}, {object_entity})")
                elif concept == "distrust":
                    hypotheses.append(f"distrusts({subject}, {object_entity})")
                elif concept == "fear":
                    hypotheses.append(f"fears({subject}, {object_entity})")
                elif concept == "avoid":
                    hypotheses.append(f"avoids({subject}, {object_entity})")
                    
        # Add meta-cognitive hypothesis if high phi
        if self.phi > 0.6 and len(hypotheses) < num_hypotheses:
            if "subject" in entities:
                meta_subject = entities["subject"]
                hypotheses.append(f"understands({meta_subject}, self)")
            hypotheses.append(f"self_aware({subject})")
        
        # Store hypotheses in memory
        self.abductive_memory.extend(hypotheses)
        
        # If learning module is available, apply positive feedback to successful hypothesis generation
        if self.learning and hypotheses:
            # Find neurons active during hypothesis generation
            active_neurons = []
            for region_name in ['conceptual', 'metacognition', 'prediction']:
                if self.regions[region_name]['activation'] > 0.5:
                    active_neurons.extend(self.regions[region_name]['neurons'])
            
            if active_neurons:
                # Apply mild positive feedback
                feedback_strength = min(0.3, 0.1 * len(hypotheses))
                self.learning.process_feedback(feedback_strength, active_neurons)
        
        return hypotheses
    
    # Add to EnhancedSpikingCore class in enhanced_snn.py
    def process_symbolic_result(self, symbolic_facts):
        """Process symbolic facts to influence neural activations."""
        # Convert symbolic facts to neural activations
        for fact in symbolic_facts:
            subj = fact['subject']
            pred = fact['predicate']
            obj = fact['object']
            
            # Try to activate relevant concept assemblies
            self.activate_concept(subj, strength=0.7)
            self.activate_concept(pred, strength=0.7)
            self.activate_concept(obj, strength=0.7)
            
            # Run a brief simulation to spread activation
            self.propagate_activation(steps=2)
            
            # This should automatically update phi value
            
        return True
    
    # In the EnhancedSpikingCore class:
    def calculate_phi(self):
        """Calculate integration level (phi) between neural regions"""
        # Simple implementation based on activation patterns
        if not hasattr(self, 'region_activations'):
            return 0.0
            
        regions = list(self.region_activations.keys())
        if len(regions) < 2:
            return 0.0
            
        # Calculate mutual information between regions
        mutual_info_sum = 0
        connections = 0
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                region1 = regions[i]
                region2 = regions[j]
                if region1 in self.region_activations and region2 in self.region_activations:
                    # Calculate correlation between activations
                    act1 = self.region_activations[region1]
                    act2 = self.region_activations[region2]
                    correlation = np.corrcoef(act1, act2)[0, 1]
                    mutual_info_sum += abs(correlation)
                    connections += 1
                    
        # Return average mutual information (phi)
        return mutual_info_sum / max(1, connections)