"""
Neurally-implemented memory system that stores memories in the SNN's weights.
This implementation stores memories directly in the network's synaptic connections,
rather than in separate Python data structures.
"""
import numpy as np
import random
import time
import re
from collections import deque, defaultdict
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from models.snn.enhanced_snn import EnhancedSpikingCore
from models.snn.bidirectional_encoding import BidirectionalProcessor, create_processor
import logging

class MemorySNN(EnhancedSpikingCore):
    """
    SNN specialized for memory functions with distributed neural representations.
    
    This implementation creates memories as distributed patterns across neuronal assemblies
    represented by strengthened synaptic connections. The actual memory encoding happens
    through weight modifications (STDP, Hebbian learning), but the metadata about which
    neurons form each memory engram and their characteristic firing patterns are stored
    in the memory_engrams dictionary for efficient retrieval and management.
    
    The memory system uses a hybrid approach:
    - Bio-inspired: Pattern completion, attractor dynamics, engram formation
    - Supervised: SNNTorch networks predict recall success and content features
    - Error signals from supervised components guide custom plasticity rules
    
    Memory operations primarily modify synaptic weights to encode, strengthen, or 
    weaken memory traces, while bookkeeping structures track engram definitions.
    """
    
    def __init__(self, neuron_count=600, topology_type="small_world", vector_dim=300, bidirectional_processor=None):
        """
        Initialize the memory SNN with specialized parameters.
        
        Args:
            neuron_count: Total number of neurons in the network
            topology_type: Type of network topology ("small_world" is recommended for memory)
            vector_dim: Dimension of the vector space for token embeddings
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
        """
        # Initialize set of neurons for custom learning before parent init
        self.custom_learning_neurons = set()
        
        # Flag to control parent class plasticity
        self.disable_parent_plasticity = False
        
        # Initialize training-related attributes
        self.training_mode = False
        self.training_iterations = 0
        self.neuron_biases = np.zeros(neuron_count)
        
        # Call parent initializer with memory-optimized parameters
        super().__init__(
            neuron_count=neuron_count, 
            topology_type=topology_type,
            model_type="memory",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )
        
        # Override with memory-optimized parameters
        self.connection_density = 0.15   # Medium density for memory formations
        self.region_density = 0.6        # Higher internal connectivity
        self.decay_rate = 0.97           # Slower decay for persistent memories
        
        # Memory parameters
        self.recency_window = 20          # Number of recent inputs to maintain
        self.recent_inputs = deque(maxlen=self.recency_window)
        self.consolidation_threshold = 3  # Repetitions needed for long-term storage
        
        # Memory encoding parameters
        self.encoding_strength = 0.8      # Strength of initial memory encoding
        self.association_threshold = 0.6  # Threshold for forming associations
        self.retrieval_threshold = 0.4    # Minimum activation for retrieval
        
        # Pattern completion parameters
        self.completion_threshold = 0.3   # Threshold for pattern completion
        self.completion_iterations = 5    # Iterations for pattern completion
        
        # Context tracking
        self.context_window = 5           # Size of temporal context window
        self.context_buffer = deque(maxlen=self.context_window)
        
        # Memory performance tracking
        self.memory_stats = {
            'encoding_count': 0,          # Number of encoded memories
            'retrieval_count': 0,         # Number of memory retrievals
            'consolidation_count': 0,     # Number of consolidated memories
            'forgetting_count': 0,        # Number of forgotten memories
            'association_count': 0        # Number of formed associations
        }
        self.recall_performance = []      # Track recall success rate
        self.encoding_performance = []    # Track encoding success rate
        
        # Memory identifier mapping (to keep track of what's stored)
        # Maps content keys to engram neuron assemblies that represent them
        # along with activation pattern information
        self.memory_engrams = {}          # Maps content_key -> memory engram info
        
        # Specialized memory structures
        self._init_memory_structures()
        
        # Set up SNNTorch components for supervised learning
        self._setup_snntorch_components()
        
        # Register specialized memory synapses for hybrid learning
        self._register_memory_synapses()
    
    def _init_topology(self):
        """Override to create memory-specific regions"""
        # Call parent method to initialize basic topology
        super()._init_topology()
        
        # Ensure memory-specific regions exist
        memory_regions = {
            'working_memory': 0.25,   # Working memory
            'episodic': 0.25,        # Episodic memory 
            'semantic': 0.25,        # Semantic memory
            'hippocampal': 0.15,     # Memory binding/formation region
            'context': 0.10          # Context binding region
        }
        
        # Create or resize memory regions
        self._create_memory_regions(memory_regions)
    
    def _create_memory_regions(self, memory_regions):
        """
        Create or resize memory-specific regions
        
        Args:
            memory_regions: Dictionary mapping region names to proportion of neurons
        """
        # Calculate neurons per region
        neurons_per_region = {}
        for region_name, proportion in memory_regions.items():
            neurons_per_region[region_name] = int(self.neuron_count * proportion)
            
        # Get already allocated neurons
        allocated_neurons = set()
        for region_name, region in self.regions.items():
            if region_name not in memory_regions:
                allocated_neurons.update(region['neurons'])
                
        # Available neurons
        available_neurons = list(set(range(self.neuron_count)) - allocated_neurons)
        
        # Create or update memory regions
        start_idx = 0
        for region_name, size in neurons_per_region.items():
            if region_name in self.regions:
                # Region exists - resize if needed
                current_size = len(self.regions[region_name]['neurons'])
                if current_size < size:
                    # Expand region
                    additional_needed = size - current_size
                    additional_neurons = available_neurons[start_idx:start_idx + additional_needed]
                    self.regions[region_name]['neurons'].extend(additional_neurons)
                    start_idx += additional_needed
            else:
                # Create new region
                if start_idx + size <= len(available_neurons):
                    region_neurons = available_neurons[start_idx:start_idx + size]
                    self.regions[region_name] = {
                        'neurons': region_neurons,
                        'activation': 0.0,
                        'recurrent': True,
                        'plasticity_factor': 1.2 if region_name in ['hippocampal', 'episodic'] else 1.0
                    }
                    start_idx += size
                else:
                    # Not enough neurons available
                    remaining = len(available_neurons) - start_idx
                    if remaining > 0:
                        region_neurons = available_neurons[start_idx:]
                        self.regions[region_name] = {
                            'neurons': region_neurons,
                            'activation': 0.0,
                            'recurrent': True,
                            'plasticity_factor': 1.2 if region_name in ['hippocampal', 'episodic'] else 1.0
                        }
                    print(f"[MemorySNN] Warning: Not enough neurons for {region_name} region")
        
        # Define connectivity between memory regions
        self._define_memory_connectivity()
    
    def _define_memory_connectivity(self):
        """Define connectivity patterns between memory regions"""
        # Ensure region_connectivity exists
        if not hasattr(self, 'region_connectivity'):
            self.region_connectivity = {}
            
        # Define memory pathways
        memory_connectivity = {
            'working_memory': ['hippocampal', 'episodic', 'semantic', 'context'],
            'hippocampal': ['working_memory', 'episodic', 'semantic'],
            'episodic': ['working_memory', 'hippocampal', 'semantic', 'context'],
            'semantic': ['working_memory', 'hippocampal', 'episodic'],
            'context': ['working_memory', 'episodic']
        }
        
        # Add to existing connectivity
        for source, targets in memory_connectivity.items():
            if source in self.regions:
                if source not in self.region_connectivity:
                    self.region_connectivity[source] = []
                
                # Add connections if target exists
                for target in targets:
                    if target in self.regions and target not in self.region_connectivity[source]:
                        self.region_connectivity[source].append(target)
        
        # Connect memory regions to other key regions
        for memory_region in ['working_memory', 'episodic', 'semantic']:
            if memory_region in self.regions:
                # Connect to cognition regions if they exist
                for region in ['higher_cognition', 'metacognition', 'sensory']:
                    if region in self.regions:
                        # Add bidirectional connections
                        if memory_region not in self.region_connectivity:
                            self.region_connectivity[memory_region] = []
                        if region not in self.region_connectivity[memory_region]:
                            self.region_connectivity[memory_region].append(region)
                            
                        if region not in self.region_connectivity:
                            self.region_connectivity[region] = []
                        if memory_region not in self.region_connectivity[region]:
                            self.region_connectivity[region].append(memory_region)
    
    def _init_memory_structures(self):
        """Initialize specialized memory structures"""
        # Attractor dynamics for pattern completion
        self.attractor_states = {}
        
        # Pattern memory
        self.pattern_memory = []
        
        # Association structures
        self.association_matrix = sparse.lil_matrix((self.neuron_count, self.neuron_count))
        
        # Memory context mapping
        self.context_mapping = {}  # Maps context tags to neuron sets
        
        # Memory modulation factors
        self.modulation_factors = {
            'encoding': 1.0,    # Modulates encoding strength
            'retrieval': 1.0,   # Modulates retrieval threshold
            'consolidation': 1.0 # Modulates consolidation threshold
        }
        
        print(f"[MemorySNN] Initialized memory structures")
    
    def _setup_snntorch_components(self):
        """
        Set up SNNTorch components for supervised learning.
        
        The supervised components predict two key aspects of memory:
        1. Recall prediction: Binary probability that the target memory will be 
        successfully retrieved given the current network state.
        2. Content prediction: A 50-dimensional feature vector representing the
        semantic content of the target memory.
        
        The errors from these predictions guide custom plasticity rules in the
        bio-inspired components.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import snntorch as snn
            
            self.torch_available = True
            
            # Beta is the decay rate for the neuron's spiking trace
            beta = 0.95
            
            # Use surrogate gradient for backpropagation through spikes
            spike_grad = surrogate.fast_sigmoid(slope=25)
            
            # Get size of memory features to use as input size for supervised networks
            memory_features_size = 100  # Fixed size for stability
            
            # 1. Recall predictor network
            # Predicts: P(target_memory ∈ retrieved_memories | network_state)
            # Output: Single value [0,1] - probability of successful target retrieval
            self.recall_network = nn.Sequential(
                nn.Linear(memory_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, 1),  # Binary classification: recalled vs not recalled
                nn.Sigmoid()       # Output probability of successful recall
            )
            
            # 2. Content prediction network
            # Predicts: f(memory_content) where f maps content to fixed-size feature vector
            # Output: 50-dimensional vector representing semantic features of target memory
            content_dim = 50  # Dimension of content feature vector
            self.content_network = nn.Sequential(
                nn.Linear(memory_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, content_dim)  # Content feature prediction
            )
            
            # Define loss functions
            # Recall loss: Binary cross-entropy for retrieval success prediction
            self.recall_loss_fn = nn.BCELoss()
            
            # Content loss: MSE between predicted and actual content features
            self.content_loss_fn = nn.MSELoss()
            
            # Define optimizers
            self.recall_optimizer = optim.Adam(self.recall_network.parameters(), lr=0.01)
            self.content_optimizer = optim.Adam(self.content_network.parameters(), lr=0.01)
            
            # Initialize neuron biases for supervised guidance
            self.neuron_biases = np.zeros(self.neuron_count)
            
            print("[MemorySNN] SNNTorch components initialized successfully")
            
        except ImportError as e:
            print(f"[MemorySNN] Warning: SNNTorch or PyTorch not available - {e}")
            self.torch_available = False
    
    def _register_memory_synapses(self):
        """
        Register specialized synapses for memory operations.
        This ensures the parent's plasticity won't interfere with our custom learning.
        """
        synapse_count = 0
        
        # 1. Register synapses between memory regions
        memory_regions = ['working_memory', 'episodic', 'semantic', 'hippocampal']
        for src_region in memory_regions:
            if src_region not in self.regions:
                continue
                
            src_neurons = self.regions[src_region]['neurons']
            
            for dst_region in memory_regions:
                if dst_region != src_region and dst_region in self.regions:
                    dst_neurons = self.regions[dst_region]['neurons']
                    
                    # Register a subset of connections to avoid excessive registration
                    # Number of synapses to register for each region pair
                    num_synapses = min(len(src_neurons), len(dst_neurons), 100)
                    
                    for _ in range(num_synapses):
                        pre = random.choice(src_neurons)
                        post = random.choice(dst_neurons)
                        self.register_specialized_synapse(pre, post)
                        synapse_count += 1
        
        # 2. Register synapses within existing memory engrams
        for content_key, engram in self.memory_engrams.items():
            neurons = engram.get('neurons', [])
            if len(neurons) > 1:
                # Register connections within the engram (up to 200 per engram)
                max_connections = min(200, len(neurons) * (len(neurons) - 1) // 2)
                for _ in range(max_connections):
                    pre, post = random.sample(neurons, 2)
                    self.register_specialized_synapse(pre, post)
                    synapse_count += 1
        
        # 3. Register hippocampal connections to other memory regions
        if 'hippocampal' in self.regions:
            hippocampal_neurons = self.regions['hippocampal']['neurons']
            for region_name in ['working_memory', 'episodic', 'semantic']:
                if region_name in self.regions:
                    target_neurons = self.regions[region_name]['neurons']
                    # Register a subset of connections (up to 100 per region pair)
                    max_connections = min(100, len(hippocampal_neurons), len(target_neurons))
                    for _ in range(max_connections):
                        pre = random.choice(hippocampal_neurons)
                        post = random.choice(target_neurons)
                        self.register_specialized_synapse(pre, post)
                        self.register_specialized_synapse(post, pre)  # Register both directions
                        synapse_count += 2
        
        # 4. Register connections between sensory and memory regions
        if 'sensory' in self.regions:
            sensory_neurons = self.regions['sensory']['neurons']
            for region_name in ['working_memory', 'episodic']:
                if region_name in self.regions:
                    target_neurons = self.regions[region_name]['neurons']
                    # Register connections from sensory to memory regions
                    max_connections = min(150, len(sensory_neurons), len(target_neurons))
                    for _ in range(max_connections):
                        pre = random.choice(sensory_neurons)
                        post = random.choice(target_neurons)
                        self.register_specialized_synapse(pre, post)
                        synapse_count += 1
        
        print(f"[MemorySNN] Registered {synapse_count} synapses for specialized memory learning")
        return synapse_count
    
    def simulate_spiking(self, input_activation, timesteps=10):
        """
        Override to control parent plasticity during training.
        
        Args:
            input_activation: Initial activation pattern
            timesteps: Number of simulation steps
            
        Returns:
            List of spike events per timestep
        """
        # Save original learning state
        original_learning_enabled = True
        
        # If parent has learning_module, check and disable it temporarily
        if hasattr(self, 'learning_module') and self.disable_parent_plasticity:
            original_stdp_enabled = self.learning_module.get('stdp', {}).get('enabled', True)
            original_hebbian_enabled = self.learning_module.get('hebbian', {}).get('enabled', True)
            original_homeostatic_enabled = self.learning_module.get('homeostatic', {}).get('enabled', True)
            
            # Disable parent plasticity
            self.learning_module['stdp']['enabled'] = False
            self.learning_module['hebbian']['enabled'] = False
            self.learning_module['homeostatic']['enabled'] = False
            original_learning_enabled = False
        
        # Call parent's simulate_spiking
        spike_patterns = super().simulate_spiking(input_activation, timesteps)
        
        # Store spike patterns for later use
        self.last_spike_patterns = spike_patterns
        
        # Restore original learning state
        if hasattr(self, 'learning_module') and not original_learning_enabled:
            self.learning_module['stdp']['enabled'] = original_stdp_enabled
            self.learning_module['hebbian']['enabled'] = original_hebbian_enabled
            self.learning_module['homeostatic']['enabled'] = original_homeostatic_enabled
        
        return spike_patterns
    
    def process_input(self, input_activation):
        """
        Process input through the SNN with custom learning approach.
        
        Args:
            input_activation: Neural activation pattern
            
        Returns:
            Processing results
        """
        # If we have neuron biases, apply them to the input activation
        if hasattr(self, 'neuron_biases') and np.any(self.neuron_biases):
            # Create a copy to avoid modifying the original
            biased_activation = input_activation.copy()
            
            # Add biases to input activation
            biased_activation += self.neuron_biases
            
            # Ensure values stay in reasonable range
            biased_activation = np.clip(biased_activation, 0, 1)
            
            # Run standard processing
            result = super().process_input(biased_activation)
            
            # Clear biases after use (they're only for this simulation)
            self.neuron_biases = np.zeros(self.neuron_count)
        else:
            # Run standard processing
            result = super().process_input(input_activation)
        
        # Store spike patterns for later feature extraction
        if 'spike_patterns' not in result and hasattr(self, 'last_spike_patterns'):
            result['spike_patterns'] = self.last_spike_patterns
            
        return result
    
    def encode_memory(self, activation_pattern, content=None, memory_type='working', importance=0.5, context=None):
        """
        Encode content into memory by forming neural assembly and strengthening connections.
        Memory is stored directly in the neural network weights using STDP and Hebbian learning.
        
        Args:
            activation_pattern: Neural activation pattern to store
            content: Optional metadata about the content (for tracking only)
            memory_type: Type of memory to encode ('working', 'episodic', 'semantic')
            importance: Importance of this memory (0-1)
            context: Optional context information for episodic memory
            
        Returns:
            Dict with encoding results
        """
        # Track encoding attempt
        self.memory_stats['encoding_count'] += 1
        
        # Generate a content_key for tracking this memory
        content_key = f"mem_{hash(str(activation_pattern.tobytes())) % 10000:04d}"
        
        # Store for potential consolidation
        self.recent_inputs.append({
            'content_key': content_key,
            'content': content,
            'activation': activation_pattern.copy(),
            'timestamp': time.time(),
            'importance': importance,
            'context': context
        })
        
        # First simulation: Present activation pattern to network to create initial activations
        self.disable_parent_plasticity = True  # Disable parent plasticity during encoding
        result = self.process_input(activation_pattern)
        spike_patterns = result.get('spike_patterns', [])
        if not spike_patterns and hasattr(self, 'last_spike_patterns'):
            spike_patterns = self.last_spike_patterns
        self.disable_parent_plasticity = False
        
        # Capture the actual firing pattern that occurred during encoding
        memory_pattern = {
            'neurons': list(self.active_neurons_cache),
            'spike_times': {}, # Store when each neuron fired
            'spike_sequence': [], # Store sequence of neuron firing
            'membrane_state': self.membrane_potentials.copy() # Store membrane potentials
        }
        
        # Extract temporal firing pattern from spike_patterns
        for t, spikes in enumerate(spike_patterns):
            for neuron, strength in spikes:
                if neuron not in memory_pattern['spike_times']:
                    memory_pattern['spike_times'][neuron] = []
                memory_pattern['spike_times'][neuron].append((t, strength))
                memory_pattern['spike_sequence'].append((t, neuron, strength))
        
        # Sort spike sequence by time
        memory_pattern['spike_sequence'].sort(key=lambda x: x[0])
        
        # Add context neurons if context provided
        context_neurons = set()
        if context:
            # Activate context region based on context info
            context_neurons = self._activate_context_neurons(context)
            # Include context neurons in the pattern
            memory_pattern['neurons'].extend(list(context_neurons))
            memory_pattern['context_neurons'] = list(context_neurons)
        
        # Encode memory directly into the network based on memory type
        result = {}
        region_name = self._get_region_for_memory_type(memory_type)
        
        # Select neurons from the target memory region to form the engram
        memory_neurons = self._select_memory_neurons(region_name, self.active_neurons_cache)
        
        if memory_neurons:
            # Store memory engram information for later retrieval
            self.memory_engrams[content_key] = {
                'neurons': list(memory_neurons),
                'memory_pattern': memory_pattern,
                'type': memory_type,
                'encoding_time': time.time(),
                'last_accessed': time.time(),
                'importance': importance,
                'content': content,  # Keep minimal content for verification/debug
                'access_count': 1,
                'context_tag': self._get_context_key(context) if context else None
            }
            
            # Apply STDP and Hebbian learning to encode the specific firing pattern
            self._encode_specific_pattern(memory_neurons, memory_pattern, importance)
            
            # Connect to context neurons if available
            if context and context_neurons:
                self._connect_engram_to_context(memory_neurons, context_neurons, importance)
            
            # If working memory, create connections to hippocampal region for potential consolidation
            if memory_type == 'working' and 'hippocampal' in self.regions:
                hippocampal_neurons = random.sample(
                    self.regions['hippocampal']['neurons'],
                    min(len(memory_neurons), len(self.regions['hippocampal']['neurons']))
                )
                self._connect_regions(memory_neurons, hippocampal_neurons, strength=0.3)
            
            # Apply reinforcement learning to increase overall network importance
            self.apply_reinforcement(importance * 0.3)
            
            # Update memory statistics
            result = {
                'success': True,
                'memory_type': memory_type,
                'content_key': content_key,
                'engram_size': len(memory_neurons),
                'encoding_strength': importance * self.encoding_strength
            }
            
            # Track performance
            self.encoding_performance.append(1)
        else:
            # Encoding failed
            result = {
                'success': False,
                'memory_type': memory_type,
                'content_key': content_key,
                'error': 'Failed to select memory neurons for engram'
            }
            # Track performance
            self.encoding_performance.append(0)
        
        # Cap performance history
        if len(self.encoding_performance) > 100:
            self.encoding_performance.pop(0)
        
        # Add activation to context buffer for temporal context
        self.context_buffer.append({
            'content_key': content_key,
            'activation': activation_pattern.copy(),
            'context': context,
            'timestamp': time.time()
        })
        
        # Update temporal connections between recent memories
        self._update_temporal_connections()
        
        return result
    
    def _encode_specific_pattern(self, engram_neurons, memory_pattern, importance=0.5):
        """
        Encode a specific firing pattern into the network's weights using enhanced
        STDP and Hebbian learning with interference reduction mechanisms.
        
        Args:
            engram_neurons: Set of neurons forming the engram
            memory_pattern: Temporal pattern information
            importance: Importance of the memory (0-1)
        """
        # First check for overlapping engrams to handle interference
        overlapping_engrams = self._find_overlapping_engrams(engram_neurons, threshold=0.4)
        
        # Apply interference reduction if significant overlap exists
        if overlapping_engrams:
            self._apply_interference_reduction(engram_neurons, overlapping_engrams)
        
        # Apply Hebbian learning to strengthen connections within the engram
        # This creates a stable neural assembly
        self._strengthen_engram_connections(engram_neurons, importance * self.encoding_strength)
        
        # Use enhanced STDP to encode the specific temporal firing pattern
        spike_sequence = memory_pattern['spike_sequence']
        
        # Apply multi-stage STDP that captures both direct and higher-order correlations
        # Stage 1: Direct sequential correlations
        for i, (t1, pre, _) in enumerate(spike_sequence):
            # Look at future spikes to find neurons that fired after this one
            for j in range(i+1, min(i+20, len(spike_sequence))):
                t2, post, _ = spike_sequence[j]
                
                # Skip if same neuron or time difference too large
                if pre == post or (t2 - t1) > 10:
                    continue
                
                # Calculate time difference
                time_diff = t2 - t1
                
                # If pre neuron fired before post neuron, strengthen connection from pre to post
                if time_diff > 0:
                    # Apply STDP learning rule with precision-based modulation
                    # Shorter time differences create stronger connections (exponential decay)
                    weight_change = importance * 0.2 * np.exp(-time_diff / 5.0)
                    
                    # Apply precision scaling - higher importance memories get more precise encoding
                    precision_factor = 0.8 + (importance * 0.4)
                    weight_change *= precision_factor
                    
                    # Update the weight
                    current_weight = self.synaptic_weights[pre, post]
                    self.synaptic_weights[pre, post] = min(1.0, current_weight + weight_change)
        
        # Stage 2: Create sparse lateral inhibition between engram neurons to sharpen the representation
        self._create_lateral_inhibition(engram_neurons, importance=importance)
        
        # Stage 3: Create second-order correlations to capture sequence chains
        # This helps with pattern completion by creating "prediction" pathways
        self._encode_sequence_chains(spike_sequence, importance)
    
    def _create_lateral_inhibition(self, engram_neurons, importance=0.5):
        """
        Create sparse lateral inhibition within the engram to sharpen the representation.
        Neurons that are part of a sequence inhibit neurons that aren't directly connected
        to them in the sequence, creating a more precise memory trace.
        
        Args:
            engram_neurons: Neurons in the engram
            importance: Importance factor for inhibition strength
        """
        # Sample a subset of neuron pairs to add lateral inhibition
        neurons = list(engram_neurons)
        
        # Number of inhibitory connections to create (sparse)
        num_connections = min(100, len(neurons) * 3)
        
        # Create random lateral inhibition
        for _ in range(num_connections):
            if len(neurons) < 2:
                break
                
            # Select two random neurons
            pre, post = random.sample(neurons, 2)
            
            # Skip if already strongly connected (part of the sequence)
            if self.synaptic_weights[pre, post] > 0.4:
                continue
            
            # Add inhibitory connection
            inhibition_strength = -0.2 * importance
            self.synaptic_weights[pre, post] = inhibition_strength
    
    def _encode_sequence_chains(self, spike_sequence, importance=0.5):
        """
        Encode higher-order sequence chains to improve pattern completion.
        This creates connections between neurons that fire in sequence patterns,
        not just direct sequential pairs.
        
        Args:
            spike_sequence: Sequence of (time, neuron, strength) tuples
            importance: Importance factor
        """
        if len(spike_sequence) < 3:
            return
            
        # Group by time bin to find neurons that fire close together
        time_bins = defaultdict(list)
        for t, neuron, strength in spike_sequence:
            # Discretize time into 3-step bins
            bin_idx = t // 3
            time_bins[bin_idx].append((neuron, strength))
        
        # Create connections between neurons in consecutive time bins
        bin_indices = sorted(time_bins.keys())
        for i in range(len(bin_indices) - 1):
            current_bin = time_bins[bin_indices[i]]
            next_bin = time_bins[bin_indices[i + 1]]
            
            # Connect neurons in current bin to neurons in next bin
            for pre_neuron, pre_strength in current_bin:
                for post_neuron, post_strength in next_bin:
                    if pre_neuron != post_neuron:
                        # Calculate weight change based on firing strengths
                        weight_change = importance * 0.15 * pre_strength * post_strength
                        current_weight = self.synaptic_weights[pre_neuron, post_neuron]
                        self.synaptic_weights[pre_neuron, post_neuron] = min(1.0, current_weight + weight_change)
    
    def _find_overlapping_engrams(self, neuron_set, threshold=0.4):
        """
        Find existing memory engrams that significantly overlap with the given neuron set.
        
        Args:
            neuron_set: Set of neurons to check for overlap
            threshold: Minimum overlap ratio to consider significant
            
        Returns:
            List of (content_key, overlap_ratio, existing_neurons) tuples for overlapping engrams
        """
        overlapping = []
        neuron_set = set(neuron_set)  # Ensure it's a set for intersection operations
        
        # Check overlap with each existing memory engram
        for content_key, memory in self.memory_engrams.items():
            existing_neurons = set(memory['neurons'])
            
            # Calculate overlap ratio
            overlap = len(neuron_set.intersection(existing_neurons))
            if len(existing_neurons) > 0:
                overlap_ratio = overlap / min(len(neuron_set), len(existing_neurons))
                
                # Consider significant overlap
                if overlap_ratio >= threshold:
                    overlapping.append((content_key, overlap_ratio, existing_neurons))
        
        return overlapping
    
    def _apply_interference_reduction(self, engram_neurons, overlapping_engrams, strength=-0.2):
        """
        Apply plasticity to reduce interference between the new engram and existing overlapping engrams.
        This helps create more distinct memory representations.
        
        Args:
            engram_neurons: New engram neurons
            overlapping_engrams: List of overlapping existing engrams
            strength: Strength of inhibitory connections to add
        """
        print(f"[MemorySNN] Reducing interference across {len(overlapping_engrams)} overlapping engrams")
        
        # Create mild lateral inhibition between new engram and overlapping engrams
        for content_key, overlap_ratio, existing_neurons in overlapping_engrams:
            # The higher the overlap, the more we need to differentiate
            inhibition_strength = strength * overlap_ratio  # Stronger inhibition for higher overlap
            
            # Find the overlapping neurons
            overlap = set(engram_neurons).intersection(existing_neurons)
            
            # Find non-overlapping neurons in both engrams
            new_only = set(engram_neurons) - overlap
            existing_only = existing_neurons - overlap
            
            # Create weak inhibitory connections between non-overlapping parts of each engram
            # This helps the network distinguish between similar but different patterns
            for pre in random.sample(list(new_only), min(10, len(new_only))):
                for post in random.sample(list(existing_only), min(10, len(existing_only))):
                    # Add inhibitory connection (negative weight)
                    current_weight = self.synaptic_weights[pre, post]
                    # Only modify if it would make the connection more inhibitory
                    if current_weight > inhibition_strength:
                        self.synaptic_weights[pre, post] = inhibition_strength
                    
                    # Bidirectional inhibition
                    current_weight = self.synaptic_weights[post, pre]
                    if current_weight > inhibition_strength:
                        self.synaptic_weights[post, pre] = inhibition_strength

    def _strengthen_engram_connections(self, neurons, strength=0.5):
        """
        Strengthen connections between neurons in an engram using Hebbian plasticity.
        This creates a stable neural assembly for the memory.
        
        Args:
            neurons: Set of neurons forming the engram
            strength: Strength of connections to form (0-1)
            
        Returns:
            Number of connections modified
        """
        neurons = list(neurons)  # Convert to list for indexing
        connections_modified = 0
        
        # Apply Hebbian learning between all neuron pairs in the engram
        for i in range(len(neurons)):
            for j in range(i+1, len(neurons)):  # Only upper triangle (no self-connections)
                pre = neurons[i]
                post = neurons[j]
                
                # Get current weight
                current_weight = self.synaptic_weights[pre, post]
                
                # Calculate weight increase (stronger for more important memories)
                weight_increase = strength * 0.2 * (1.0 - current_weight)
                
                # Update weights bidirectionally (symmetric connections for stable patterns)
                new_weight = min(1.0, current_weight + weight_increase)
                
                # Only count if there was a significant change
                if abs(new_weight - current_weight) > 0.001:
                    self.synaptic_weights[pre, post] = new_weight
                    self.synaptic_weights[post, pre] = new_weight  # Symmetric connection
                    connections_modified += 2
        
        return connections_modified

    def retrieve_memory(self, query_activation, memory_type=None, max_results=10, context=None):
        """
        Retrieve memories from the network based on query activation pattern.
        Memory retrieval works through pattern completion and activation dynamics.
        
        Args:
            query_activation: Query activation pattern
            memory_type: Type of memory to search ('working', 'episodic', 'semantic', or None for all)
            max_results: Maximum number of results to return
            context: Optional context to help guide retrieval
            
        Returns:
            List of retrieved memories with relevance scores
        """
        # Track retrieval attempt
        self.memory_stats['retrieval_count'] += 1
        
        # Activate context neurons if context provided
        context_neurons = set()
        if context:
            context_neurons = self._activate_context_neurons(context)
            
            # Add context activation to the query
            for neuron in context_neurons:
                query_activation[neuron] = max(query_activation[neuron], 0.7)
        
        # Run the query through the network for multiple iterations to allow pattern completion
        # This is where retrieval actually happens through network dynamics
        retrieved_memories = []
        
        # Determine regions to search based on memory_type
        target_regions = self._get_target_regions(memory_type)
        
        # PATTERN COMPLETION: Process query through network with extended iterations
        # to allow for pattern completion and attractor dynamics
        # This is the key process where the network reactivates stored memory patterns
        spike_patterns = self.simulate_spiking(query_activation, timesteps=self.completion_iterations)
        
        # Get the network state after processing
        # This state reflects the retrieved memory patterns through attractor dynamics
        network_state = self._get_network_state(spike_patterns=spike_patterns)
        
        # Identify the retrieved memory engrams by comparing the network's current activity pattern
        # to the patterns stored during encoding
        results = self._identify_activated_memories(network_state, target_regions, spike_patterns)
        
        # If we found results, format them
        if results:
            # Add to retrieved_memories list
            retrieved_memories.extend(results)
            
            # Update memory engrams - retrieval strengthens memory
            for result in results:
                # Update last access time
                if result['content_key'] in self.memory_engrams:
                    memory = self.memory_engrams[result['content_key']]
                    memory['last_accessed'] = time.time()
                    memory['access_count'] += 1
                    
                    # Strengthen the engram using retrieved pattern
                    # This is memory reconsolidation - the memory is strengthened each time it's recalled
                    self._strengthen_memory_after_retrieval(
                        memory,
                        network_state,
                        spike_patterns,
                        strength=0.3 * result['relevance']
                    )
            
            # Apply reinforcement learning to strengthen retrieval paths
            self.apply_reinforcement(0.2)
            self.recall_performance.append(1)
        else:
            # No results found
            self.recall_performance.append(0)
        
        # Cap performance history
        if len(self.recall_performance) > 100:
            self.recall_performance.pop(0)
            
        # Sort by relevance
        retrieved_memories.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Limit results
        results = retrieved_memories[:max_results]
            
        return results
    
    def _strengthen_memory_after_retrieval(self, memory, network_state, spike_patterns, strength=0.3):
        """
        Strengthen a memory after successful retrieval (memory reconsolidation).
        
        Args:
            memory: Memory engram data
            network_state: Current network state after retrieval
            spike_patterns: Spike patterns from retrieval
            strength: Strength of reconsolidation
        """
        # Get neurons in the engram
        engram_neurons = memory['neurons']
        
        # Strengthen connections within the engram
        self._strengthen_engram_connections(engram_neurons, strength)
        
        # Enhanced reconsolidation: update the temporal pattern based on retrieval
        # This adapts the memory to the current retrieval pattern, which is a core
        # feature of biological memory reconsolidation
        retrieved_pattern = {}
        retrieved_pattern['spike_sequence'] = []
        
        # Extract temporal pattern from retrieval
        for t, spikes in enumerate(spike_patterns):
            for neuron, spike_strength in spikes:
                if neuron in engram_neurons:
                    retrieved_pattern['spike_sequence'].append((t, neuron, spike_strength))
        
        # Sort by time
        retrieved_pattern['spike_sequence'].sort(key=lambda x: x[0])
        
        # STDP-based reconsolidation: strengthen connections based on the retrieval pattern
        if retrieved_pattern['spike_sequence']:
            for i, (t1, pre, _) in enumerate(retrieved_pattern['spike_sequence']):
                # Look at future spikes to find neurons that fired after this one
                for j in range(i+1, min(i+10, len(retrieved_pattern['spike_sequence']))):
                    t2, post, _ = retrieved_pattern['spike_sequence'][j]
                    
                    # Skip if same neuron or time difference too large
                    if pre == post or (t2 - t1) > 5:
                        continue
                    
                    # Calculate time difference
                    time_diff = t2 - t1
                    
                    # Apply STDP to strengthen connections in the direction of activation flow
                    weight_change = strength * 0.2 * np.exp(-time_diff / 3.0)
                    current_weight = self.synaptic_weights[pre, post]
                    self.synaptic_weights[pre, post] = min(1.0, current_weight + weight_change)
            
            # Integrate this retrieval pattern with the original pattern (memory updating)
            # This is how memories evolve and update with each retrieval - a core feature of
            # biological memory systems
            if 'memory_pattern' in memory and 'spike_sequence' in memory['memory_pattern']:
                original_sequence = memory['memory_pattern']['spike_sequence']
                # Update memory pattern by integrating retrieval pattern (30% influence)
                # This gradually updates the memory through repeated retrievals
                updated_sequence = original_sequence + retrieved_pattern['spike_sequence']
                # Keep only the most recent 100 spikes to avoid excessive storage
                if len(updated_sequence) > 100:
                    updated_sequence = updated_sequence[-100:]
                memory['memory_pattern']['spike_sequence'] = updated_sequence
    
    def _get_network_state(self, spike_patterns=None):
        """
        Get the current network state after processing.
        
        Args:
            spike_patterns: Optional spike patterns from simulation
            
        Returns:
            Dict with network state information
        """
        # Get membrane potentials
        membrane_potentials = self.membrane_potentials.copy()
        
        # Get active neurons
        if hasattr(self, 'active_neurons_cache'):
            active_neurons = self.active_neurons_cache
        else:
            threshold = 0.3  # Lower threshold for retrieval than encoding
            active_neurons = set(np.where(membrane_potentials > threshold)[0])
        
        # If spike patterns provided, extract temporal information
        temporal_info = {}
        if spike_patterns:
            spike_times = {}
            spike_sequence = []
            
            for t, spikes in enumerate(spike_patterns):
                for neuron, strength in spikes:
                    if neuron not in spike_times:
                        spike_times[neuron] = []
                    spike_times[neuron].append((t, strength))
                    spike_sequence.append((t, neuron, strength))
            
            spike_sequence.sort(key=lambda x: x[0])
            
            temporal_info = {
                'spike_times': spike_times,
                'spike_sequence': spike_sequence
            }
        
        # Get region activations
        region_activations = {}
        for region_name, region in self.regions.items():
            region_neurons = set(region['neurons'])
            
            # Calculate region activation as proportion of active neurons
            if region_neurons:
                active_in_region = region_neurons.intersection(active_neurons)
                activation = len(active_in_region) / len(region_neurons)
                
                # Also factor in average membrane potential
                if region_neurons:
                    avg_potential = np.mean(membrane_potentials[list(region_neurons)])
                    activation = 0.7 * activation + 0.3 * avg_potential
                    
                region_activations[region_name] = activation
            else:
                region_activations[region_name] = 0.0
        
        return {
            'membrane_potentials': membrane_potentials,
            'active_neurons': active_neurons,
            'region_activations': region_activations,
            'phi': self.phi,  # Integration metrics
            'integration': self.integration,
            'differentiation': self.differentiation,
            'temporal_info': temporal_info
        }
        
    def _identify_activated_memories(self, network_state, target_regions, spike_patterns=None):
        """
        Identify which memory engrams have been activated by the query.
        Compare current neural activity patterns with stored memory patterns.
        
        Args:
            network_state: Current network state
            target_regions: Regions to search for memories
            spike_patterns: Optional spike patterns for temporal comparison
            
        Returns:
            List of retrieved memories with relevance scores
        """
        results = []
        active_neurons = network_state['active_neurons']
        temporal_info = network_state.get('temporal_info', {})
        
        # Need some active neurons to identify memories
        if not active_neurons:
            return results
            
        # Check each memory engram for activation
        for content_key, memory in self.memory_engrams.items():
            # Skip if not in target region
            if memory['type'] not in target_regions and memory['type'] not in target_regions:
                continue
                
            # Calculate activation overlap - spatial pattern similarity
            memory_neurons = set(memory['neurons'])
            overlap = memory_neurons.intersection(active_neurons)
            
            # Calculate overlap ratio
            overlap_ratio = len(overlap) / len(memory_neurons) if memory_neurons else 0
            
            # Calculate average membrane potential for memory neurons - magnitude of activation
            memory_potentials = network_state['membrane_potentials'][list(memory_neurons)]
            avg_potential = np.mean(memory_potentials) if len(memory_neurons) > 0 else 0
            
            # Calculate temporal pattern similarity if spike patterns available
            temporal_similarity = 0.0
            if 'memory_pattern' in memory and spike_patterns and 'spike_sequence' in memory['memory_pattern']:
                # Compare current spike sequence with stored memory pattern
                # This compares the temporal order of neuron activation
                temporal_similarity = self._calculate_temporal_similarity(
                    memory['memory_pattern']['spike_sequence'],
                    temporal_info.get('spike_sequence', [])
                )
            
            # Calculate memory age factor (newer memories more accessible)
            age_seconds = time.time() - memory.get('encoding_time', time.time())
            age_days = age_seconds / (24 * 3600)
            recency_factor = max(0.3, min(1.0, 1.0 - (age_days / 30)))  # Decay over 30 days to 0.3
            
            # Calculate access frequency factor (more frequently accessed = more accessible)
            access_count = memory.get('access_count', 1)
            access_factor = min(1.0, 0.7 + (access_count / 20) * 0.3)  # Increases with access count
            
            # Calculate importance factor
            importance_factor = memory.get('importance', 0.5)
            
            # Calculate final relevance score
            relevance = (
                0.4 * overlap_ratio +       # Spatial overlap with active neurons
                0.2 * avg_potential +       # Average membrane potential
                0.2 * temporal_similarity + # Temporal pattern similarity
                0.1 * recency_factor +      # Recency
                0.05 * access_factor +      # Access frequency
                0.05 * importance_factor    # Importance
            )
            
            # Include if sufficiently relevant
            if relevance >= self.retrieval_threshold or overlap_ratio > 0.4 or temporal_similarity > 0.6:
                content = memory.get('content', f"Memory {content_key}")
                
                # Add to results
                results.append({
                    'content': content,
                    'content_key': content_key,
                    'memory_type': memory['type'],
                    'relevance': relevance,
                    'overlap_ratio': overlap_ratio,
                    'temporal_similarity': temporal_similarity,
                    'encoding_time': memory.get('encoding_time'),
                    'access_count': access_count,
                    'context_tag': memory.get('context_tag')
                })
        
        return results
    
    def _calculate_temporal_similarity(self, pattern1, pattern2):
        """
        Calculate similarity between two temporal patterns.
        
        Args:
            pattern1: First spike sequence
            pattern2: Second spike sequence
            
        Returns:
            Similarity score (0-1)
        """
        if not pattern1 or not pattern2:
            return 0.0
        
        # Prepare sequence of neurons ordered by spike time
        seq1 = [(t, n) for t, n, _ in pattern1]
        seq2 = [(t, n) for t, n, _ in pattern2]
        
        # Normalize time values to the range 0-1
        if seq1:
            min_t1 = min(t for t, _ in seq1)
            max_t1 = max(t for t, _ in seq1)
            time_range1 = max(1, max_t1 - min_t1)
            seq1 = [((t - min_t1) / time_range1, n) for t, n in seq1]
        
        if seq2:
            min_t2 = min(t for t, _ in seq2)
            max_t2 = max(t for t, _ in seq2)
            time_range2 = max(1, max_t2 - min_t2)
            seq2 = [((t - min_t2) / time_range2, n) for t, n in seq2]
        
        # Extract neuron sequences (ignore exact timing, just keep order)
        neurons1 = [n for _, n in seq1]
        neurons2 = [n for _, n in seq2]
        
        # Find longest common subsequence of firing neurons
        lcs_length = self._longest_common_subsequence(neurons1, neurons2)
        
        # Calculate order similarity (how much of the firing sequence is preserved)
        order_similarity = lcs_length / min(len(neurons1), len(neurons2)) if min(len(neurons1), len(neurons2)) > 0 else 0
        
        # Calculate set similarity (what fraction of neurons are common regardless of timing)
        neurons1_set = set(neurons1)
        neurons2_set = set(neurons2)
        set_similarity = len(neurons1_set.intersection(neurons2_set)) / len(neurons1_set.union(neurons2_set)) if neurons1_set.union(neurons2_set) else 0
        
        # Calculate timing similarity
        # Match neurons that appear in both sequences and compare their relative timing
        timing_diffs = []
        for neuron in neurons1_set.intersection(neurons2_set):
            # Find all occurrences of this neuron in both sequences
            times1 = [t for t, n in seq1 if n == neuron]
            times2 = [t for t, n in seq2 if n == neuron]
            
            # Calculate average timing differences
            if times1 and times2:
                # Compare first occurrence timing (normalized)
                timing_diff = abs(times1[0] - times2[0])
                timing_diffs.append(timing_diff)
        
        # Average timing difference (lower is better)
        avg_timing_diff = sum(timing_diffs) / len(timing_diffs) if timing_diffs else 1.0
        timing_similarity = 1.0 - min(1.0, avg_timing_diff)
        
        # Combine different similarity aspects
        combined_similarity = (0.4 * order_similarity + 
                              0.4 * set_similarity + 
                              0.2 * timing_similarity)
        
        return combined_similarity
    
    def _longest_common_subsequence(self, seq1, seq2):
        """
        Find length of longest common subsequence (neurons firing in same order)
        
        Args:
            seq1: First sequence of neurons
            seq2: Second sequence of neurons
            
        Returns:
            Length of LCS
        """
        if not seq1 or not seq2:
            return 0
        
        # Create DP table for LCS
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _extract_memory_features_for_snntorch(self, network_state):
        """
        Extract memory-relevant features from the SNN state for supervised learning.
        
        Args:
            network_state: Current network state (from process_input or simulation)
            
        Returns:
            Feature vector for SNNTorch networks
        """
        # Initialize feature vector
        features = []
        
        # FEATURE GROUP 1: Region activations
        # Extract activation levels from memory regions
        memory_regions = ['working_memory', 'episodic', 'semantic', 'hippocampal', 'context']
        region_activations = []
        
        for region in memory_regions:
            if region in self.regions:
                activation = self.regions[region]['activation']
                region_activations.append(activation)
            else:
                region_activations.append(0.0)  # Padding for missing regions
        
        features.extend(region_activations)
        
        # FEATURE GROUP 2: Memory retrieval metrics
        # Extract metrics related to memory retrieval and pattern completion
        
        # a) Active neurons ratio in each memory region
        active_ratios = []
        for region in memory_regions:
            if region in self.regions:
                region_neurons = set(self.regions[region]['neurons'])
                active_neurons = region_neurons.intersection(network_state.get('active_neurons', set()))
                ratio = len(active_neurons) / len(region_neurons) if region_neurons else 0
                active_ratios.append(ratio)
            else:
                active_ratios.append(0.0)
        
        features.extend(active_ratios)
        
        # b) Pattern completion metrics
        # Calculate stability and convergence of network state
        if hasattr(self, 'last_spike_patterns') and self.last_spike_patterns:
            # Calculate neural activity stability across timesteps
            activity_stability = self._calculate_activity_stability(self.last_spike_patterns)
            features.append(activity_stability)
            
            # Calculate convergence speed (how quickly activity stabilizes)
            convergence_speed = self._calculate_convergence_speed(self.last_spike_patterns)
            features.append(convergence_speed)
        else:
            # Padding if no spike patterns available
            features.extend([0.0, 0.0])
        
        # FEATURE GROUP 3: Memory engram overlap
        # Calculate overlap between current activity and existing memory engrams
        
        # Get top 3 engram overlaps
        engram_overlaps = self._calculate_engram_overlaps(network_state)
        
        # Add top 3 overlaps to features
        top_overlaps = [0.0, 0.0, 0.0]  # Default if no overlaps
        for i, (_, overlap) in enumerate(engram_overlaps[:3]):
            top_overlaps[i] = overlap
        
        features.extend(top_overlaps)
        
        # FEATURE GROUP 4: Network integration metrics
        # Add IIT-inspired metrics for consciousness
        features.append(network_state.get('phi', 0.0))  # Integration x differentiation
        features.append(network_state.get('integration', 0.0))
        features.append(network_state.get('differentiation', 0.0))
        
        # FEATURE GROUP 5: Temporal context features
        # Add features related to temporal context
        if self.context_buffer:
            # Recent context activation similarity
            recent_context = [item['activation'] for item in self.context_buffer]
            if recent_context:
                # Calculate average similarity between current activation and recent contexts
                similarity_sum = 0
                count = 0
                current_act = network_state.get('membrane_potentials', np.zeros(self.neuron_count))
                
                for context_act in recent_context:
                    # Calculate cosine similarity
                    dot_product = np.dot(current_act, context_act)
                    norm_product = np.linalg.norm(current_act) * np.linalg.norm(context_act)
                    if norm_product > 0:
                        similarity = dot_product / norm_product
                        similarity_sum += similarity
                        count += 1
                
                avg_similarity = similarity_sum / count if count > 0 else 0
                features.append(avg_similarity)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Additional padding to reach desired feature dimension (100)
        current_dim = len(features)
        target_dim = 100
        
        if current_dim < target_dim:
            # Add remaining integration metrics and zeros if needed
            features.extend([0.0] * (target_dim - current_dim))
        elif current_dim > target_dim:
            # Truncate if somehow we have too many features
            features = features[:target_dim]
        
        return np.array(features)
    
    def _calculate_activity_stability(self, spike_patterns):
        """
        Calculate the stability of neural activity across timesteps.
        
        Args:
            spike_patterns: List of spike events per timestep
            
        Returns:
            Stability metric (0-1)
        """
        if not spike_patterns or len(spike_patterns) < 2:
            return 0.0
        
        # Extract active neurons at each timestep
        active_per_step = []
        for step in spike_patterns:
            active = set(n for n, _ in step)
            active_per_step.append(active)
        
        # Calculate stability as average Jaccard similarity between consecutive timesteps
        stability_sum = 0
        for i in range(len(active_per_step) - 1):
            set1 = active_per_step[i]
            set2 = active_per_step[i + 1]
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            # Avoid division by zero
            if union > 0:
                stability_sum += intersection / union
        
        # Calculate average stability
        stability = stability_sum / (len(active_per_step) - 1) if len(active_per_step) > 1 else 0
        return stability

    def _calculate_convergence_speed(self, spike_patterns):
        """
        Calculate how quickly the network activity converges to a stable state.
        
        Args:
            spike_patterns: List of spike events per timestep
            
        Returns:
            Convergence speed metric (0-1)
        """
        if not spike_patterns or len(spike_patterns) < 3:
            return 0.0
        
        # Extract active neurons at each timestep
        active_per_step = []
        for step in spike_patterns:
            active = set(n for n, _ in step)
            active_per_step.append(active)
        
        # Calculate similarity changes between timesteps
        similarities = []
        for i in range(len(active_per_step) - 1):
            set1 = active_per_step[i]
            set2 = active_per_step[i + 1]
            
            # Calculate Jaccard similarity
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
        
        # Detect when similarity stabilizes (convergence)
        convergence_point = 0
        threshold = 0.8  # High similarity threshold indicating convergence
        
        for i, sim in enumerate(similarities):
            if sim > threshold:
                convergence_point = i + 1  # +1 because we're looking at pairs of timesteps
                break
        
        # Calculate convergence speed as inverse of convergence point
        # Earlier convergence = higher speed
        if convergence_point > 0:
            # Normalize to 0-1 range
            steps = len(spike_patterns)
            convergence_speed = 1.0 - (convergence_point / steps)
        else:
            # No convergence detected
            convergence_speed = 0.0
        
        return convergence_speed

    def _calculate_engram_overlaps(self, network_state):
        """
        Calculate overlap between current network activity and existing memory engrams.
        
        Args:
            network_state: Current network state
            
        Returns:
            List of (content_key, overlap_ratio) tuples, sorted by overlap
        """
        active_neurons = network_state.get('active_neurons', set())
        
        if not active_neurons or not self.memory_engrams:
            return []
        
        overlaps = []
        
        # Calculate overlap with each memory engram
        for content_key, memory in self.memory_engrams.items():
            engram_neurons = set(memory.get('neurons', []))
            
            if engram_neurons:
                # Calculate overlap between active neurons and engram
                intersection = active_neurons.intersection(engram_neurons)
                overlap_ratio = len(intersection) / len(engram_neurons)
                
                overlaps.append((content_key, overlap_ratio))
        
        # Sort by overlap ratio (descending)
        overlaps.sort(key=lambda x: x[1], reverse=True)
        
        return overlaps

    def _extract_content_features(self, content):
        """
        Extract feature vector from memory content.
        
        This method uses a hybrid approach:
        1. Word embeddings when available for semantic representation
        2. Bag-of-words with consistent hashing as fallback
        3. Deterministic random features for non-text content
        
        Future improvements could include:
        - Trained content encoder network
        - Attention-based feature extraction
        - Multi-modal representations
        
        Args:
            content: Memory content (typically text)
            
        Returns:
            Feature vector (50-dimensional)
        """
        # Default feature dimension
        feature_dim = 50
        features = np.zeros(feature_dim)
        
        if not content:
            return features
        
        # For text content, use enhanced tokenization and embeddings
        if isinstance(content, str):
            # Tokenize the content
            tokens = self.tokenizer.tokenize(content)
            
            # Use word vectors if available
            if hasattr(self, 'word_vectors') and self.has_embeddings:
                # Average word vectors for the tokens
                embeddings = []
                oov_tokens = []  # Track out-of-vocabulary tokens
                
                for token in tokens:
                    if token in self.word_vectors:
                        embeddings.append(self.word_vectors[token])
                    else:
                        oov_tokens.append(token)
                
                # Combine embeddings and OOV representations
                if embeddings:
                    # Average embeddings for in-vocabulary tokens
                    avg_embedding = np.mean(embeddings, axis=0)
                    
                    # Resize to feature dimension if needed
                    if len(avg_embedding) > feature_dim:
                        features = avg_embedding[:feature_dim]
                    else:
                        # Pad with zeros if needed
                        features[:len(avg_embedding)] = avg_embedding
                
                # Add representation for OOV tokens using consistent hashing
                if oov_tokens and np.sum(features) == 0:  # Only if no embeddings found
                    for token in oov_tokens:
                        # Create consistent hash-based features
                        token_hash = hash(token) % (feature_dim * 10)
                        index = token_hash % feature_dim
                        value = 0.1 + (token_hash % 90) / 100
                        features[index] = max(features[index], value)  # Keep maximum activation
            else:
                # Enhanced bag-of-words with n-gram features
                # Create unigram and bigram features
                for i, token in enumerate(tokens):
                    # Unigram feature
                    token_hash = hash(token) % (feature_dim * 10)
                    index = token_hash % feature_dim
                    value = 0.1 + (token_hash % 90) / 100
                    features[index] = max(features[index], value)
                    
                    # Bigram feature (if next token exists)
                    if i < len(tokens) - 1:
                        bigram = f"{token}_{tokens[i+1]}"
                        bigram_hash = hash(bigram) % (feature_dim * 10)
                        index = bigram_hash % (feature_dim // 2) + (feature_dim // 2)
                        value = 0.05 + (bigram_hash % 90) / 200
                        features[index] = max(features[index], value)
        
        # For other content types or as a fallback, create consistent features
        if np.sum(features) == 0:
            # Generate a deterministic feature vector based on content hash
            content_hash = hash(str(content))
            # Use the hash to seed a random generator for consistency
            rng = np.random.RandomState(content_hash)
            features = rng.rand(feature_dim)
        
        # Normalize to unit length for stability
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features

    def train_memory_retrieval(self, query_activation, target_memory, learn_rate=0.02):
        """
        Train memory retrieval using hybrid learning approach.
        
        Error signals and custom plasticity relationship:
        1. recall_loss: BCE loss between predicted and actual retrieval success
        - High loss -> strengthen query-to-engram connections
        - Low loss -> reinforce existing pathways
        
        2. content_loss: MSE between predicted and actual content features
        - Used to adjust memory content representation
        - Guides which dimensions of the engram need modification
        
        3. recall_error: Simple difference (actual - predicted) for bio-inspired learning
        - Positive error: successful retrieval, strengthen pathways
        - Negative error: failed retrieval, create new pathways
        
        The combined_error drives custom plasticity rules that modify:
        - Query-to-engram connections
        - Within-engram connections
        - Hippocampal support pathways
        
        Args:
            query_activation: Retrieval cue activation pattern
            target_memory: Target memory to retrieve
            learn_rate: Learning rate for weight updates
            
        Returns:
            Training results with loss metrics and update counts
        """
        # PHASE 1: PREPARE INPUTS AND TARGETS
        
        # Set training mode to enable supervised guidance
        self.training_mode = True
        self.disable_parent_plasticity = True
        
        # Process query through network
        network_result = self.process_input(query_activation)
        
        # Extract bio-inspired features for supervised networks
        bio_features = self._extract_memory_features_for_snntorch(network_result)
        
        # Run memory retrieval to get actual result
        retrieved_memories = self.retrieve_memory(query_activation, max_results=3)
        
        # Determine if retrieval was successful
        target_key = target_memory.get('content_key')
        target_retrieved = False
        retrieved_memory = None
        
        for memory in retrieved_memories:
            if memory['content_key'] == target_key:
                target_retrieved = True
                retrieved_memory = memory
                break
        
        # PHASE 2: SUPERVISED LEARNING
        # Prepare data for SNNTorch networks
        if self.torch_available:
            import torch
            
            # Convert features to tensor
            features_tensor = torch.tensor(bio_features, dtype=torch.float32).unsqueeze(0)
            
            # Train recall prediction network
            recall_target = torch.tensor([[1.0 if target_retrieved else 0.0]], dtype=torch.float32)
            
            # Forward pass and loss calculation
            self.recall_optimizer.zero_grad()
            recall_output = self.recall_network(features_tensor)
            recall_loss = self.recall_loss_fn(recall_output, recall_target)
            
            # Backward pass - update supervised network
            recall_loss.backward()
            self.recall_optimizer.step()
            
            # Train content prediction network if content is available
            content_loss = 0.0
            if target_memory.get('content'):
                content_features = self._extract_content_features(target_memory['content'])
                content_target = torch.tensor(content_features, dtype=torch.float32).unsqueeze(0)
                
                # Forward pass and loss calculation
                self.content_optimizer.zero_grad()
                content_output = self.content_network(features_tensor)
                content_loss = self.content_loss_fn(content_output, content_target)
                
                # Backward pass - update supervised network
                content_loss.backward()
                self.content_optimizer.step()
                
                # Get content prediction for bio-inspired learning
                content_prediction = content_output.detach().numpy()[0]
            else:
                content_prediction = None
        else:
            # No supervised learning if torch isn't available
            recall_loss = 0.0
            content_loss = 0.0
            content_prediction = None
        
        # PHASE 3: BIO-INSPIRED LEARNING
        # Calculate errors for bio-inspired learning
        
        # 1. Recall error
        if self.torch_available:
            # Use supervised prediction as guidance
            predicted_recall_prob = recall_output.item()
            recall_error = float(target_retrieved) - predicted_recall_prob
        else:
            # Simple error based on retrieval success
            recall_error = 1.0 if target_retrieved else -0.5
        
        # 2. Combine errors to drive bio-inspired plasticity
        combined_error = recall_error
        
        # Apply specialized weight updates based on error
        if target_retrieved:
            # Target was successfully retrieved - reinforce this pathway
            weights_updated = self._strengthen_successful_retrieval(
                query_activation, retrieved_memory, combined_error, learn_rate)
        else:
            # Target was not retrieved - modify weights to improve future recall
            weights_updated = self._improve_failed_retrieval(
                query_activation, target_memory, combined_error, learn_rate)
        
        # Additional updates for memory representation if content prediction is available
        if content_prediction is not None and target_memory.get('content'):
            target_content = self._extract_content_features(target_memory['content'])
            content_error = target_content - content_prediction
            content_updates = self._update_memory_content_representation(
                target_key, content_error, learn_rate * 0.5)
            weights_updated += content_updates
        
        # PHASE 4: UPDATE PLASTIC THRESHOLDS BASED ON ERRORS
        threshold_updates = 0
        
        # Update thresholds for memory engram neurons
        if target_memory.get('content_key') in self.memory_engrams:
            engram = self.memory_engrams[target_memory['content_key']]
            engram_neurons = engram.get('neurons', [])
            
            # If retrieval failed, decrease thresholds to make neurons more excitable
            # If retrieval succeeded, adjust thresholds based on confidence
            if not target_retrieved:
                # Make engram neurons more excitable if retrieval failed
                error_signal = 0.3  # Positive error -> decrease threshold
                threshold_updates += self._apply_threshold_plasticity(
                    engram_neurons, 
                    error_signal, 
                    learning_rate=learn_rate * 0.5
                )
            else:
                # For successful retrieval, modulate based on confidence
                confidence = retrieved_memory.get('relevance', 0.5)
                if confidence < 0.7:  # If confidence is low, make neurons more excitable
                    error_signal = 0.2
                else:  # If confidence is high, maintain or slightly reduce excitability
                    error_signal = -0.1
                
                threshold_updates += self._apply_threshold_plasticity(
                    engram_neurons,
                    error_signal,
                    learning_rate=learn_rate * 0.3
                )
        
        # Reset flags after training
        self.disable_parent_plasticity = False
        self.training_mode = False
        
        # Return training results
        return {
            'retrieved_success': target_retrieved,
            'recall_loss': float(recall_loss) if isinstance(recall_loss, (int, float)) else float(recall_loss.item()),
            'content_loss': float(content_loss) if isinstance(content_loss, (int, float)) else float(content_loss.item()) if content_loss else 0.0,
            'recall_error': recall_error,
            'weights_updated': weights_updated,
            'threshold_updates': threshold_updates,  # Add this to output
            'accuracy': 1.0 if target_retrieved else 0.0
        }

    def _strengthen_successful_retrieval(self, query, retrieved_memory, error, learn_rate):
        """
        Strengthen connections for successful memory retrieval.
        
        PLASTICITY RULE LOGIC:
        - error > 0: Model correctly predicted retrieval (reinforce pathway)
        - error < 0: Model incorrectly thought retrieval would fail (mild weakening)
        
        Weight updates target:
        1. Query → Engram connections: Make retrieval cue more likely to activate memory
        2. Within-engram connections: Stabilize memory pattern for better completion
        
        The error magnitude modulates the strength of weight changes, with positive
        errors leading to stronger reinforcement.
        
        Args:
            query: Query activation pattern
            retrieved_memory: Successfully retrieved memory
            error: Error signal from supervised prediction (recall_error)
            learn_rate: Learning rate
            
        Returns:
            Number of weights updated
        """
        if not retrieved_memory:
            return 0
        
        weights_updated = 0
        content_key = retrieved_memory.get('content_key')
        
        # Get memory engram
        if content_key not in self.memory_engrams:
            return 0
        
        engram = self.memory_engrams[content_key]
        engram_neurons = engram.get('neurons', [])
        
        if not engram_neurons:
            return 0
        
        # Get active query neurons
        active_query_neurons = np.where(query > 0.3)[0]
        
        # Strengthen connections from query to engram
        for query_neuron in active_query_neurons:
            for engram_neuron in engram_neurons:
                # Skip if not registered for custom learning
                if (query_neuron, engram_neuron) not in self.specialized_learning_synapses:
                    continue
                    
                current_weight = self.synaptic_weights[query_neuron, engram_neuron]
                
                # Calculate weight change based on error and query activation
                weight_change = learn_rate * abs(error) * query[query_neuron] * 0.5
                if error > 0:
                    # Strengthen connection (positive error)
                    new_weight = min(1.0, current_weight + weight_change)
                else:
                    # Weaken connection (negative error)
                    new_weight = max(0.0, current_weight - weight_change * 0.3)  # Less aggressive weakening
                
                if abs(new_weight - current_weight) > 0.001:
                    self.synaptic_weights[query_neuron, engram_neuron] = new_weight
                    weights_updated += 1
        
        # Strengthen within-engram connections
        if error > 0:
            # Strengthen internal engram connections for better pattern completion
            for i, pre in enumerate(engram_neurons):
                for post in engram_neurons[i+1:]:
                    # Skip if not registered for custom learning
                    if (pre, post) not in self.specialized_learning_synapses:
                        continue
                        
                    current_weight = self.synaptic_weights[pre, post]
                    
                    # Strengthen connection (make uniform for better pattern completion)
                    target_weight = 0.7  # Target uniform weight for memory engram
                    weight_change = learn_rate * abs(error) * (target_weight - current_weight)
                    
                    new_weight = current_weight + weight_change
                    new_weight = max(0.0, min(1.0, new_weight))
                    
                    if abs(new_weight - current_weight) > 0.001:
                        # Apply symmetrically
                        self.synaptic_weights[pre, post] = new_weight
                        self.synaptic_weights[post, pre] = new_weight
                        weights_updated += 2
        
        return weights_updated

    def _improve_failed_retrieval(self, query, target_memory, error, learn_rate):
        """
        Improve weights for cases where retrieval failed but should have succeeded.
        
        PLASTICITY RULE LOGIC:
        - error < 0: Model correctly predicted failure, but we want success
        - Creates new pathways from query to target engram
        - Strengthens engram internal connections for stability
        - Adds hippocampal support pathways
        
        Key insight: Failed retrievals often indicate missing or weak connections
        between the cue and memory engram. This rule creates those connections.
        
        Args:
            query: Query activation pattern that failed to retrieve target
            target_memory: Target memory that should have been retrieved
            error: Error signal (negative for failed retrievals)
            learn_rate: Learning rate
            
        Returns:
            Number of weights updated
        """
        weights_updated = 0
        content_key = target_memory.get('content_key')
        
        # Get memory engram
        if content_key not in self.memory_engrams:
            return 0
        
        engram = self.memory_engrams[content_key]
        engram_neurons = engram.get('neurons', [])
        
        if not engram_neurons:
            return 0
        
        # Get active query neurons
        active_query_neurons = np.where(query > 0.3)[0]
        
        if not active_query_neurons.size:
            return 0
        
        # Scale error - abs value since error should be negative for failed retrieval
        scaled_error = abs(error) * learn_rate
        
        # 1. Create new connections between query and engram
        # Focus on creating stronger pathways where they don't exist
        for query_neuron in active_query_neurons:
            # Select a subset of engram neurons for efficiency
            target_neurons = random.sample(engram_neurons, min(10, len(engram_neurons)))
            for engram_neuron in target_neurons:
                # Skip if not registered for custom learning
                if (query_neuron, engram_neuron) not in self.specialized_learning_synapses:
                    continue
                    
                current_weight = self.synaptic_weights[query_neuron, engram_neuron]
                
                # Calculate weight increase
                weight_increase = scaled_error * query[query_neuron] * 2.0  # Stronger increase for failed retrievals
                
                # Significant boost, especially if current connection is weak
                new_weight = min(1.0, current_weight + weight_increase)
                
                if abs(new_weight - current_weight) > 0.001:
                    self.synaptic_weights[query_neuron, engram_neuron] = new_weight
                    weights_updated += 1
        
        # 2. Strengthen internal engram connections for more stable pattern completion
        # This is crucial for failed retrievals - make the engram more self-sustaining
        for i, pre in enumerate(engram_neurons):
            for post in engram_neurons[i+1:]:
                # Skip if not registered for custom learning
                if (pre, post) not in self.specialized_learning_synapses:
                    continue
                    
                current_weight = self.synaptic_weights[pre, post]
                
                # Target higher weights for failed retrievals (strengthen internal connections)
                target_weight = 0.8  # Higher target for failed retrievals
                weight_change = scaled_error * (target_weight - current_weight)
                
                new_weight = current_weight + weight_change
                new_weight = max(0.0, min(1.0, new_weight))
                
                if abs(new_weight - current_weight) > 0.001:
                    # Apply symmetrically
                    self.synaptic_weights[pre, post] = new_weight
                    self.synaptic_weights[post, pre] = new_weight
                    weights_updated += 2
        
        # 3. Connect to hippocampal region for retrieval support
        if 'hippocampal' in self.regions:
            # Add connections through hippocampal region to aid retrieval
            hippocampal_neurons = random.sample(
                self.regions['hippocampal']['neurons'],
                min(5, len(self.regions['hippocampal']['neurons']))
            )
            
            for hpc_neuron in hippocampal_neurons:
                # Create query → hippocampal → engram pathway
                for query_neuron in active_query_neurons[:5]:  # Limit to 5 query neurons
                    # Skip if not registered for custom learning
                    if (query_neuron, hpc_neuron) not in self.specialized_learning_synapses:
                        continue
                        
                    # Strengthen query → hippocampal connection
                    current_weight = self.synaptic_weights[query_neuron, hpc_neuron]
                    new_weight = min(1.0, current_weight + scaled_error * 0.5)
                    
                    if abs(new_weight - current_weight) > 0.001:
                        self.synaptic_weights[query_neuron, hpc_neuron] = new_weight
                        weights_updated += 1
                
                # Connect hippocampal → engram
                for engram_neuron in random.sample(engram_neurons, min(5, len(engram_neurons))):
                    # Skip if not registered for custom learning
                    if (hpc_neuron, engram_neuron) not in self.specialized_learning_synapses:
                        continue
                        
                    # Strengthen hippocampal → engram connection
                    current_weight = self.synaptic_weights[hpc_neuron, engram_neuron]
                    new_weight = min(1.0, current_weight + scaled_error * 0.5)
                    
                    if abs(new_weight - current_weight) > 0.001:
                        self.synaptic_weights[hpc_neuron, engram_neuron] = new_weight
                        weights_updated += 1
        
        return weights_updated

    def _update_memory_content_representation(self, content_key, content_error, learn_rate):
        """
        Update memory representation based on content prediction error.
        This helps align the bio-inspired memory representation with supervised content predictions.
        
        Args:
            content_key: Memory content key
            content_error: Content prediction error vector
            learn_rate: Learning rate for updates
            
        Returns:
            Number of weights updated
        """
        if content_key not in self.memory_engrams:
            return 0
            
        engram = self.memory_engrams[content_key]
        engram_neurons = engram.get('neurons', [])
        
        if not engram_neurons:
            return 0
            
        weights_updated = 0
        
        # Normalize error vector
        error_magnitude = np.linalg.norm(content_error)
        if error_magnitude > 0:
            normalized_error = content_error / error_magnitude
        else:
            return 0
            
        # Use error to guide memory pattern adjustments
        
        # 1. Get semantic region neurons
        semantic_neurons = self.regions.get('semantic', {}).get('neurons', [])
        
        # 2. Create neuron groups associated with different dimensions of the content
        error_dimension = len(normalized_error)
        neuron_groups = []
        
        # Create neuron groups from both engram and semantic region
        if semantic_neurons:
            # Divide semantic neurons into groups
            neurons_per_group = max(1, len(semantic_neurons) // error_dimension)
            for i in range(min(error_dimension, len(semantic_neurons) // neurons_per_group)):
                start_idx = i * neurons_per_group
                end_idx = min((i + 1) * neurons_per_group, len(semantic_neurons))
                neuron_groups.append(semantic_neurons[start_idx:end_idx])
                
        # Add engram neurons divided into groups
        neurons_per_group = max(1, len(engram_neurons) // error_dimension)
        for i in range(min(error_dimension, len(engram_neurons) // neurons_per_group)):
            start_idx = i * neurons_per_group
            end_idx = min((i + 1) * neurons_per_group, len(engram_neurons))
            neuron_groups.append(engram_neurons[start_idx:end_idx])
            
        # Apply weight updates based on content error dimensions
        for i, group in enumerate(neuron_groups):
            if i >= len(normalized_error):
                break
                
            # Get error component for this dimension
            error_component = normalized_error[i]
            
            if abs(error_component) < 0.1:
                continue  # Skip small errors
                
            # Adjust connections within this group
            for j, pre in enumerate(group):
                for k, post in enumerate(group):
                    if j != k:  # Skip self-connections
                        # Skip if not registered for custom learning
                        if (pre, post) not in self.specialized_learning_synapses:
                            continue
                        
                        current_weight = self.synaptic_weights[pre, post]
                        
                        # Calculate weight change (strengthen if positive error, weaken if negative)
                        weight_change = learn_rate * error_component
                        
                        # Update weight with bounds
                        new_weight = max(0.0, min(1.0, current_weight + weight_change))
                        
                        if abs(new_weight - current_weight) > 0.001:
                            self.synaptic_weights[pre, post] = new_weight
                            weights_updated += 1
        
        return weights_updated
    
    def train_memory(self, cue_activation, target_memory, epochs=3, learn_rate=0.02):
        """
        Train memory encoding and retrieval for a single memory example.
        
        TRAINING STRUCTURE:
        This method processes ONE memory example through multiple internal iterations (epochs).
        It is designed to be called from an external training script that handles the outer
        loop over training samples.
        
        Expected usage pattern:
        ```python
        # In external training script
        for epoch in range(outer_epochs):
            for cue, memory in training_data:
                # Train this single example with internal iterations
                result = memory_snn.train_memory(cue, memory, epochs=3)
        ```
        
        The internal epochs allow the model to refine learning on each individual
        memory through repeated encoding/retrieval cycles before moving to the next sample.
        
        Args:
            cue_activation: Activation pattern for retrieval cue
            target_memory: Target memory to encode and retrieve
            epochs: Number of internal training iterations for this memory example
            learn_rate: Learning rate for weight updates
            
        Returns:
            Training results including success rates and losses
        """
        # Initialize results
        training_results = {
            'epochs': epochs,
            'encoding_success': 0,
            'retrieval_success': 0,
            'recall_loss': [],
            'content_loss': [],
            'encoding_strength': 0,
            'weights_updated': 0
        }
        
        # Check if memory already exists
        content_key = target_memory.get('content_key')
        existing_memory = None
        if content_key and content_key in self.memory_engrams:
            existing_memory = self.memory_engrams[content_key]
        
        # 1. First encode or strengthen the memory
        if existing_memory:
            # Memory already exists - strengthen it
            encoding_result = self._strengthen_existing_memory(target_memory, learn_rate)
            training_results['encoding_strength'] = encoding_result.get('encoding_strength', 0)
            training_results['weights_updated'] += encoding_result.get('weights_updated', 0)
            training_results['encoding_success'] = 1
        else:
            # Encode new memory
            encoding_result = self.encode_memory(
                target_memory.get('activation', cue_activation), 
                target_memory.get('content'),
                memory_type=target_memory.get('memory_type', 'working'),
                importance=target_memory.get('importance', 0.5)
            )
            
            content_key = encoding_result.get('content_key')
            training_results['encoding_success'] = 1 if encoding_result.get('success', False) else 0
        
        # Update target_memory with content_key if needed
        if content_key and 'content_key' not in target_memory:
            target_memory['content_key'] = content_key
        
        # 2. Run hybrid training loop for retrieval
        for epoch in range(epochs):
            # Train memory retrieval using hybrid approach
            retrieval_result = self.train_memory_retrieval(cue_activation, target_memory, learn_rate)
            
            # Update results
            training_results['weights_updated'] += retrieval_result.get('weights_updated', 0)
            training_results['retrieval_success'] += 1 if retrieval_result.get('retrieved_success', False) else 0
            training_results['recall_loss'].append(retrieval_result.get('recall_loss', 0))
            training_results['content_loss'].append(retrieval_result.get('content_loss', 0))
            
            # Break early if perfect retrieval
            if retrieval_result.get('accuracy', 0) == 1.0 and epoch >= 1:
                break
        
        # Calculate average losses
        if training_results['recall_loss']:
            training_results['avg_recall_loss'] = sum(training_results['recall_loss']) / len(training_results['recall_loss'])
        if training_results['content_loss']:
            training_results['avg_content_loss'] = sum(training_results['content_loss']) / len(training_results['content_loss'])
            
        # Normalize retrieval success
        training_results['retrieval_success'] /= max(1, epochs)
        
        return training_results

    def _strengthen_existing_memory(self, memory, learn_rate=0.02):
        """
        Strengthen an existing memory engram.
        
        Args:
            memory: Memory to strengthen
            learn_rate: Learning rate for weight updates
            
        Returns:
            Results of strengthening operation
        """
        content_key = memory.get('content_key')
        if not content_key or content_key not in self.memory_engrams:
            return {'success': False, 'error': 'Memory not found'}
            
        engram = self.memory_engrams[content_key]
        engram_neurons = engram.get('neurons', [])
        
        if not engram_neurons:
            return {'success': False, 'error': 'Invalid engram'}
            
        # 1. Strengthen within-engram connections
        strength = 0.5 * learn_rate
        weights_updated = self._strengthen_engram_connections(engram_neurons, strength)
        
        # 2. Strengthen connections to hippocampal region (for consolidation)
        if 'hippocampal' in self.regions:
            hippocampal_neurons = self.regions['hippocampal']['neurons']
            
            # Connect a subset of engram neurons to hippocampal neurons
            engram_subset = random.sample(engram_neurons, min(len(engram_neurons), 20))
            hpc_subset = random.sample(hippocampal_neurons, min(len(hippocampal_neurons), 20))
            
            for engram_neuron in engram_subset:
                for hpc_neuron in hpc_subset:
                    # Skip if not registered for custom learning
                    if (engram_neuron, hpc_neuron) not in self.specialized_learning_synapses:
                        continue
                        
                    current_weight = self.synaptic_weights[engram_neuron, hpc_neuron]
                    weight_change = strength * 0.5
                    new_weight = min(1.0, current_weight + weight_change)
                    
                    if abs(new_weight - current_weight) > 0.001:
                        self.synaptic_weights[engram_neuron, hpc_neuron] = new_weight
                        weights_updated += 1
        
        # 3. Update memory statistics
        engram['importance'] = min(1.0, engram.get('importance', 0.5) + 0.1)
        engram['encoding_strength'] = min(1.0, engram.get('encoding_strength', 0.5) + 0.1)
        
        return {
            'success': True,
            'memory': content_key,
            'weights_updated': weights_updated,
            'encoding_strength': engram.get('encoding_strength', 0.5)
        }

    def learn_from_experience(self, memory_examples, epochs=3):
        """
        Learn from a batch of memory examples using the hybrid learning approach.
        
        Args:
            memory_examples: List of (cue_activation, target_memory) pairs
            epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        if not memory_examples:
            return {"error": "No training data provided"}
        
        metrics = {
            "epochs": epochs,
            "samples": len(memory_examples),
            "recall_loss": [],
            "content_loss": [],
            "encoding_success": 0,
            "retrieval_success": 0,
            "learning_curve": []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_recall_loss = 0
            epoch_content_loss = 0
            epoch_encoding_success = 0
            epoch_retrieval_success = 0
            
            # Process each training sample
            random.shuffle(memory_examples)  # Shuffle for better learning
            for cue_activation, target_memory in memory_examples:
                # Train memory using hybrid approach
                train_result = self.train_memory(
                    cue_activation, 
                    target_memory, 
                    epochs=1,  # Single epoch per sample
                    learn_rate=0.02
                )
                
                # Update metrics
                epoch_encoding_success += train_result.get('encoding_success', 0)
                epoch_retrieval_success += train_result.get('retrieval_success', 0)
                
                if train_result.get('recall_loss'):
                    epoch_recall_loss += sum(train_result['recall_loss']) / len(train_result['recall_loss'])
                    
                if train_result.get('content_loss'):
                    epoch_content_loss += sum(train_result['content_loss']) / len(train_result['content_loss'])
                
                # Update training iteration counter
                if hasattr(self, 'training_iterations'):
                    self.training_iterations += 1
            
            # Calculate averages for this epoch
            avg_recall_loss = epoch_recall_loss / len(memory_examples)
            avg_content_loss = epoch_content_loss / len(memory_examples)
            avg_encoding_success = epoch_encoding_success / len(memory_examples)
            avg_retrieval_success = epoch_retrieval_success / len(memory_examples)
            
            # Store metrics
            metrics["recall_loss"].append(avg_recall_loss)
            metrics["content_loss"].append(avg_content_loss)
            metrics["encoding_success_rate"] = avg_encoding_success
            metrics["retrieval_success_rate"] = avg_retrieval_success
            
            # Simple learning curve (inverse of losses)
            learning_curve_value = 1.0 - (avg_recall_loss + avg_content_loss) / 4.0
            metrics["learning_curve"].append(learning_curve_value)
            
            print(f"Epoch {epoch+1}/{epochs}: Recall Loss: {avg_recall_loss:.4f}, " +
                  f"Content Loss: {avg_content_loss:.4f}, " +
                  f"Retrieval Success: {avg_retrieval_success:.2f}")
        
        # Evaluate on the data set to measure progress
        evaluation = self.evaluate_memory_performance(memory_examples)
        metrics.update(evaluation)
        
        # Transfer knowledge between bio-inspired and supervised components
        if hasattr(self, 'training_iterations') and self.training_iterations % 50 == 0:
            # Periodically transfer knowledge to keep components aligned
            print("Transferring knowledge between bio-inspired and supervised components...")
            if hasattr(self, 'transfer_knowledge_to_supervised'):
                self.transfer_knowledge_to_supervised()
            if hasattr(self, 'transfer_supervised_to_bio'):
                self.transfer_supervised_to_bio()
        
        return metrics

    def evaluate_memory_performance(self, test_data):
        """
        Evaluate the memory system's performance on a set of test queries.
        
        Args:
            test_data: List of (query, target_memory) pairs
            
        Returns:
            Evaluation metrics including precision, recall, F1, content MSE, retrieval latency, and completion quality
        """
        if hasattr(self, 'training_mode'):
            self.training_mode = False  # Ensure we're in inference mode
        
        metrics = {
            "recall_accuracy": 0,
            "recall_precision": 0,
            "recall_f1": 0,
            "content_mse": 0,
            "retrieval_latency": 0,
            "completion_quality": 0
        }
        
        total_samples = len(test_data)
        if total_samples == 0:
            return metrics
        
        # Track performance on each test case
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Track content prediction errors and latencies
        content_mse_sum = 0
        content_predictions_count = 0
        total_latency = 0
        
        for query, target_memory in test_data:
            # Process query through network
            start_time = time.time()
            retrieved_memories = self.retrieve_memory(query, max_results=3)
            retrieval_time = time.time() - start_time
            total_latency += retrieval_time
            
            # Check if target was successfully retrieved
            target_content_key = target_memory.get('content_key')
            target_retrieved = False
            retrieved_memory = None
            
            # Check if the target memory is in the retrieved memories
            for memory in retrieved_memories:
                if memory.get('content_key') == target_content_key:
                    target_retrieved = True
                    retrieved_memory = memory
                    break
            
            # Update true positives and false negatives
            if target_retrieved:
                true_positives += 1
                
                # Calculate content MSE if possible
                if (self.torch_available and 
                    retrieved_memory and 
                    target_memory.get('content') is not None):
                    
                    # Extract actual content features
                    actual_content_features = self._extract_content_features(target_memory['content'])
                    
                    # Get network state and features for content prediction
                    network_state = self._get_network_state()
                    network_features = self._extract_memory_features_for_snntorch(network_state)
                    
                    # Predict content using the content network
                    import torch
                    features_tensor = torch.tensor(network_features, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        predicted_content = self.content_network(features_tensor).squeeze(0).numpy()
                    
                    # Calculate MSE between predicted and actual content
                    content_mse = np.mean((actual_content_features - predicted_content) ** 2)
                    content_mse_sum += content_mse
                    content_predictions_count += 1
                    
            else:
                false_negatives += 1
            
            # Count false positives (retrieved memories that aren't the target)
            for memory in retrieved_memories:
                if memory.get('content_key') != target_content_key:
                    false_positives += 1
        
        # Calculate final metrics
        # Precision = TP / (TP + FP)
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        # Recall = TP / (TP + FN)
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Accuracy = TP / Total
        accuracy = true_positives / total_samples if total_samples > 0 else 0.0
        
        # Average content MSE (only for successfully retrieved targets)
        avg_content_mse = content_mse_sum / content_predictions_count if content_predictions_count > 0 else 0.0
        
        # Average retrieval latency
        avg_latency = total_latency / total_samples if total_samples > 0 else 0.0
        
        # Calculate pattern completion quality
        completion_quality = self._calculate_average_completion_quality(test_data)
        
        # Update metrics dictionary
        metrics["recall_accuracy"] = accuracy
        metrics["recall_precision"] = precision
        metrics["recall_f1"] = f1_score
        metrics["content_mse"] = avg_content_mse
        metrics["retrieval_latency"] = avg_latency
        metrics["completion_quality"] = completion_quality
        
        return metrics
    
    def _evaluate_completion_quality(self, test_data):
        """
        Evaluate the quality of pattern completion during memory retrieval.
        
        Args:
            test_data: List of (query, target_memory) pairs
            
        Returns:
            Completion quality score (0-1)
        """
        if not test_data:
            return 0.0
        
        total_quality = 0.0
        valid_samples = 0
        
        for query, target_memory in test_data:
            # Test pattern completion with partial query
            partial_query = query.copy()
            
            # Mask out 30% of the query activation randomly
            mask_indices = np.random.choice(len(partial_query), size=int(len(partial_query) * 0.3), replace=False)
            partial_query[mask_indices] = 0.0
            
            # Perform pattern completion
            completion_result = self.perform_pattern_completion(partial_query)
            
            if completion_result.get('success'):
                # Evaluate quality based on convergence and stability
                quality = (
                    completion_result.get('convergence_quality', 0.0) * 0.4 +
                    completion_result.get('final_stability', 0.0) * 0.3 +
                    completion_result.get('confidence', 0.0) * 0.3
                )
                
                total_quality += quality
                valid_samples += 1
        
        # Return average quality
        return total_quality / valid_samples if valid_samples > 0 else 0.0

    def _apply_homeostatic_plasticity(self):
        """
        Apply homeostatic plasticity to maintain stability in the network.
        This prevents runaway activity or silencing of neurons.
        """
        # Use parent class's homeostatic plasticity implementation if available
        if hasattr(super(), '_apply_homeostatic'):
            super()._apply_homeostatic()
            return
            
        # Target activation level for neurons (fraction of time active)
        target_activity = 0.1  # 10% activity level
        
        # Calculate current activity level
        if hasattr(self, 'activity_stats') and 'neuron_activity' in self.activity_stats:
            neuron_activity = self.activity_stats['neuron_activity']
        else:
            # Use a simpler measure if activity stats not available
            if hasattr(self, 'active_neurons_cache'):
                active_ratio = len(self.active_neurons_cache) / self.neuron_count
                neuron_activity = np.zeros(self.neuron_count)
                neuron_activity[list(self.active_neurons_cache)] = active_ratio
            else:
                # Fall back to membrane potentials
                active_threshold = 0.5
                active_neurons = self.membrane_potentials > active_threshold
                neuron_activity = active_neurons.astype(float) * 0.1  # Approximate activity level
        
        # Calculate deviation from target
        activity_deviation = neuron_activity - target_activity
        
        # Identify neurons needing adjustment (more than 5% from target)
        adjust_threshold = 0.05
        hyperactive_neurons = np.where(activity_deviation > adjust_threshold)[0]
        hypoactive_neurons = np.where(activity_deviation < -adjust_threshold)[0]
        
        # Adjustment rate (smaller for gradual adjustment)
        adjustment_rate = 0.01
        
        # Adjust hyperactive neurons - decrease incoming weights
        if len(hyperactive_neurons) > 0:
            # For computational efficiency, adjust a random subset of incoming connections
            num_adjustments = min(1000, len(hyperactive_neurons) * 10)
            for _ in range(num_adjustments):
                # Choose a random hyperactive neuron
                neuron_idx = random.choice(hyperactive_neurons)
                
                # Skip if this neuron is managed by custom learning
                if neuron_idx in self.custom_learning_neurons:
                    continue
                
                # Choose a random incoming connection
                if sparse.issparse(self.synaptic_weights):
                    # For sparse matrix, find non-zero inputs
                    inputs = np.array(self.synaptic_weights[:, neuron_idx].nonzero()[0])
                    if len(inputs) > 0:
                        input_idx = random.choice(inputs)
                        # Decrease weight
                        current_weight = self.synaptic_weights[input_idx, neuron_idx]
                        self.synaptic_weights[input_idx, neuron_idx] = current_weight * (1.0 - adjustment_rate)
                else:
                    # For dense matrix, choose any neuron
                    input_idx = random.randint(0, self.neuron_count-1)
                    # Decrease weight if non-zero
                    current_weight = self.synaptic_weights[input_idx, neuron_idx]
                    if current_weight > 0:
                        self.synaptic_weights[input_idx, neuron_idx] = current_weight * (1.0 - adjustment_rate)
        
        # Adjust hypoactive neurons - increase incoming weights
        if len(hypoactive_neurons) > 0:
            # For computational efficiency, adjust a random subset of incoming connections
            num_adjustments = min(1000, len(hypoactive_neurons) * 10)
            for _ in range(num_adjustments):
                # Choose a random hypoactive neuron
                neuron_idx = random.choice(hypoactive_neurons)
                
                # Skip if this neuron is managed by custom learning
                if neuron_idx in self.custom_learning_neurons:
                    continue
                
                # Choose a random incoming connection
                if sparse.issparse(self.synaptic_weights):
                    # For sparse matrix, find non-zero inputs
                    inputs = np.array(self.synaptic_weights[:, neuron_idx].nonzero()[0])
                    if len(inputs) > 0:
                        input_idx = random.choice(inputs)
                        # Increase weight
                        current_weight = self.synaptic_weights[input_idx, neuron_idx]
                        self.synaptic_weights[input_idx, neuron_idx] = min(1.0, current_weight * (1.0 + adjustment_rate))
                else:
                    # For dense matrix, choose any neuron with non-zero weight
                    non_zero_inputs = np.where(self.synaptic_weights[:, neuron_idx] > 0)[0]
                    if len(non_zero_inputs) > 0:
                        input_idx = random.choice(non_zero_inputs)
                        # Increase weight
                        current_weight = self.synaptic_weights[input_idx, neuron_idx]
                        self.synaptic_weights[input_idx, neuron_idx] = min(1.0, current_weight * (1.0 + adjustment_rate))

    def store_text_memory(self, text_input, memory_key=None):
        """
        Store text as a memory in the network.
        
        Args:
            text_input: Text to store as memory
            memory_key: Optional key for the memory (if None, generated from text)
            
        Returns:
            memory_key: The key used to store the memory
        """
        # Convert text to spikes using bidirectional processor
        spike_patterns = self.process_text_input(text_input)
        
        # Generate a key from the text if not provided
        if memory_key is None:
            memory_key = f"text_{hash(text_input) % 10000000}"
        
        # Use the spike patterns to create a memory engram
        content = {"text": text_input, "type": "text_memory"}
        self.encode_memory(spike_patterns, content=content, memory_type='episodic')
        
        return memory_key

    def retrieve_text_memory(self, query_text=None, memory_key=None):
        """
        Retrieve text memory using text query or memory key.
        
        Args:
            query_text: Text query to use for memory retrieval
            memory_key: Specific memory key to retrieve
            
        Returns:
            List of retrieved text memories ordered by relevance
        """
        if query_text:
            # Convert query text to spikes using bidirectional processor
            query_spikes = self.process_text_input(query_text)
            
            # Use the spikes to retrieve memories
            retrieved = self.retrieve_memory(query_spikes, max_results=5)
        elif memory_key:
            # Directly retrieve by key
            if memory_key in self.memory_engrams:
                content = self.memory_engrams[memory_key].get('content', {})
                if content and content.get('type') == 'text_memory':
                    return [content.get('text', '')]
            return []
        else:
            return []
        
        # Extract text from retrieved memories
        results = []
        for mem in retrieved:
            content = mem.get('content', {})
            if content and content.get('type') == 'text_memory':
                results.append(content.get('text', ''))
        
        return results

    def summarize_text_memories(self):
        """
        Generate a summary of all text memories stored in the network.
        
        Returns:
            Dictionary mapping memory keys to short summaries
        """
        summaries = {}
        
        for key, engram in self.memory_engrams.items():
            content = engram.get('content', {})
            if content and content.get('type') == 'text_memory':
                text = content.get('text', '')
                # Generate a short summary (first 50 chars)
                summary = text[:50] + '...' if len(text) > 50 else text
                summaries[key] = summary
        
        return summaries

    def find_similar_memory(self, query_activation, threshold=0.6):
        """
        Find memories that are similar to the query activation pattern.
        
        Args:
            query_activation: Activation pattern to compare against memories
            threshold: Similarity threshold for matching (0.0-1.0)
            
        Returns:
            List of similar memories with similarity scores
        """
        results = []
        
        # Convert query to proper format if needed
        if isinstance(query_activation, str):
            query_activation = self.process_text_input(query_activation)
        
        # Calculate network state from query
        query_state = self._get_network_state(query_activation)
        
        # Compare query with all stored memories
        for key, engram in self.memory_engrams.items():
            pattern = engram.get('pattern', [])
            if not pattern:
                continue
                
            # Calculate similarity score
            similarity = self._calculate_pattern_similarity(query_state, pattern)
            
            if similarity >= threshold:
                results.append({
                    'key': key,
                    'similarity': similarity,
                    'content': engram.get('content', {})
                })
        
        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results

# MemoryTokenizer has been removed; using BidirectionalProcessor instead