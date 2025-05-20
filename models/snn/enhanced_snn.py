# enhanced_snn.py
import numpy as np
import random
import time
import math
from collections import defaultdict, deque
from scipy import sparse
import logging
import re
import torch

# Import bidirectional processing components
from models.snn.bidirectional_encoding import BidirectionalProcessor, create_processor

class EnhancedSpikingCore:
    """
    Advanced Spiking Neural Network with improved architecture and learning capabilities.
    
    Features:
    - Enhanced vectorization using word embeddings
    - Flexible network topologies
    - Sophisticated learning mechanisms
    - Optimized parameters through network analysis
    - Improved hypothesis generation with domain knowledge integration
    - GPU acceleration for tensor operations
    - Integrated bidirectional text processing for text ↔ spikes conversion
    """
    def __init__(self, neuron_count=1000, topology_type="flexible", 
             connection_density=0.1, decay_rate=0.95, plasticity_rate=0.01,
             word_embedding_file=None, device=None, 
             bidirectional_processor=None, model_type="generic", 
             vector_dim=300):
        """
        Initialize the enhanced SNN with configurable parameters.
        
        Args:
            neuron_count: Total number of neurons
            topology_type: Network topology type ("flexible", "modular", "small_world", "scale_free")
            connection_density: Initial connection density
            decay_rate: Membrane potential decay rate
            plasticity_rate: Learning rate for synaptic plasticity
            word_embedding_file: Path to pre-trained word embeddings
            device: Device to run computations on ('cpu', 'cuda', 'cuda:0', etc. or None for auto-detection)
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
            model_type: Type of SNN model for specific processing (e.g., "memory", "perceptual", "reasoning")
            vector_dim: Dimension of the vector space for token embeddings
        """
        # Set device for GPU acceleration
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        self.neuron_count = neuron_count
        self.topology_type = topology_type
        self.connection_density = connection_density
        self.decay_rate = decay_rate
        self.plasticity_rate = plasticity_rate
        self.model_type = model_type
        
        # Initialize bidirectional processor for text ↔ spikes conversion
        if bidirectional_processor is not None:
            self.bidirectional_processor = bidirectional_processor
        else:
            self.bidirectional_processor = create_processor(
                model_type=self.model_type,
                vector_dim=vector_dim,
                device=self.device
            )
        
        # Neural parameters
        self.base_threshold = 0.5  # Base spike threshold (constant component)
        self.spike_threshold = self.base_threshold  # Current spike threshold (will be replaced by array)
        self.refractory_period = 0.3
        
        # Initialize tensors on selected device
        self.membrane_potentials = torch.zeros(neuron_count, device=self.device)
        self.refractory_counters = torch.zeros(neuron_count, device=self.device)
        
        # Initialize plastic thresholds array
        self.plastic_thresholds = torch.zeros(neuron_count, device=self.device)
        
        # Initialize base thresholds array (initially all same value)
        self.base_thresholds = torch.full((neuron_count,), self.base_threshold, device=self.device)
        
        # Activity tracking
        self.spike_history = defaultdict(lambda: deque(maxlen=100))
        self.active_neurons_cache = set()
        self.activity_record = torch.zeros(neuron_count, device=self.device)
        self.last_spike_times = torch.full((neuron_count,), -1000.0, device=self.device)
        
        # Initialize word embeddings from local file
        if word_embedding_file:
            try:
                self.word_vectors = self._load_word_vectors(word_embedding_file)
                self.embedding_dim = len(next(iter(self.word_vectors.values())))
                self.has_embeddings = True
                logging.info(f"Loaded word embeddings with {self.embedding_dim} dimensions")
            except Exception as e:
                logging.warning(f"Failed to load word embeddings: {e}")
                self.has_embeddings = False
                self.embedding_dim = 100
        else:
            self.has_embeddings = False
            self.embedding_dim = 100

        # Add this line to initialize the set for specialized learning synapses
        self.specialized_learning_synapses = set()
        self.custom_learning_neurons = set()  # Track neurons with specialized learning

        # Adaptive learning parameters
        self.learning_parameters = {
            'stdp_window': 20,  # ms, temporal window for STDP
            'hebbian_rate': 0.02,  # Rate for Hebbian learning
            'homeostatic_rate': 0.005,  # Rate for homeostatic plasticity
            'reinforcement_rate': 0.03,  # Rate for reinforcement learning
            'confidence_baseline': 0.5,  # Baseline for confidence estimation
            'adaptation_rate': 0.01  # Rate for parameter adaptation
        }
        
        # Performance metrics
        self.performance_metrics = {
            'accuracy_history': [],
            'loss_history': [],
            'phi_history': [],
            'concept_activation_precision': defaultdict(list),
            'hypothesis_quality': []
        }
        
        # Initialize network topology and components
        self._init_topology()
        self.synaptic_weights = self._init_weights()
        self.concept_mappings = self._init_concept_mappings()
        self._init_learning_module()
        
        # IIT metrics
        self.phi = 0.0
        self.differentiation = 0.0
        self.integration = 0.0
        
        # Domain knowledge for improved hypothesis generation
        self.domain_knowledge = self._init_domain_knowledge()
        
        logging.info(f"Enhanced SNN initialized with {neuron_count} neurons, {topology_type} topology on {self.device}")

    def _apply_threshold_plasticity(self, neuron_indices, error_signal, learning_rate=0.01):
        """
        Apply threshold plasticity to specified neurons based on error signal.
        This method updates the plastic component of the spike threshold for neurons,
        allowing dynamic adjustment based on supervised learning signals.
        
        Args:
            neuron_indices: Array/list of neuron indices to update
            error_signal: Error signal from supervised learning (can be scalar or array)
                        Positive error suggests threshold should decrease (neuron should fire more)
                        Negative error suggests threshold should increase (neuron should fire less)
            learning_rate: Learning rate for threshold updates
        """
        # Convert to torch tensor if needed
        if not isinstance(neuron_indices, torch.Tensor):
            if isinstance(neuron_indices, np.ndarray):
                neuron_indices = torch.tensor(neuron_indices, device=self.device)
            else:
                neuron_indices = torch.tensor(neuron_indices, device=self.device)
        
        # Ensure we only update neurons with custom learning enabled
        filtered_indices = []
        for idx in neuron_indices.cpu().numpy():
            if idx in self.custom_learning_neurons:
                filtered_indices.append(idx)
        
        if not filtered_indices:
            return 0  # No neurons to update
        
        filtered_indices = torch.tensor(filtered_indices, device=self.device)
        
        # Handle scalar vs array error signal
        if np.isscalar(error_signal):
            # Broadcast scalar to all neurons
            weight_change = -learning_rate * error_signal  # Negative sign since positive error should decrease threshold
        else:
            # Ensure error signal matches number of neurons
            if len(error_signal) != len(filtered_indices):
                raise ValueError(f"Error signal length {len(error_signal)} does not match neuron count {len(filtered_indices)}")
            
            # Convert error signal to tensor if needed
            if not isinstance(error_signal, torch.Tensor):
                error_signal = torch.tensor(error_signal, device=self.device)
                
            weight_change = -learning_rate * error_signal
        
        # Update plastic thresholds
        old_thresholds = self.plastic_thresholds[filtered_indices].clone()
        self.plastic_thresholds[filtered_indices] += weight_change
        
        # Keep plastic thresholds within reasonable bounds
        # Plastic component can be negative (making neuron easier to fire) or positive (harder to fire)
        self.plastic_thresholds[filtered_indices] = torch.clamp(
            self.plastic_thresholds[filtered_indices], 
            -0.3,  # Minimum plastic threshold (makes base threshold effectively lower)
            0.5    # Maximum plastic threshold (makes base threshold effectively higher)
        )
        
        # Track how many thresholds were actually updated
        updates_made = torch.sum((self.plastic_thresholds[filtered_indices] - old_thresholds).abs() > 1e-6).item()
        
        return updates_made

    def get_effective_thresholds(self):
        """
        Get the effective spike thresholds (base + plastic) for all neurons.
        
        Returns:
            torch.Tensor: Tensor of effective thresholds for each neuron
        """
        return self.base_thresholds + self.plastic_thresholds

    def reset_plastic_thresholds(self):
        """
        Reset all plastic thresholds to zero.
        Useful for starting fresh training or evaluation.
        """
        self.plastic_thresholds = torch.zeros(self.neuron_count, device=self.device)
    
    def _init_topology(self):
        """Initialize the network topology based on selected type"""
        if self.topology_type == "flexible":
            self._init_flexible_topology()
        elif self.topology_type == "modular":
            self._init_modular_topology()
        elif self.topology_type == "small_world":
            self._init_small_world_topology()
        elif self.topology_type == "scale_free":
            self._init_scale_free_topology()
        else:
            logging.warning(f"Unknown topology type: {self.topology_type}, falling back to flexible")
            self._init_flexible_topology()

    def _init_flexible_topology(self):
        """
        Initialize a flexible topology with adaptive regions and connectivity.
        Similar to brain functional areas with distributed processing.
        """
        # Define functional regions based on cognitive processes
        region_proportions = {
            'sensory': 0.20,           # Perception and input processing
            'memory': 0.20,            # Working memory and storage
            'higher_cognition': 0.25,  # Conceptual, planning, decision functions
            'affective': 0.15,         # Emotional processing and evaluation
            'metacognition': 0.10,     # Self-reflection and monitoring
            'output': 0.10             # Action and response
        }
        
        # Allocate neurons to regions
        self.regions = {}
        start_idx = 0
        for region_name, proportion in region_proportions.items():
            size = int(self.neuron_count * proportion)
            self.regions[region_name] = {
                'neurons': list(range(start_idx, start_idx + size)),
                'activation': 0.0,
                'recurrent': True if region_name != 'output' else False,
                'plasticity_factor': 1.0,  # Default plasticity factor
                'inhibitory_ratio': 0.2 if region_name in ['sensory', 'output'] else 0.3,
                'adaptation_rate': self.plasticity_rate * (1.2 if region_name == 'metacognition' else 1.0)
            }
            start_idx += size
        
        # Define connectivity between regions
        self.region_connectivity = {
            'sensory': ['memory', 'higher_cognition', 'affective'],
            'memory': ['sensory', 'higher_cognition', 'metacognition', 'affective'],
            'higher_cognition': ['sensory', 'memory', 'affective', 'metacognition', 'output'],
            'affective': ['sensory', 'memory', 'higher_cognition', 'output'],
            'metacognition': ['memory', 'higher_cognition', 'output'],
            'output': ['higher_cognition', 'affective', 'metacognition']
        }
        
        # Define sub-assemblies within regions (less rigid than hardcoded approach)
        self.assemblies = {}
        
        # Create concept interface for symbolic grounding
        self.symbolic_layer = {
            'neurons': self.regions['higher_cognition']['neurons'][:100],  # Symbolic representation
            'feedback_neurons': self.regions['memory']['neurons'][:50]     # For symbolic feedback
        }
        
        # Create composite representation units
        self.composite_units = {
            'concept_binding': self.regions['higher_cognition']['neurons'][100:130],  # For binding concepts
            'temporal_context': self.regions['memory']['neurons'][50:80],  # For temporal context
            'relation_encoding': self.regions['higher_cognition']['neurons'][130:160]  # For encoding relations
        }

    def _init_modular_topology(self):
        """
        Initialize a modular network topology with densely connected modules
        that are sparsely connected with each other.
        """
        # Define modules (similar to cortical columns)
        num_modules = 10
        neurons_per_module = self.neuron_count // num_modules
        
        self.modules = []
        self.regions = {}
        
        # Create modules
        for i in range(num_modules):
            start_idx = i * neurons_per_module
            end_idx = start_idx + neurons_per_module
            
            module = {
                'id': i,
                'neurons': list(range(start_idx, end_idx)),
                'activation': 0.0,
                'connected_modules': [],  # Will be filled later
                'within_density': 0.3,  # Dense within-module connectivity
                'inhibitory_ratio': 0.2,  # 20% inhibitory neurons
                'plasticity_factor': 1.0
            }
            
            self.modules.append(module)
        
        # Assign functional roles to modules
        module_functions = [
            'sensory_input', 'feature_extraction', 'working_memory', 
            'semantic_processing', 'emotional_evaluation', 'decision_making',
            'attentional_control', 'error_detection', 'prediction', 'motor_output'
        ]
        
        # Create regions based on module functions
        for i, function in enumerate(module_functions):
            self.regions[function] = {
                'neurons': self.modules[i]['neurons'],
                'activation': 0.0,
                'recurrent': True if function not in ['sensory_input', 'motor_output'] else False,
                'plasticity_factor': 1.2 if function in ['working_memory', 'semantic_processing'] else 1.0
            }
        
        # Create sparse connections between modules (small-world-like)
        for i, module in enumerate(self.modules):
            # Connect to neighboring modules
            neighbors = [(i-1) % num_modules, (i+1) % num_modules]
            
            # Add a few random long-range connections
            long_range = [j for j in range(num_modules) if j not in neighbors and j != i]
            random_connections = random.sample(long_range, min(2, len(long_range)))
            
            module['connected_modules'] = neighbors + random_connections
        
        # Create symbolic interface
        self.symbolic_layer = {
            'neurons': self.regions['semantic_processing']['neurons'][:30] + 
                      self.regions['working_memory']['neurons'][:20],
            'feedback_neurons': self.regions['attentional_control']['neurons'][:20] +
                               self.regions['prediction']['neurons'][:20]
        }
        
        # Create composite representations
        self.composite_units = {
            'concept_binding': self.regions['semantic_processing']['neurons'][30:50],
            'temporal_context': self.regions['working_memory']['neurons'][20:40],
            'relation_encoding': self.regions['prediction']['neurons'][20:40]
        }

    def _init_small_world_topology(self):
        """
        Initialize a small-world network topology with high clustering
        and short path lengths (based on Watts-Strogatz model).
        """
        # Create ring lattice first
        k = 10  # Each neuron initially connected to k nearest neighbors
        self.small_world_connections = []
        
        for i in range(self.neuron_count):
            for j in range(1, k // 2 + 1):
                # Connect to k/2 neighbors on each side (ring topology)
                self.small_world_connections.append((i, (i + j) % self.neuron_count))
                self.small_world_connections.append((i, (i - j) % self.neuron_count))
        
        # Rewire connections with probability p
        p = 0.1  # Rewiring probability
        for i, (src, dst) in enumerate(self.small_world_connections):
            if random.random() < p:
                # Rewire to random neuron (avoiding self-loops and duplicates)
                new_dst = random.randint(0, self.neuron_count - 1)
                while new_dst == src or (src, new_dst) in self.small_world_connections:
                    new_dst = random.randint(0, self.neuron_count - 1)
                self.small_world_connections[i] = (src, new_dst)
        
        # Create artificial regions based on neuron position in ring
        region_size = self.neuron_count // 6
        self.regions = {}
        
        region_names = ['sensory', 'processing', 'memory', 'integration', 'decision', 'output']
        
        for i, name in enumerate(region_names):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size
            self.regions[name] = {
                'neurons': list(range(start_idx, end_idx)),
                'activation': 0.0,
                'recurrent': True if name not in ['sensory', 'output'] else False,
                'plasticity_factor': 1.0
            }
        
        # Define region connectivity (simpler in small-world since neurons are all connected)
        self.region_connectivity = {
            'sensory': ['processing', 'memory'],
            'processing': ['sensory', 'memory', 'integration'],
            'memory': ['processing', 'integration', 'decision'],
            'integration': ['processing', 'memory', 'decision'],
            'decision': ['memory', 'integration', 'output'],
            'output': ['decision']
        }
        
        # Create symbolic interface
        self.symbolic_layer = {
            'neurons': self.regions['integration']['neurons'][:50] + 
                      self.regions['memory']['neurons'][:50],
            'feedback_neurons': self.regions['decision']['neurons'][:50]
        }

    def _init_scale_free_topology(self):
        """
        Initialize a scale-free network topology where connectivity follows
        a power law distribution (based on Barabási–Albert model).
        """
        # Initialize with a small complete graph
        m0 = 5  # Initial number of nodes
        m = 3   # Number of edges to attach from a new node to existing nodes
        
        # Initial complete graph
        self.scale_free_connections = [(i, j) for i in range(m0) for j in range(i+1, m0)]
        
        # List to track the degrees of each node for preferential attachment
        node_degrees = [m0-1] * m0  # Initially all nodes in complete graph have same degree
        
        # Add remaining nodes with preferential attachment
        for i in range(m0, self.neuron_count):
            # Extend node_degrees list to include the new node with initial degree 0
            node_degrees.append(0)
            
            # Select m existing nodes with probability proportional to their degree
            targets = []
            for _ in range(m):
                # Calculate total degree
                total_degree = sum(node_degrees)
                
                # Select a target node based on degree
                target = None
                r = random.uniform(0, total_degree)
                cumulative = 0
                for j, degree in enumerate(node_degrees):
                    cumulative += degree
                    if cumulative >= r and j not in targets:
                        target = j
                        break
                
                if target is not None:
                    targets.append(target)
                    self.scale_free_connections.append((i, target))
                    # Update node degrees
                    node_degrees[i] += 1
                    node_degrees[target] += 1
        
        # Identify hub neurons (top 5% by degree)
        node_degrees_dict = defaultdict(int)
        for src, dst in self.scale_free_connections:
            node_degrees_dict[src] += 1
            node_degrees_dict[dst] += 1
        
        sorted_nodes = sorted(node_degrees_dict.items(), key=lambda x: x[1], reverse=True)
        hub_count = int(self.neuron_count * 0.05)
        self.hub_neurons = [node for node, _ in sorted_nodes[:hub_count]]
        
        # Create functional regions based on structural properties
        # Hub regions (integration centers)
        hub_region_neurons = []
        for hub in self.hub_neurons:
            hub_region_neurons.extend([hub])
            # Add neurons directly connected to hubs
            for src, dst in self.scale_free_connections:
                if src == hub:
                    hub_region_neurons.append(dst)
                elif dst == hub:
                    hub_region_neurons.append(src)
        
        hub_region_neurons = list(set(hub_region_neurons))  # Remove duplicates
        
        # Remaining neurons distributed across other regions
        remaining_neurons = [n for n in range(self.neuron_count) if n not in hub_region_neurons]
        region_size = len(remaining_neurons) // 5
        
        self.regions = {
            'hub_integration': {
                'neurons': hub_region_neurons,
                'activation': 0.0,
                'recurrent': True,
                'plasticity_factor': 1.2  # Higher plasticity for hub regions
            }
        }
        
        # Add other regions
        region_names = ['sensory', 'processing', 'memory', 'decision', 'output']
        for i, name in enumerate(region_names):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size if i < len(region_names) - 1 else len(remaining_neurons)
            
            self.regions[name] = {
                'neurons': remaining_neurons[start_idx:end_idx],
                'activation': 0.0,
                'recurrent': True if name not in ['sensory', 'output'] else False,
                'plasticity_factor': 0.9  # Lower plasticity for non-hub regions
            }
        
        # Define region connectivity
        self.region_connectivity = {
            'hub_integration': ['sensory', 'processing', 'memory', 'decision', 'output'],
            'sensory': ['hub_integration', 'processing'],
            'processing': ['sensory', 'hub_integration', 'memory'],
            'memory': ['processing', 'hub_integration', 'decision'],
            'decision': ['memory', 'hub_integration', 'output'],
            'output': ['decision', 'hub_integration']
        }
        
        # Create symbolic interface using hub neurons
        self.symbolic_layer = {
            'neurons': self.hub_neurons[:20] + self.regions['memory']['neurons'][:30],
            'feedback_neurons': self.hub_neurons[20:40] + self.regions['decision']['neurons'][:10]
        }

    def _init_weights(self):
        """Initialize synaptic weights based on network topology"""
        # Create sparse matrix representation for efficiency
        rows = []
        cols = []
        data = []
        
        if self.topology_type == "flexible":
            # Set up weights based on regions
            for source_region, targets in self.region_connectivity.items():
                source_neurons = self.regions[source_region]['neurons']
                
                for target_region in targets:
                    target_neurons = self.regions[target_region]['neurons']
                    
                    # Determine connection density between regions
                    region_density = self.connection_density * 0.8
                    num_connections = int(len(source_neurons) * len(target_neurons) * region_density)
                    
                    # Add random connections
                    for _ in range(num_connections):
                        src = random.choice(source_neurons)
                        dst = random.choice(target_neurons)
                        
                        # Determine if inhibitory (20% of connections)
                        is_inhibitory = random.random() < 0.2
                        weight = -0.5 - 0.3 * random.random() if is_inhibitory else 0.1 + 0.2 * random.random()
                        
                        rows.append(src)
                        cols.append(dst)
                        data.append(weight)
            
            # Add recurrent connections within regions
            for region_name, region in self.regions.items():
                if region.get('recurrent', False):
                    neurons = region['neurons']
                    # Higher density for recurrent connections
                    num_connections = int(len(neurons) * len(neurons) * 0.15)
                    
                    for _ in range(num_connections):
                        src = random.choice(neurons)
                        dst = random.choice(neurons)
                        if src != dst:  # Avoid self-connections
                            is_inhibitory = random.random() < 0.3  # Higher inhibition in recurrent connections
                            weight = -0.3 - 0.2 * random.random() if is_inhibitory else 0.05 + 0.1 * random.random()
                            
                            rows.append(src)
                            cols.append(dst)
                            data.append(weight)
            
            # Ensure symbolic layer has connectivity
            symbolic_neurons = self.symbolic_layer['neurons']
            feedback_neurons = self.symbolic_layer['feedback_neurons']
            
            for s_neuron in symbolic_neurons:
                for region_name in ['higher_cognition', 'memory']:
                    if region_name in self.regions:
                        for p_neuron in random.sample(self.regions[region_name]['neurons'], 
                                                    min(30, len(self.regions[region_name]['neurons']))):
                            weight = 0.2 + 0.3 * random.random()
                            rows.append(s_neuron)
                            cols.append(p_neuron)
                            data.append(weight)
                            
                            # Bidirectional connections
                            rows.append(p_neuron)
                            cols.append(s_neuron)
                            data.append(weight * 0.8)
            
            # Connect feedback neurons
            for f_neuron in feedback_neurons:
                for p_neuron in random.sample(list(range(self.neuron_count)), 50):
                    weight = 0.3 + 0.3 * random.random()
                    rows.append(f_neuron)
                    cols.append(p_neuron)
                    data.append(weight)
            
        elif self.topology_type == "modular":
            # Set up intra-module connections (dense)
            for module in self.modules:
                neurons = module['neurons']
                # Calculate number of connections based on density
                within_connections = int(len(neurons) * len(neurons) * module['within_density'])
                
                for _ in range(within_connections):
                    src = random.choice(neurons)
                    dst = random.choice(neurons)
                    if src != dst:  # Avoid self-connections
                        is_inhibitory = random.random() < module['inhibitory_ratio']
                        weight = -0.5 - 0.3 * random.random() if is_inhibitory else 0.2 + 0.4 * random.random()
                        
                        rows.append(src)
                        cols.append(dst)
                        data.append(weight)
            
            # Set up inter-module connections (sparse)
            for i, module in enumerate(self.modules):
                for connected_module_id in module['connected_modules']:
                    source_neurons = module['neurons']
                    target_neurons = self.modules[connected_module_id]['neurons']
                    
                    # Sparse connectivity between modules
                    num_connections = int(len(source_neurons) * 0.05)
                    
                    for _ in range(num_connections):
                        src = random.choice(source_neurons)
                        dst = random.choice(target_neurons)
                        weight = 0.1 + 0.1 * random.random()  # Weaker inter-module connections
                        
                        rows.append(src)
                        cols.append(dst)
                        data.append(weight)
            
            # Add symbolic layer connections
            for s_neuron in self.symbolic_layer['neurons']:
                target_modules = [2, 3]  # Connect to working_memory and semantic_processing
                for module_id in target_modules:
                    target_neurons = self.modules[module_id]['neurons']
                    for dst in random.sample(target_neurons, min(10, len(target_neurons))):
                        weight = 0.2 + 0.2 * random.random()
                        rows.append(s_neuron)
                        cols.append(dst)
                        data.append(weight)
        
        elif self.topology_type == "small_world":
            # Initialize weights from small-world connections
            for src, dst in self.small_world_connections:
                is_inhibitory = random.random() < 0.2
                weight = -0.4 - 0.3 * random.random() if is_inhibitory else 0.1 + 0.3 * random.random()
                
                rows.append(src)
                cols.append(dst)
                data.append(weight)
            
            # Add symbolic layer connections
            for s_neuron in self.symbolic_layer['neurons']:
                for region_name in ['integration', 'memory']:
                    if region_name in self.regions:
                        for dst in random.sample(self.regions[region_name]['neurons'], 
                                                min(20, len(self.regions[region_name]['neurons']))):
                            weight = 0.2 + 0.2 * random.random()
                            rows.append(s_neuron)
                            cols.append(dst)
                            data.append(weight)
        
        elif self.topology_type == "scale_free":
            # Initialize weights from scale-free connections
            for src, dst in self.scale_free_connections:
                is_hub_connection = src in self.hub_neurons or dst in self.hub_neurons
                
                # Hub connections are stronger
                if is_hub_connection:
                    is_inhibitory = random.random() < 0.15  # Less inhibition for hubs
                    weight = -0.4 - 0.3 * random.random() if is_inhibitory else 0.2 + 0.4 * random.random()
                else:
                    is_inhibitory = random.random() < 0.2
                    weight = -0.3 - 0.2 * random.random() if is_inhibitory else 0.1 + 0.2 * random.random()
                
                rows.append(src)
                cols.append(dst)
                data.append(weight)
                
                # Make bidirectional (with slightly different weight)
                weight_back = weight * (0.8 + 0.4 * random.random())
                rows.append(dst)
                cols.append(src)
                data.append(weight_back)
            
            # Add symbolic layer connections focused on hub neurons
            for s_neuron in self.symbolic_layer['neurons']:
                # Connect to hub neurons for efficient information distribution
                for hub in self.hub_neurons:
                    weight = 0.3 + 0.3 * random.random()
                    rows.append(s_neuron)
                    cols.append(hub)
                    data.append(weight)
                    
                    # Bidirectional connections
                    rows.append(hub)
                    cols.append(s_neuron)
                    data.append(weight * 0.9)
        
        # Create sparse matrix from coordinates
        weights = sparse.lil_matrix((self.neuron_count, self.neuron_count))
        
        # Add data directly to lil_matrix for better efficiency
        for i in range(len(rows)):
            weights[rows[i], cols[i]] = data[i]
            
        return weights

    def _init_concept_mappings(self):
        """Initialize concept-to-neuron mappings using word embeddings when available"""
        # Define key concepts we want to represent
        key_concepts = [
            "trust", "distrust", "fear", "like", "dislike",
            "know", "learn", "remember", "forget", "understand",
            "person", "animal", "object", "place", "concept",
            "happy", "sad", "angry", "surprised", "disgusted",
            "cause", "effect", "enable", "prevent", "allow"
        ]
        
        concepts = {}
        
        # Determine concept size based on neuron count
        concept_size = max(20, min(50, self.neuron_count // 100))
        
        # Mapping strategy varies by topology
        if self.topology_type == "flexible":
            # Map concepts to appropriate regions
            concept_regions = {
                "trust": "affective", "distrust": "affective", "fear": "affective", 
                "like": "affective", "dislike": "affective", "happy": "affective", 
                "sad": "affective", "angry": "affective", "surprised": "affective", 
                "disgusted": "affective",
                
                "know": "memory", "learn": "memory", "remember": "memory", 
                "forget": "memory", "understand": "higher_cognition",
                
                "person": "higher_cognition", "animal": "higher_cognition", 
                "object": "higher_cognition", "place": "higher_cognition", 
                "concept": "higher_cognition",
                
                "cause": "higher_cognition", "effect": "higher_cognition", 
                "enable": "higher_cognition", "prevent": "higher_cognition", 
                "allow": "higher_cognition"
            }
            
            # For each concept, assign neurons from appropriate regions
            for concept in key_concepts:
                region = concept_regions.get(concept, "higher_cognition")
                if region in self.regions:
                    region_neurons = self.regions[region]['neurons']
                    concepts[concept] = random.sample(region_neurons, min(concept_size, len(region_neurons)))
                    
                    # Include some symbolic neurons
                    symbolic_neurons = random.sample(self.symbolic_layer['neurons'], 
                                                    min(concept_size // 4, len(self.symbolic_layer['neurons'])))
                    concepts[concept].extend(symbolic_neurons)
                else:
                    # Fall back to random neurons if region doesn't exist
                    concepts[concept] = random.sample(range(self.neuron_count), concept_size)
        
        elif self.topology_type == "modular":
            # Map concepts to appropriate functional modules
            concept_modules = {
                # Emotional concepts to emotional_evaluation module
                "trust": 4, "distrust": 4, "fear": 4, "like": 4, "dislike": 4,
                "happy": 4, "sad": 4, "angry": 4, "surprised": 4, "disgusted": 4,
                
                # Knowledge concepts to semantic_processing and working_memory modules
                "know": 3, "learn": 3, "remember": 2, "forget": 2, "understand": 3,
                
                # Object concepts to semantic_processing
                "person": 3, "animal": 3, "object": 3, "place": 3, "concept": 3,
                
                # Causal concepts to prediction module
                "cause": 8, "effect": 8, "enable": 8, "prevent": 8, "allow": 8
            }
            
            for concept in key_concepts:
                module_id = concept_modules.get(concept, 3)  # Default to semantic_processing
                if module_id < len(self.modules):
                    module_neurons = self.modules[module_id]['neurons']
                    concepts[concept] = random.sample(module_neurons, min(concept_size, len(module_neurons)))
                else:
                    concepts[concept] = random.sample(range(self.neuron_count), concept_size)
        
        elif self.topology_type in ["small_world", "scale_free"]:
            # For less structured topologies, use hub neurons for important concepts
            
            # For scale-free, use hub neurons
            if self.topology_type == "scale_free" and hasattr(self, 'hub_neurons'):
                hub_concepts = ["trust", "know", "understand", "concept", "cause"]
                for concept in hub_concepts:
                    # Assign both hub neurons and random neurons
                    # Assign both hub neurons and random neurons
                    hub_sample = random.sample(self.hub_neurons, min(concept_size // 2, len(self.hub_neurons)))
                    random_sample = random.sample(range(self.neuron_count), concept_size // 2)
                    concepts[concept] = hub_sample + random_sample
            
            # For non-hub concepts, use more distributed representations
            for concept in key_concepts:
                if concept not in concepts:
                    # Choose neurons from different regions
                    concept_neurons = []
                    for region_name in self.regions:
                        region_neurons = self.regions[region_name]['neurons']
                        region_sample = random.sample(region_neurons, min(concept_size // 6, len(region_neurons)))
                        concept_neurons.extend(region_sample)
                    
                    # Ensure we have enough neurons
                    if len(concept_neurons) < concept_size:
                        additional = random.sample(range(self.neuron_count), concept_size - len(concept_neurons))
                        concept_neurons.extend(additional)
                    
                    concepts[concept] = concept_neurons
        
        # Organize concept mappings by similarity if word vectors are available
        if self.has_embeddings:
            # Create distributed representations based on word vector similarity
            for concept in key_concepts:
                if concept in self.word_vectors:
                    # Get similar concepts from embedding model
                    try:
                        similar_concepts = [word for word, _ in self.word_vectors.most_similar(concept, topn=5)]
                        
                        # For concepts with similar words, share some neurons
                        for similar in similar_concepts:
                            if similar in concepts:
                                # Share ~20% of neurons between similar concepts
                                shared_neurons = random.sample(concepts[similar], min(concept_size // 5, len(concepts[similar])))
                                if concept in concepts:
                                    # Replace some neurons with shared ones
                                    concepts[concept] = concepts[concept][:-len(shared_neurons)] + shared_neurons
                    except:
                        # Word might not be in vocabulary
                        pass
        
        # Add composite concept capabilities
        self.composite_mappings = {}
        
        # For each pair of basic concepts, create a composite mapping
        for concept1 in key_concepts[:10]:  # Limit to first 10 for efficiency
            for concept2 in key_concepts[:10]:
                if concept1 != concept2:
                    composite_name = f"{concept1}_{concept2}"
                    
                    # Create composite mapping from individual concept neurons plus binding neurons
                    if concept1 in concepts and concept2 in concepts:
                        # Take subset of neurons from each concept
                        neurons1 = random.sample(concepts[concept1], min(10, len(concepts[concept1])))
                        neurons2 = random.sample(concepts[concept2], min(10, len(concepts[concept2])))
                        
                        # Add binding neurons if available
                        binding_neurons = []
                        if hasattr(self, 'composite_units') and 'concept_binding' in self.composite_units:
                            binding_neurons = random.sample(
                                self.composite_units['concept_binding'], 
                                min(5, len(self.composite_units['concept_binding']))
                            )
                        
                        # Combine all neurons
                        self.composite_mappings[composite_name] = neurons1 + neurons2 + binding_neurons
        
        return concepts

    def _init_learning_module(self):
        """Initialize enhanced learning module with multiple learning mechanisms"""
        # Initialize neural plasticity parameters
        self.learning_module = {
            # STDP (Spike-Timing-Dependent Plasticity)
            'stdp': {
                'enabled': True,
                'window_size': 20.0,  # ms
                'learning_rate': self.plasticity_rate,
                'potentiation_factor': 1.2,  # Strengthening factor
                'depression_factor': 0.9,  # Weakening factor
                'adaptive_window': True  # Window size adapts based on activity
            },
            
            # Hebbian learning
            'hebbian': {
                'enabled': True,
                'base_rate': self.plasticity_rate * 1.5,
                'decay_factor': 0.999,  # Slight decay over time
                'activation_threshold': 0.3  # Minimum activation for Hebbian update
            },
            
            # Homeostatic plasticity (regulates overall activity)
            'homeostatic': {
                'enabled': True,
                'target_activity': 0.1,  # Target average activity
                'adjustment_rate': 0.005,
                'check_interval': 100,  # Check every 100 time steps
                'last_check': 0
            },
            
            # Reinforcement learning
            'reinforcement': {
                'enabled': True,
                'base_reward_factor': 0.1,
                'penalty_factor': 0.05,
                'eligibility_trace_decay': 0.9,
                'eligibility_traces': torch.zeros(self.neuron_count, device=self.device)
            },
            
            # Metaplasticity (plasticity of plasticity)
            'metaplasticity': {
                'enabled': True,
                'threshold_adaptation_rate': 0.01,
                'learning_rate_adaptation': 0.005,
                'stability_factor': 0.9
            },
            
            # Consolidation (convert short-term to long-term changes)
            'consolidation': {
                'enabled': True,
                'short_term_changes': {},  # Map of (pre, post) -> change
                'consolidation_threshold': 5,  # Number of changes before consolidation
                'consolidation_rate': 0.2  # Portion of short-term change to consolidate
            }
        }
        
        # Activity statistics for learning
        self.activity_stats = {
            'neuron_activity': torch.zeros(self.neuron_count, device=self.device),  # Running average of activity
            'integration_history': deque(maxlen=100),  # History of phi values
            'coactivation_matrix': sparse.lil_matrix((self.neuron_count, self.neuron_count)),  # Using LIL format for efficient updates
            'activation_frequency': torch.zeros(self.neuron_count, device=self.device),  # How often neurons activate
            'recent_spikes': [deque(maxlen=100) for _ in range(self.neuron_count)]  # Recent spike times for STDP
        }

    def _init_domain_knowledge(self):
        """Initialize domain knowledge for improved hypothesis generation"""
        # Basic relational knowledge
        domain_knowledge = {
            'relation_properties': {
                'trusts': {
                    'transitivity': 0.6,  # trusts(A,B) and trusts(B,C) -> trusts(A,C) with 0.6 confidence
                    'symmetry': 0.4,  # trusts(A,B) -> trusts(B,A) with 0.4 confidence
                    'reflexivity': 0.9  # trusts(A,A) with 0.9 confidence
                },
                'likes': {
                    'transitivity': 0.3,
                    'symmetry': 0.6,
                    'reflexivity': 0.8
                },
                'fears': {
                    'transitivity': 0.1,
                    'symmetry': 0.2,
                    'reflexivity': 0.1
                },
                'knows': {
                    'transitivity': 0.7,
                    'symmetry': 0.6,
                    'reflexivity': 1.0
                }
            },
            
            # Relation implications
            'relation_implications': {
                'trusts': ['likes', 'knows'],  # trusts(A,B) -> likes(A,B), knows(A,B)
                'fears': ['knows', 'not_trusts'],
                'loves': ['likes', 'trusts'],
                'hates': ['not_likes', 'not_trusts']
            },
            
            # Entity types and constraints
            'entity_types': {
                'person': ['alice', 'bob', 'charlie', 'dave', 'eve', 'frank'],
                'emotion': ['happy', 'sad', 'angry', 'afraid', 'surprised'],
                'concept': ['trust', 'fear', 'knowledge', 'belief', 'doubt']
            },
            
            # Relation constraints (which entity types can participate in relations)
            'relation_constraints': {
                'trusts': {'subject': ['person'], 'object': ['person']},
                'likes': {'subject': ['person'], 'object': ['person', 'concept']},
                'knows': {'subject': ['person'], 'object': ['person', 'concept']},
                'feels': {'subject': ['person'], 'object': ['emotion']}
            },
            
            # Causal knowledge for reasoning
            'causal_patterns': [
                {'cause': 'likes', 'effect': 'helps', 'confidence': 0.7},
                {'cause': 'trusts', 'effect': 'confides_in', 'confidence': 0.8},
                {'cause': 'fears', 'effect': 'avoids', 'confidence': 0.9}
            ]
        }
        
        return domain_knowledge

    def register_specialized_synapse(self, pre, post):
        """
        Register a synapse for specialized learning controlled by a child class.
        
        Args:
            pre: Presynaptic neuron index
            post: Postsynaptic neuron index
            
        Returns:
            True if registered successfully
        """
        # Initialize specialized_learning_synapses if it doesn't exist
        if not hasattr(self, 'specialized_learning_synapses'):
            self.specialized_learning_synapses = set()
            
        self.specialized_learning_synapses.add((pre, post))
        
        # Also track the postsynaptic neuron
        if not hasattr(self, 'custom_learning_neurons'):
            self.custom_learning_neurons = set()
        self.custom_learning_neurons.add(post)
        
        return True
        
    def register_specialized_synapses(self, pre_post_pairs):
        """
        Register multiple synapses for specialized learning.
        
        Args:
            pre_post_pairs: List of (pre, post) tuples
            
        Returns:
            Number of synapses registered
        """
        count = 0
        for pre, post in pre_post_pairs:
            if self.register_specialized_synapse(pre, post):
                count += 1
        return count
        
    def unregister_specialized_synapse(self, pre, post):
        """
        Remove a synapse from specialized learning control.
        
        Args:
            pre: Presynaptic neuron index
            post: Postsynaptic neuron index
            
        Returns:
            True if unregistered successfully
        """
        if not hasattr(self, 'specialized_learning_synapses'):
            return False
            
        if (pre, post) in self.specialized_learning_synapses:
            self.specialized_learning_synapses.remove((pre, post))
            
            # Check if post neuron still has any specialized synapses
            has_other_synapses = False
            for p, p_post in self.specialized_learning_synapses:
                if p_post == post:
                    has_other_synapses = True
                    break
                    
            # Remove from custom_learning_neurons if no remaining specialized synapses
            if not has_other_synapses and hasattr(self, 'custom_learning_neurons'):
                if post in self.custom_learning_neurons:
                    self.custom_learning_neurons.remove(post)
                    
            return True
        return False

    def vectorize_input(self, text_input):
        """
        Enhanced vectorization using word embeddings and concept mappings.
        Converts text input to neural activation pattern.
        
        Args:
            text_input: Text to vectorize
            
        Returns:
            Neural activation pattern (numpy array)
        """
        # Initialize activation pattern
        activation = torch.zeros(self.neuron_count, device=self.device)
        
        # Preprocess text
        text_lower = text_input.lower()
        words = text_lower.split()
        
        # Remove punctuation from words
        words = [word.strip('.,;:!?()[]{}"\'-') for word in words]
        words = [word for word in words if word]  # Remove empty strings
        
        # Approach 1: Direct concept mapping
        # Check for concept matches and activate corresponding neurons
        for concept, neurons in self.concept_mappings.items():
            if concept in words or concept in text_lower:
                # Found concept - activate neurons with varying strength
                for neuron in neurons:
                    # Small random variation for more natural activation
                    activation[neuron] = 0.7 + 0.3 * torch.randn(1, device=self.device)
        
        # Also check composite concepts
        for composite, neurons in self.composite_mappings.items():
            concepts = composite.split('_')
            if all(concept in words or concept in text_lower for concept in concepts):
                # Found all concepts in the composite - activate binding neurons
                for neuron in neurons:
                    activation[neuron] = 0.8 + 0.2 * torch.randn(1, device=self.device)
        
        # Approach 2: Enhanced embedding-based activation if available
        if self.has_embeddings:
            # Get embeddings for words in the input
            embedded_vectors = []
            for word in words:
                if word in self.word_vectors:
                    embedded_vectors.append(self.word_vectors[word])
            
            # If we have embeddings, use them to activate neurons
            if embedded_vectors:
                # Combine embeddings (simple average)
                combined_vector = torch.mean(torch.stack(embedded_vectors), dim=0)
                
                # Map embedding dimensions to neurons
                neurons_per_dimension = max(1, self.neuron_count // self.embedding_dim)
                
                for i, value in enumerate(combined_vector):
                    # Map each embedding dimension to a group of neurons
                    start_idx = (i % self.embedding_dim) * neurons_per_dimension
                    end_idx = min(start_idx + neurons_per_dimension, self.neuron_count)
                    
                    # Scale activation based on embedding value
                    neuron_activation = (value + 1) / 2  # Normalize to 0-1 range (from -1 to 1)
                    
                    # Apply activation with decay based on position
                    for j in range(start_idx, end_idx):
                        decay = 1 - (j - start_idx) / neurons_per_dimension
                        activation[j] = max(activation[j], neuron_activation * decay * 0.5)
        
        # Approach 3: Special keywords with stronger activation
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
        
        for keyword, strength in keywords.items():
            if keyword in words or keyword in text_lower:
                # Activate corresponding neurons if found
                if keyword in self.concept_mappings:
                    for neuron_idx in self.concept_mappings[keyword]:
                        activation[neuron_idx] = max(activation[neuron_idx], strength)
        
        # Approach 4: Context-sensitive activation
        # Activate neurons based on sentence structure and relationships
        relationship_patterns = [
            (r"(\w+)\s+(trusts|likes|loves|knows|fears|hates|avoids)\s+(\w+)", 1, 2, 3),  # X verb Y
            (r"(\w+)\s+(?:doesn't|does\s+not|didn't|did\s+not)\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)", 1, 2, 3)  # X doesn't verb Y
        ]
        
        for pattern, subj_group, verb_group, obj_group in relationship_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if match:
                    subj = match.group(subj_group)
                    verb = match.group(verb_group)
                    obj = match.group(obj_group)
                    
                    # Activate subject neurons
                    if subj in self.concept_mappings:
                        for neuron in self.concept_mappings[subj]:
                            activation[neuron] = 0.85 + 0.15 * torch.randn(1, device=self.device)
                    
                    # Activate verb neurons
                    if verb in self.concept_mappings:
                        for neuron in self.concept_mappings[verb]:
                            activation[neuron] = 0.9 + 0.1 * torch.randn(1, device=self.device)
                    
                    # Activate object neurons
                    if obj in self.concept_mappings:
                        for neuron in self.concept_mappings[obj]:
                            activation[neuron] = 0.85 + 0.15 * torch.randn(1, device=self.device)
                    
                    # Activate relation binding units if available
                    if hasattr(self, 'composite_units') and 'relation_encoding' in self.composite_units:
                        for neuron in self.composite_units['relation_encoding']:
                            activation[neuron] = 0.7 + 0.3 * torch.randn(1, device=self.device)
                            
                    # Activate composite concept if exists
                    composite_key = f"{verb}_{subj}_{obj}"
                    if composite_key in self.composite_mappings:
                        for neuron in self.composite_mappings[composite_key]:
                            activation[neuron] = 0.95 + 0.05 * torch.randn(1, device=self.device)
        
        # Ensure activation is within valid range
        activation = torch.clamp(activation, 0.0, 1.0)
        
        return activation

    def simulate_spiking(self, input_activation, timesteps=10):
        """
        Simulate spiking network for given timesteps
        
        Args:
            input_activation: Input activation pattern
            timesteps: Number of simulation timesteps
            
        Returns:
            List of spikes [(neuron_id, voltage), ...] for each timestep
        """
        # Process various input formats
        if input_activation is not None:
            if isinstance(input_activation, str):
                # Text input
                activation = self.vectorize_input(input_activation)
                input_activation = activation
            elif isinstance(input_activation, list):
                # List of neuron IDs
                activation = torch.zeros(self.neuron_count, device=self.device)
                for idx in input_activation:
                    # Handle case where idx might be a numpy array
                    if isinstance(idx, torch.Tensor):
                        try:
                            idx = idx.item()  # Convert single-element array to its value
                        except (ValueError, TypeError):
                            continue  # Skip arrays that can't be converted to a scalar
                    
                    # Convert idx to integer if it's a string or numpy.str_
                    if isinstance(idx, (str, torch.str_, torch.bytes_)):
                        try:
                            idx = int(idx)
                        except ValueError:
                            continue  # Skip non-integer string indices
                    
                    # Now we can safely compare with an integer
                    if isinstance(idx, (int, torch.integer)) and 0 <= idx < self.neuron_count:
                        activation[idx] = 1.0
                input_activation = activation
            elif isinstance(input_activation, dict):
                # Convert dict of {neuron_id: activation} to array
                activation = torch.zeros(self.neuron_count, device=self.device)
                for idx, value in input_activation.items():
                    # Handle case where idx might be a numpy array
                    if isinstance(idx, torch.Tensor):
                        try:
                            idx = idx.item()  # Convert single-element array to its value
                        except (ValueError, TypeError):
                            continue  # Skip arrays that can't be converted to a scalar
                    
                    # Convert idx to integer if it's a string or numpy.str_
                    if isinstance(idx, (str, torch.str_, torch.bytes_)):
                        try:
                            idx = int(idx)
                        except ValueError:
                            continue  # Skip non-integer string indices
                    
                    # Now we can safely compare with an integer
                    if isinstance(idx, (int, torch.integer)) and 0 <= idx < self.neuron_count:
                        activation[idx] = value
                input_activation = activation
        
        # Apply input activation to membrane potentials
        self.membrane_potentials += input_activation
        
        # Reset active neurons cache
        self.active_neurons_cache = set()
        
        # Record spikes over time
        spikes_over_time = []
        
        # Reset region activations
        for _, region in self.regions.items():
            region['activation'] = 0.0
        
        # Run simulation for multiple timesteps
        for t in range(timesteps):
            # Determine which neurons spike by comparing to combined threshold
            # Use base_thresholds + plastic_thresholds instead of single spike_threshold
            combined_thresholds = self.base_thresholds + self.plastic_thresholds
            
            spiking_neurons = torch.where(
                (self.membrane_potentials > combined_thresholds) & 
                (self.refractory_counters <= 0)
            )[0]
            
            # Record current spikes
            current_spikes = [(int(n), float(self.membrane_potentials[n])) for n in spiking_neurons]
            spikes_over_time.append(current_spikes)
            
            # Update active neurons cache
            self.active_neurons_cache.update(spiking_neurons)
            
            # Reset membrane potential of spiking neurons and set refractory period
            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_counters[spiking_neurons] = self.refractory_period
            
            # Decrease refractory counters
            self.refractory_counters = torch.maximum(torch.zeros_like(self.refractory_counters), self.refractory_counters - 0.1)
            
            # Record spike times for STDP
            for n in spiking_neurons:
                self.last_spike_times[n] = t
                self.activity_stats['recent_spikes'][n].append(t)
                
                # Update activity statistics
                self.activity_stats['neuron_activity'][n] = 0.9 * self.activity_stats['neuron_activity'][n] + 0.1
                self.activity_stats['activation_frequency'][n] += 1
            
            # Propagate spikes to connected neurons
            if sparse.issparse(self.synaptic_weights):
                # Efficient sparse matrix method
                if len(spiking_neurons) > 0:
                    # Create sparse vector representation of spiking neurons
                    spiking_vector = torch.zeros(self.neuron_count, device=self.device)
                    spiking_vector[spiking_neurons] = 1.0
                    
                    # Multiply with weight matrix (sparse operation)
                    delta_potentials = self.synaptic_weights.T.dot(spiking_vector)
                    
                    # Update membrane potentials
                    self.membrane_potentials += delta_potentials
            else:
                # Standard method for dense matrix
                for src in spiking_neurons:
                    # Get all connections from this neuron
                    for dst in range(self.neuron_count):
                        weight = self.synaptic_weights[src, dst]
                        if weight != 0:
                            # Add weighted input to target neuron
                            self.membrane_potentials[dst] += weight
            
            # Apply decay to membrane potentials
            self.membrane_potentials *= self.decay_rate
            
            # Apply learning mechanisms
            if len(spiking_neurons) > 0:
                self._apply_learning_mechanisms(spiking_neurons, t)
        
        # Update region and assembly activations based on spiking activity
        self._update_activations(spikes_over_time)
        
        # Calculate integration metrics
        self._calculate_integration_metrics()
        
        # Record current phi in history
        self.activity_stats['integration_history'].append(self.phi)
        
        return spikes_over_time

    def _apply_learning_mechanisms(self, spiking_neurons, current_time):
        """Apply multiple learning mechanisms based on current neural activity"""
        if not self.learning_module:
            return
        
        # Apply STDP (Spike-Timing-Dependent Plasticity)
        if self.learning_module['stdp']['enabled']:
            self._apply_stdp(spiking_neurons, current_time)
        
        # Apply Hebbian learning
        if self.learning_module['hebbian']['enabled']:
            self._apply_hebbian(spiking_neurons)
        
        # Apply homeostatic plasticity
        if self.learning_module['homeostatic']['enabled']:
            # Only check periodically to save computation
            if current_time % self.learning_module['homeostatic']['check_interval'] == 0:
                self._apply_homeostatic()
                self.learning_module['homeostatic']['last_check'] = current_time
        
        # Apply metaplasticity
        if self.learning_module['metaplasticity']['enabled']:
            self._apply_metaplasticity(spiking_neurons)
        
        # Update eligibility traces for reinforcement learning
        if self.learning_module['reinforcement']['enabled']:
            self._update_eligibility_traces(spiking_neurons)

    def _apply_stdp(self, spiking_neurons, current_time):
        """Apply Spike-Timing-Dependent Plasticity with synapse control checks"""
        if not hasattr(self, 'learning_module') or not self.learning_module:
            return
            
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = hasattr(self, 'specialized_learning_synapses') and self.specialized_learning_synapses
        
        stdp_params = self.learning_module['stdp']
        window_size = stdp_params['window_size']
        learning_rate = stdp_params['learning_rate']
        potentiation_factor = stdp_params.get('potentiation_factor', 1.2)
        depression_factor = stdp_params.get('depression_factor', 0.9)
        
        # For each spiking neuron, update incoming connections
        for post in spiking_neurons:
            # Skip if the postsynaptic neuron is in the custom_learning_neurons set
            if hasattr(self, 'custom_learning_neurons') and post in self.custom_learning_neurons:
                continue
                
            # Iterate through all potential presynaptic neurons
            if sparse.issparse(self.synaptic_weights):
                # More efficient for sparse weights
                for pre in range(self.neuron_count):
                    # Only process if there's a connection
                    if self.synaptic_weights[pre, post] != 0:
                        # Skip if this specific synapse is specialized
                        if has_specialized_synapses and (pre, post) in self.specialized_learning_synapses:
                            continue
                            
                        self._apply_stdp_to_synapse(pre, post, current_time, window_size, 
                                                learning_rate, potentiation_factor, depression_factor)
            else:
                # For dense weights
                for pre in range(self.neuron_count):
                    # Skip if this specific synapse is specialized
                    if has_specialized_synapses and (pre, post) in self.specialized_learning_synapses:
                        continue
                        
                    self._apply_stdp_to_synapse(pre, post, current_time, window_size, 
                                            learning_rate, potentiation_factor, depression_factor)


    def _apply_stdp_to_synapse(self, pre, post, current_time, window_size, 
                            learning_rate, potentiation_factor, depression_factor):
        """Apply STDP to a specific synapse"""
        # Skip self-connections
        if pre == post:
            return
        # CRITICAL FIX: Skip if post neuron is in custom_learning_neurons set
        if hasattr(self, 'custom_learning_neurons') and post in self.custom_learning_neurons:
            return
            
        # Get the last spike times
        pre_spike_time = self.last_spike_times[pre]
        post_spike_time = current_time  # Current neuron just spiked
        
        # Only apply STDP if pre neuron has spiked recently
        if pre_spike_time > -900:  # Some reasonable threshold
            time_diff = post_spike_time - pre_spike_time
            
            # Check if within STDP window
            if -window_size <= time_diff <= window_size:
                # Calculate weight change
                if time_diff > 0:
                    # Pre neuron spiked before post neuron (potentiation - strengthen connection)
                    weight_change = learning_rate * potentiation_factor * torch.exp(-time_diff / window_size)
                else:
                    # Post neuron spiked before pre neuron (depression - weaken connection)
                    weight_change = -learning_rate * depression_factor * torch.exp(time_diff / window_size)
                
                # Record short-term changes for potential consolidation
                synapse_key = (pre, post)
                if self.learning_module['consolidation']['enabled']:
                    if synapse_key in self.learning_module['consolidation']['short_term_changes']:
                        self.learning_module['consolidation']['short_term_changes'][synapse_key] += weight_change
                    else:
                        self.learning_module['consolidation']['short_term_changes'][synapse_key] = weight_change
                
                # Ensure synaptic_weights is in LIL format for efficient updates
                if not isinstance(self.synaptic_weights, sparse.lil_matrix):
                    self.synaptic_weights = self.synaptic_weights.tolil()
                    
                # Update weight
                current_weight = self.synaptic_weights[pre, post]
                new_weight = max(0, min(1.0, current_weight + weight_change))
                self.synaptic_weights[pre, post] = new_weight
    
    # Enhanced version of _apply_hebbian method
    def _apply_hebbian(self, spiking_neurons):
        """Apply Hebbian learning to co-active neurons with synapse control checks"""
        if not hasattr(self, 'learning_module') or not self.learning_module:
            return
            
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = bool(self.specialized_learning_synapses)
        
        hebbian_params = self.learning_module['hebbian']
        base_rate = hebbian_params['base_rate']
        decay_factor = hebbian_params['decay_factor']
        threshold = hebbian_params['activation_threshold']
        
        # Apply only if multiple neurons are spiking
        if len(spiking_neurons) > 1:
            # Make sure coactivation_matrix is in lil format for efficient updates
            if not isinstance(self.activity_stats['coactivation_matrix'], sparse.lil_matrix):
                self.activity_stats['coactivation_matrix'] = self.activity_stats['coactivation_matrix'].tolil()
                
            # Update coactivation matrix for metaplasticity
            for i, pre in enumerate(spiking_neurons):
                for post in spiking_neurons[i+1:]:
                    # Skip if either neuron is managed by custom learning
                    if pre in self.custom_learning_neurons or post in self.custom_learning_neurons:
                        continue
                    
                    # Skip if this specific synapse pair is specialized
                    if has_specialized_synapses:
                        if (pre, post) in self.specialized_learning_synapses or (post, pre) in self.specialized_learning_synapses:
                            continue
                            
                    # Update coactivation count
                    self.activity_stats['coactivation_matrix'][pre, post] += 1
                    self.activity_stats['coactivation_matrix'][post, pre] += 1
                    
                    # Apply Hebbian learning
                    # "Neurons that fire together, wire together"
                    if self.activity_stats['coactivation_matrix'][pre, post] > threshold:
                        weight_change = base_rate * decay_factor * (1.0 - (self.synaptic_weights[pre, post] ** 2))
                    else:
                        weight_change = 0
                    
                    # Adjust weights in both directions (symmetric)
                    self.synaptic_weights[pre, post] = min(1.0, self.synaptic_weights[pre, post] + weight_change)
                    self.synaptic_weights[post, pre] = min(1.0, self.synaptic_weights[post, pre] + weight_change)
                    
                    # Record for consolidation
                    if self.learning_module['consolidation']['enabled']:
                        synapse_key = (pre, post)
                        self.learning_module['consolidation']['short_term_changes'][synapse_key] = \
                            self.learning_module['consolidation']['short_term_changes'].get(synapse_key, 0) + weight_change
                        
                        synapse_key = (post, pre)
                        self.learning_module['consolidation']['short_term_changes'][synapse_key] = \
                            self.learning_module['consolidation']['short_term_changes'].get(synapse_key, 0) + weight_change

    def _check_consolidation(self):
        """Check and apply weight consolidation with synapse control checks"""
        consolidation_params = self.learning_module['consolidation']
        threshold = consolidation_params['consolidation_threshold']
        rate = consolidation_params['consolidation_rate']
        
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = bool(self.specialized_learning_synapses)
        
        # Synapses to consolidate (copy to avoid modifying during iteration)
        to_consolidate = dict(consolidation_params['short_term_changes'])
        
        for (pre, post), change in to_consolidate.items():
            # Skip if either neuron is managed by custom learning
            if (pre in self.custom_learning_neurons or post in self.custom_learning_neurons):
                continue
                
            # Skip if this specific synapse is specialized
            if has_specialized_synapses and ((pre, post) in self.specialized_learning_synapses):
                continue
                
            # Only consolidate if change has accumulated beyond threshold
            if abs(change) >= threshold:
                # Consolidate a portion of the change
                consolidation_change = change * rate
                
                # Update weight
                current_weight = self.synaptic_weights[pre, post]
                new_weight = max(0, min(1.0, current_weight + consolidation_change))
                self.synaptic_weights[pre, post] = new_weight
                
                # Reduce short-term change
                consolidation_params['short_term_changes'][pre, post] -= consolidation_change
                
                # Remove if close to zero
                if abs(consolidation_params['short_term_changes'][pre, post]) < 0.01:
                    del consolidation_params['short_term_changes'][pre, post]
    
    def _apply_homeostatic(self):
        """Apply homeostatic plasticity to regulate neural activity with synapse control checks"""
        homeostatic_params = self.learning_module['homeostatic']
        target_activity = homeostatic_params['target_activity']
        adjustment_rate = homeostatic_params['adjustment_rate']
        
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = bool(self.specialized_learning_synapses)
        
        # Calculate activity deviation
        activity_deviation = self.activity_stats['neuron_activity'] - target_activity
        
        # Identify neurons needing adjustment
        high_activity_neurons = torch.where(activity_deviation > 0.1)[0]
        low_activity_neurons = torch.where(activity_deviation < -0.1)[0]
        
        # For hyperactive neurons, decrease incoming weights
        for neuron in high_activity_neurons:
            # Skip if this neuron is managed by custom learning
            if neuron in self.custom_learning_neurons:
                continue
                
            # Calculate downscaling factor based on deviation
            scale_factor = 1.0 - (adjustment_rate * activity_deviation[neuron])
            
            # Scale incoming weights
            if sparse.issparse(self.synaptic_weights):
                # For sparse matrix, we need to iterate over non-zero entries
                for i in range(self.neuron_count):
                    if self.synaptic_weights[i, neuron] != 0:
                        # Skip if this specific synapse is specialized
                        if has_specialized_synapses and (i, neuron) in self.specialized_learning_synapses:
                            continue
                        
                        self.synaptic_weights[i, neuron] *= scale_factor
            else:
                # For dense matrix, we need to handle each connection individually
                for i in range(self.neuron_count):
                    # Skip if this specific synapse is specialized
                    if has_specialized_synapses and (i, neuron) in self.specialized_learning_synapses:
                        continue
                    
                    self.synaptic_weights[i, neuron] *= scale_factor
        
        # For hypoactive neurons, increase incoming weights
        for neuron in low_activity_neurons:
            # Skip if this neuron is managed by custom learning
            if neuron in self.custom_learning_neurons:
                continue
                
            # Calculate upscaling factor based on deviation
            scale_factor = 1.0 - (adjustment_rate * activity_deviation[neuron])  # Negative deviation, so > 1
            
            # Scale incoming weights
            if sparse.issparse(self.synaptic_weights):
                for i in range(self.neuron_count):
                    if self.synaptic_weights[i, neuron] != 0:
                        # Skip if this specific synapse is specialized
                        if has_specialized_synapses and (i, neuron) in self.specialized_learning_synapses:
                            continue
                        
                        self.synaptic_weights[i, neuron] = min(1.0, self.synaptic_weights[i, neuron] * scale_factor)
            else:
                for i in range(self.neuron_count):
                    # Skip if this specific synapse is specialized
                    if has_specialized_synapses and (i, neuron) in self.specialized_learning_synapses:
                        continue
                    
                    if self.synaptic_weights[i, neuron] > 0:
                        self.synaptic_weights[i, neuron] = min(1.0, self.synaptic_weights[i, neuron] * scale_factor)

    def _apply_metaplasticity(self, spiking_neurons):
        """Apply metaplasticity - adaptation of plasticity parameters with synapse control checks"""
        meta_params = self.learning_module['metaplasticity']
        threshold_rate = meta_params['threshold_adaptation_rate']
        learning_rate_adaptation = meta_params['learning_rate_adaptation']
        
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = bool(self.specialized_learning_synapses)
        
        # Adjust base thresholds based on activity history
        recent_phi = list(self.activity_stats['integration_history'])
        if recent_phi:
            avg_phi = sum(recent_phi) / len(recent_phi)
            
            # If phi is decreasing, lower thresholds to encourage activity
            if len(recent_phi) > 5 and avg_phi < sum(recent_phi[-5:]) / 5:
                # Adjust base thresholds (not plastic thresholds)
                self.base_thresholds -= threshold_rate
                self.base_thresholds = torch.maximum(0.3, self.base_thresholds)
            # If phi is high, increase thresholds to prevent overactivity
            elif avg_phi > 0.7:
                self.base_thresholds += threshold_rate
                self.base_thresholds = torch.minimum(0.7, self.base_thresholds)
        
        # Make sure coactivation_matrix is in the right format before processing
        if hasattr(self, 'activity_stats') and 'coactivation_matrix' in self.activity_stats:
            # Ensure it's a LIL matrix for efficient access
            if not isinstance(self.activity_stats['coactivation_matrix'], sparse.lil_matrix):
                self.activity_stats['coactivation_matrix'] = self.activity_stats['coactivation_matrix'].tolil()
                
            # Find highly coactivated neurons pairs for metaplastic adjustment
            coactivation = self.activity_stats['coactivation_matrix']
            
            # Convert to array for easier threshold calculation
            # This is more efficient than accessing individual elements of a sparse matrix for comparison
            if isinstance(coactivation, sparse.spmatrix):
                coactivation_array = coactivation.toarray()
                threshold = np.mean(coactivation_array[coactivation_array > 0]) + np.std(coactivation_array[coactivation_array > 0])
                
                # Get the nonzero coordinates
                rows, cols = coactivation.nonzero()
                
                # Process each nonzero element
                for i, j in zip(rows, cols):
                    if coactivation[i, j] > threshold:
                        # Skip if either neuron is managed by custom learning
                        if i in self.custom_learning_neurons or j in self.custom_learning_neurons:
                            continue
                        
                        # Skip if this specific synapse is specialized
                        if has_specialized_synapses and ((i, j) in self.specialized_learning_synapses or 
                                                        (j, i) in self.specialized_learning_synapses):
                            continue
                        
                        # Apply metaplastic change to this synapse
                        self._modify_synapse_metaplasticity(i, j, learning_rate_adaptation)

    def _modify_synapse_metaplasticity(self, pre, post, adjustment_factor):
        """Helper to modify a synapse's metaplasticity"""
        # Current weight
        current_weight = self.synaptic_weights[pre, post]
        
        # Adjust based on current value - move toward 0.5 for balanced plasticity potential
        if current_weight < 0.5:
            # Increase weight slightly for weak connections to improve plasticity potential
            new_weight = min(0.5, current_weight * (1 + adjustment_factor))
        else:
            # Decrease weight slightly for strong connections to allow for more plasticity
            new_weight = max(0.5, current_weight * (1 - adjustment_factor))
        
        # Apply change
        self.synaptic_weights[pre, post] = new_weight
        self.synaptic_weights[post, pre] = new_weight  # Apply to both directions

    def _update_eligibility_traces(self, spiking_neurons):
        """Update eligibility traces for reinforcement learning"""
        reinforcement_params = self.learning_module['reinforcement']
        decay = reinforcement_params['eligibility_trace_decay']
        
        # Decay all eligibility traces
        reinforcement_params['eligibility_traces'] *= decay
        
        # Increase eligibility for active neurons
        reinforcement_params['eligibility_traces'][list(spiking_neurons)] += 1.0
        
        # Clip to reasonable range
        reinforcement_params['eligibility_traces'] = torch.clamp(
            reinforcement_params['eligibility_traces'], 0.0, 5.0)

    def apply_reinforcement(self, reward_signal, active_neurons=None):
        """
        Apply reinforcement learning based on reward signal with synapse control checks
        
        Args:
            reward_signal: Scalar reward signal (-1 to 1)
            active_neurons: Optional list of active neurons to reinforce
        """
        if not self.learning_module['reinforcement']['enabled']:
            return
            
        reinforcement_params = self.learning_module['reinforcement']
        traces = reinforcement_params['eligibility_traces']
        base_factor = reinforcement_params['base_reward_factor']
        
        # Check if we have specialized synapses to exclude
        has_specialized_synapses = bool(self.specialized_learning_synapses)
        
        # Use specified active neurons or default to neurons with non-zero eligibility
        if active_neurons is None:
            active_neurons = torch.where(traces > 0.01)[0]
        
        # Scale factor based on reward direction
        if reward_signal >= 0:
            factor = base_factor * reward_signal
        else:
            factor = reinforcement_params['penalty_factor'] * reward_signal
        
        # Apply weight changes based on eligibility traces
        for pre in active_neurons:
            # Skip if pre neuron is managed by custom learning
            if pre in self.custom_learning_neurons:
                continue
                
            for post in active_neurons:
                # Skip if post neuron is managed by custom learning
                if post in self.custom_learning_neurons:
                    continue
                    
                # Skip if this specific synapse is specialized
                if has_specialized_synapses and ((pre, post) in self.specialized_learning_synapses or 
                                            (post, pre) in self.specialized_learning_synapses):
                    continue
                
                # Apply change proportional to eligibility and reward
                pre_trace = traces[pre]
                post_trace = traces[post]
                weight_change = factor * pre_trace * post_trace
                
                # Skip if change is negligible
                if abs(weight_change) < 0.001:
                    continue
                    
                # Update weights in both directions (symmetric)
                current_weight_pre_post = self.synaptic_weights[pre, post]
                current_weight_post_pre = self.synaptic_weights[post, pre]
                
                if current_weight_pre_post != 0:  # Only update existing connections
                    new_weight = max(0, min(1.0, current_weight_pre_post + weight_change))
                    self.synaptic_weights[pre, post] = new_weight
                
                if current_weight_post_pre != 0:  # Only update existing connections
                    new_weight = max(0, min(1.0, current_weight_post_pre + weight_change))
                    self.synaptic_weights[post, pre] = new_weight

    def _update_activations(self, spikes_over_time):
        """Update region activations based on spiking activity"""
        # Count spikes per neuron
        spike_counts = defaultdict(int)
        for time_step in spikes_over_time:
            for neuron_idx, _ in time_step:
                spike_counts[neuron_idx] += 1
        
        # Update region activations
        for _, region in self.regions.items():
            region_neurons = region['neurons']
            if region_neurons:
                # Calculate activation based on spike counts and region size
                region_spike_count = sum(spike_counts.get(n, 0) for n in region_neurons)
                # Scale by region size and timesteps
                activation = min(1.0, region_spike_count / (len(region_neurons) * 0.5))
                region['activation'] = activation

    def _calculate_integration_metrics(self):
        """Calculate IIT-inspired metrics of consciousness"""
        # Get region activations
        activations = [region['activation'] for region in self.regions.values()]
        
        if not activations:
            return
        
        # Calculate differentiation (variance of activations)
        mean_activation = sum(activations) / len(activations)
        variance = sum((a - mean_activation) ** 2 for a in activations) / len(activations)
        self.differentiation = math.sqrt(variance)
        
        # Calculate integration (based on active connections between regions)
        total_connections = 0
        active_connections = 0
        
        for region_name, targets in self.region_connectivity.items():
            region_activation = self.regions[region_name]['activation']
            
            if region_activation > 0.3:  # Only consider significantly active regions
                for target_name in targets:
                    total_connections += 1
                    if self.regions[target_name]['activation'] > 0.3:
                        active_connections += 1
        
        self.integration = active_connections / max(1, total_connections)
        
        # Calculate phi (simplified IIT measure)
        # Phi is high when the system is both differentiated and integrated
        # Also consider overall activation level
        self.phi = self.differentiation * self.integration * mean_activation
        
        # Scale by complexity factor based on topology
        if self.topology_type == "scale_free":
            # Scale-free networks tend to have higher integration
            self.phi *= 1.2
        elif self.topology_type == "small_world":
            # Small world networks balance integration and differentiation
            self.phi *= 1.1
            
        # Ensure phi is in reasonable range
        self.phi = min(1.0, self.phi)

    def process_input(self, input_activation):
        """
        Process input through the SNN.
        
        Args:
            input_activation: Input activation pattern or spike data dictionary
                If dictionary, should contain 'spikes' key with 'times', 'units', and 'mask'
            
        Returns:
            Processing results
        """
        # Handle spike data input
        if isinstance(input_activation, dict) and 'spikes' in input_activation:
            # Convert spike data to activation pattern
            spike_data = input_activation['spikes']
            times = spike_data['times']
            units = spike_data['units']
            mask = spike_data['mask']
            
            # Create activation pattern from spikes
            activation = torch.zeros(self.neuron_count, device=self.device)
            for time, unit in zip(times[mask], units[mask]):
                # Handle case where unit might be a numpy array
                if isinstance(unit, torch.Tensor):
                    try:
                        unit = unit.item()  # Convert single-element array to its value
                    except (ValueError, TypeError):
                        continue  # Skip arrays that can't be converted to a scalar
                
                # Convert unit to integer if it's a string or numpy.str_
                if isinstance(unit, (str, torch.str_, torch.bytes_)):
                    try:
                        unit = int(unit)
                    except ValueError:
                        continue  # Skip non-integer string indices
                
                # Now we can safely compare with an integer
                if isinstance(unit, (int, torch.integer)) and unit < self.neuron_count:
                    # Add temporal encoding - earlier spikes get higher activation
                    temporal_factor = 1.0 - (time / 100.0)  # Normalize by time window
                    activation[unit] = max(activation[unit], temporal_factor)
            
            input_activation = activation
        
        # Run simulation
        spike_patterns = self.simulate_spiking(input_activation)
        
        # Calculate network metrics
        self._update_network_metrics(spike_patterns)
        
        # Store results
        result = {
            'spike_patterns': spike_patterns,
            'phi': self.phi,
            'integration': self.integration,
            'differentiation': self.differentiation
        }
        
        return result

    def _update_network_metrics(self, spike_patterns):
        """Update network metrics based on spike patterns"""
        # Calculate phi
        self._calculate_integration_metrics()
        
        # Update other metrics if needed
        # ... (this method should be implemented to update any additional metrics)

    def _apply_attention_mechanism(self, input_activation, context=None):
        """
        Apply an attention mechanism to focus processing on the most relevant parts of the input.
        
        Args:
            input_activation: Initial activation pattern
            context: Optional context from previous processing
            
        Returns:
            Attention-modulated activation pattern
        """
        # Initialize attention weights (all equal by default)
        attention_weights = torch.ones(self.neuron_count, device=self.device)
        
        # If we have context from previous processing, use it to bias attention
        if context is not None and 'active_concepts' in context:
            active_concepts = context['active_concepts']
            
            # Increase attention weights for neurons associated with active concepts
            for concept, activation in active_concepts.items():
                if concept in self.concept_mappings:
                    for neuron in self.concept_mappings[concept]:
                        attention_weights[neuron] *= (1.0 + activation)
        
        # Apply attention based on current network state and recency
        if hasattr(self, 'activity_stats') and 'neuron_activity' in self.activity_stats:
            # Blend recency (recent activity gets higher attention)
            recency_factor = 0.3
            attention_weights *= (1.0 + recency_factor * self.activity_stats['neuron_activity'])
        
        # Normalize attention weights
        attention_weights = attention_weights / torch.mean(attention_weights)
        
        # Apply attention to input activation
        modulated_activation = input_activation * attention_weights
        
        return modulated_activation
    
    def integrate_with_vector_symbolic(self, vector_engine):
        """
        Integrate with a vector symbolic engine for improved concept representation.
        
        Args:
            vector_engine: Instance of VectorSymbolicEngine
            
        Returns:
            Success status
        """
        # Import adapter if needed
        try:
            from models.symbolic.vector_symbolic import VectorSymbolicAdapter
            # Wrap the engine with our adapter to ensure all needed methods are available
            self.vector_symbolic = VectorSymbolicAdapter(vector_engine)
            print(f"Using VectorSymbolicAdapter to integrate with engine (dimension: {self.vector_symbolic.dimension})")
        except ImportError:
            # Fallback to using the engine directly
            self.vector_symbolic = vector_engine
            print("VectorSymbolicAdapter not available, using engine directly")
        
        # Create a cache for vector bindings
        self.vector_bindings = {}
        
        # Update concept mappings based on vector symbolic engine
        if hasattr(self.vector_symbolic, 'concept_vectors'):
            # Map concepts from vector symbolic engine to neural assemblies
            mapped_count = 0
            for concept, vector in self.vector_symbolic.concept_vectors.items():
                # Skip if concept already mapped
                if concept in self.concept_mappings:
                    continue
                    
                # Create a new neural assembly for this concept
                concept_size = max(20, min(self.neuron_count // 100, 50))
                
                # Select neurons from appropriate region based on concept type
                region_name = self._determine_concept_region(concept)
                
                if region_name in self.regions:
                    # Select neurons from this region
                    region_neurons = self.regions[region_name]['neurons']
                    if len(region_neurons) >= concept_size:
                        # Randomly select neurons from region
                        neurons = random.sample(region_neurons, concept_size)
                        self.concept_mappings[concept] = neurons
                        mapped_count += 1
                        
                        print(f"Mapped concept '{concept}' to {len(neurons)} neurons in {region_name} region")
            
            print(f"Mapped {mapped_count} concepts from vector symbolic engine")
            
            # Set up additional integration capabilities
            self._create_relation_vectors()
            
            return True
        return False

    def _create_relation_vectors(self):
        """Create composite relation vectors from concept vectors"""
        if not hasattr(self, 'vector_symbolic'):
            return
        
        # Set up a dictionary for subject-predicate-object relation vectors
        self.relation_vectors = {}
        
        # Find concepts that can be subjects, predicates, and objects
        subjects = []
        predicates = []
        objects = []
        
        # Simple categorization
        for concept in self.concept_mappings:
            if concept.lower() in ['likes', 'trusts', 'knows', 'fears', 'loves', 'hates', 'is_a', 'part_of']:
                predicates.append(concept)
            elif concept.lower() in ['person', 'alice', 'bob', 'charlie', 'dave', 'eve', 'animal', 'bird']:
                subjects.append(concept)
                objects.append(concept)
        
        # Create some relation vectors
        for subject in subjects[:5]:  # Limit to 5 subjects to avoid creating too many
            for predicate in predicates[:3]:  # Limit to 3 predicates
                for obj in objects[:5]:  # Limit to 5 objects
                    # Skip self-relations unless it's a reflexive predicate
                    if subject == obj and predicate not in ['knows', 'is_a', 'part_of']:
                        continue
                    
                    try:
                        # Create relation vector using VSA binding operations
                        relation_vec = self.vector_symbolic.encode_relation(subject, predicate, obj)
                        
                        # Store the relation vector
                        relation_key = f"{subject}_{predicate}_{obj}"
                        self.relation_vectors[relation_key] = relation_vec
                        
                        # Also create a neural assembly for this composite relation
                        if not hasattr(self, 'composite_mappings'):
                            self.composite_mappings = {}
                        
                        if relation_key not in self.composite_mappings:
                            # Create a dedicated assembly for this relation
                            # First, check if we have enough neurons in the conceptual region
                            conceptual_region = 'higher_cognition'
                            if conceptual_region in self.regions and len(self.regions[conceptual_region]['neurons']) >= 20:
                                # Use neurons from the conceptual region
                                composite_neurons = random.sample(self.regions[conceptual_region]['neurons'], 20)
                                self.composite_mappings[relation_key] = composite_neurons
                                print(f"Created composite mapping for relation: {relation_key}")
                    except Exception as e:
                        print(f"Error creating relation vector for {subject}_{predicate}_{obj}: {e}")
        
        print(f"Created {len(self.relation_vectors)} relation vectors")

    def _determine_concept_region(self, concept):
        """Determine appropriate region for a concept based on its semantic properties"""
        # Simple heuristic for concept categorization
        emotional_concepts = ['happy', 'sad', 'angry', 'like', 'love', 'hate', 'fear', 'trust']
        knowledge_concepts = ['know', 'understand', 'learn', 'think', 'believe', 'concept']
        entity_concepts = ['person', 'object', 'place', 'thing', 'animal', 'plant']
        action_concepts = ['run', 'jump', 'move', 'act', 'make', 'create']
        
        concept_lower = concept.lower()
        
        if any(emotional in concept_lower for emotional in emotional_concepts):
            return 'affective' if 'affective' in self.regions else 'metacognition'
        elif any(knowledge in concept_lower for knowledge in knowledge_concepts):
            return 'higher_cognition' if 'higher_cognition' in self.regions else 'memory'
        elif any(entity in concept_lower for entity in entity_concepts):
            return 'memory' if 'memory' in self.regions else 'sensory'
        elif any(action in concept_lower for action in action_concepts):
            return 'output' if 'output' in self.regions else 'action'
        else:
            # Default to higher cognition or other appropriate region
            for region in ['higher_cognition', 'memory', 'integration', 'processing']:
                if region in self.regions:
                    return region
            
            # Fallback to first available region
            return next(iter(self.regions.keys()))
        
    def generate_ensemble_hypotheses(self, text, ensemble_count=5):
        """
        Generate hypotheses using an ensemble of network configurations.
        
        Args:
            text: Input text to reason about
            ensemble_count: Number of ensemble members
            
        Returns:
            List of hypotheses with consensus scores
        """
        all_hypotheses = []
        
        # Save original state
        original_weights = self.synaptic_weights.copy() if not sparse.issparse(self.synaptic_weights) else self.synaptic_weights.copy()
        original_threshold = self.spike_threshold
        
        # Run multiple configurations with small variations
        for i in range(ensemble_count):
            # Vary parameters slightly
            self.spike_threshold = original_threshold * (0.9 + 0.2 * torch.randn(1, device=self.device))
            
            # For sparse matrices
            if sparse.issparse(self.synaptic_weights):
                # Add small random noise to non-zero elements
                data = self.synaptic_weights.data
                noise = torch.randn(len(data), device=self.device)
                self.synaptic_weights.data = data * (1 + noise)
            else:
                # Add small random noise
                noise = torch.randn(self.synaptic_weights.shape, device=self.device)
                self.synaptic_weights = original_weights * (1 + noise)
            
            # Generate hypotheses with this configuration
            hypotheses = self.abductive_reasoning(text)
            
            # Add to collection
            all_hypotheses.extend(hypotheses)
        
        # Restore original state
        self.synaptic_weights = original_weights
        self.spike_threshold = original_threshold
        
        # Aggregate hypotheses - count occurrences to determine consensus
        hypothesis_counts = defaultdict(int)
        hypothesis_confidence = defaultdict(float)
        
        for hypothesis in all_hypotheses:
            relation = hypothesis['relation']
            hypothesis_counts[relation] += 1
            hypothesis_confidence[relation] += hypothesis['confidence']
        
        # Calculate consensus score
        consensus_hypotheses = []
        for relation, count in hypothesis_counts.items():
            consensus_score = count / ensemble_count
            avg_confidence = hypothesis_confidence[relation] / count
            
            # Extract components
            components = next((h['components'] for h in all_hypotheses if h['relation'] == relation), {})
            
            consensus_hypotheses.append({
                'relation': relation,
                'confidence': avg_confidence,
                'consensus': consensus_score,
                'components': components
            })
        
        # Sort by consensus then confidence
        consensus_hypotheses.sort(key=lambda x: (x['consensus'], x['confidence']), reverse=True)
        
        return consensus_hypotheses
    
    def adaptive_vectorization(self, text_input, learning_feedback=None):
        """
        Adaptively vectorize input text based on historical performance.
        
        Args:
            text_input: Text to vectorize
            learning_feedback: Optional feedback from previous performance
            
        Returns:
            Neural activation pattern
        """
        # Get standard vectorization
        base_activation = self.vectorize_input(text_input)
        
        # If we have learning feedback, adjust vectorization
        if learning_feedback is not None and 'concept_errors' in learning_feedback:
            concept_errors = learning_feedback['concept_errors']
            
            # For concepts that were incorrectly activated, reduce activation
            for concept, error in concept_errors.items():
                if error > 0 and concept in self.concept_mappings:
                    neurons = self.concept_mappings[concept]
                    # Reduce activation proportional to error
                    reduction_factor = max(0.5, 1.0 - error)
                    for neuron in neurons:
                        base_activation[neuron] *= reduction_factor
            
            # For concepts that should have been activated but weren't, increase activation
            for concept, error in concept_errors.items():
                if error < 0 and concept in self.concept_mappings:
                    neurons = self.concept_mappings[concept]
                    # Increase activation proportional to abs(error)
                    boost_factor = min(1.5, 1.0 + abs(error))
                    for neuron in neurons:
                        base_activation[neuron] = min(1.0, base_activation[neuron] * boost_factor)
        
        # If we have performance metrics, adjust based on historical accuracy
        if hasattr(self, 'performance_metrics') and 'concept_activation_precision' in self.performance_metrics:
            precision_metrics = self.performance_metrics['concept_activation_precision']
            
            for concept, precision_history in precision_metrics.items():
                if precision_history and concept in self.concept_mappings:
                    # Get average precision for this concept
                    avg_precision = sum(precision_history) / len(precision_history)
                    
                    # If precision is low, adjust activation
                    if avg_precision < 0.7:
                        # Make activation more conservative (higher threshold)
                        neurons = self.concept_mappings[concept]
                        for neuron in neurons:
                            if base_activation[neuron] < 0.7:
                                base_activation[neuron] *= 0.8
                    
                    # If precision is high, make activation more liberal
                    elif avg_precision > 0.9:
                        neurons = self.concept_mappings[concept]
                        for neuron in neurons:
                            if base_activation[neuron] > 0:
                                base_activation[neuron] = min(1.0, base_activation[neuron] * 1.2)
        
        return base_activation
    
    def apply_neuron_specialization(self, epochs=10):
        """
        Apply neuron specialization to enhance network organization.
        This enables neurons to self-organize into functional groups.
        
        Args:
            epochs: Number of specialization iterations
            
        Returns:
            Specialization metrics
        """
        # Track metrics
        metrics = {
            'specialization_index': [],
            'functional_clusters': [],
            'information_capacity': []
        }
        
        for epoch in range(epochs):
            # 1. Identity highly active neurons
            active_neurons = torch.where(self.activity_stats['neuron_activity'] > 0.3)[0]
            
            if len(active_neurons) < 10:
                # Not enough active neurons for meaningful specialization
                continue
            
            # 2. Cluster neurons by coactivation patterns
            clusters = self._cluster_by_coactivation(active_neurons)
            
            # 3. Strengthen within-cluster connections
            for cluster in clusters:
                if len(cluster) > 1:
                    for i, pre in enumerate(cluster):
                        for post in cluster[i+1:]:
                            # Increase connection strength within cluster
                            current_weight = self.synaptic_weights[pre, post]
                            # Avoid weakening existing strong connections
                            if current_weight > 0:
                                new_weight = min(1.0, current_weight * 1.1)
                                self.synaptic_weights[pre, post] = new_weight
                                self.synaptic_weights[post, pre] = new_weight
            
            # 4. Calculate specialization metrics
            specialization_index = self._calculate_specialization(clusters)
            metrics['specialization_index'].append(specialization_index)
            metrics['functional_clusters'].append(len(clusters))
            
            # Optional: Map specialized clusters to concepts
            self._map_clusters_to_concepts(clusters)
        
        return metrics

    def _cluster_by_coactivation(self, neurons):
        """Cluster neurons based on coactivation patterns"""
        # Extract coactivation submatrix for the neurons
        if sparse.issparse(self.activity_stats['coactivation_matrix']):
            coactivation = self.activity_stats['coactivation_matrix'][neurons, :][:, neurons].toarray()
        else:
            coactivation = self.activity_stats['coactivation_matrix'][neurons, :][:, neurons]
        
        # Simple clustering - group neurons that coactivate frequently
        clusters = []
        remaining = set(range(len(neurons)))
        
        # Threshold for considering neurons strongly coactivated
        threshold = np.mean(coactivation) + np.std(coactivation)
        
        while remaining:
            # Start a new cluster with the first remaining neuron
            idx = min(remaining)
            cluster = {idx}
            remaining.remove(idx)
            
            # Find strongly connected neurons
            changed = True
            while changed:
                changed = False
                for i in list(remaining):
                    # Check connections to current cluster
                    connections = [coactivation[i, j] for j in cluster]
                    if any(conn > threshold for conn in connections):
                        cluster.add(i)
                        remaining.remove(i)
                        changed = True
            
            # Map cluster indices back to original neuron indices
            neuron_cluster = [neurons[i] for i in cluster]
            clusters.append(neuron_cluster)
        
        return clusters

    def _calculate_specialization(self, clusters):
        """Calculate a specialization index based on clusters"""
        if not clusters:
            return 0.0
        
        # Calculate average cluster size
        avg_size = sum(len(c) for c in clusters) / len(clusters)
        
        # Calculate variance in cluster sizes (normalized)
        sizes = [len(c) for c in clusters]
        size_variance = np.var(sizes) / (avg_size ** 2) if avg_size > 0 else 0
        
        # Calculate within-cluster connection density
        within_density = 0
        between_density = 0
        
        total_within_connections = 0
        potential_within_connections = 0
        
        for cluster in clusters:
            # Count within-cluster connections
            for i, pre in enumerate(cluster):
                for post in cluster[i+1:]:
                    potential_within_connections += 1
                    if self.synaptic_weights[pre, post] > 0.1:
                        total_within_connections += 1
        
        if potential_within_connections > 0:
            within_density = total_within_connections / potential_within_connections
        
        # Specialization index combines clustering, size distribution, and density
        specialization = (1.0 + within_density - size_variance) / 2.0
        
        return specialization

    def _map_clusters_to_concepts(self, clusters):
        """Map specialized neuron clusters to concepts"""
        # For each cluster, check overlap with existing concept mappings
        for cluster_idx, cluster in enumerate(clusters):
            cluster_set = set(cluster)
            
            # Check overlap with each concept
            concept_overlap = {}
            for concept, neurons in self.concept_mappings.items():
                neuron_set = set(neurons)
                overlap = len(cluster_set.intersection(neuron_set)) / len(neuron_set) if neuron_set else 0
                if overlap > 0.3:  # Significant overlap
                    concept_overlap[concept] = overlap
            
            # If no significant overlap, create a new specialized concept
            if not concept_overlap:
                new_concept = f"specialized_cluster_{cluster_idx}"
                self.concept_mappings[new_concept] = list(cluster)
            else:
                # For concepts with significant overlap, update their mappings
                for concept, overlap in sorted(concept_overlap.items(), key=lambda x: x[1], reverse=True):
                    # Update with neurons from this cluster
                    self.concept_mappings[concept] = list(set(self.concept_mappings[concept]).union(cluster))

    def _apply_relation_properties(self, results):
        """Apply relation properties from domain knowledge"""
        if not hasattr(self, 'domain_knowledge') or 'relation_properties' not in self.domain_knowledge:
            return
            
        # Get existing facts
        facts = results['facts'].copy()
        
        # Track new facts
        new_facts = []
        
        # Apply properties to each relation
        for fact in facts:
            pred = fact['predicate']
            subj = fact['subject']
            obj = fact['object']
            conf = fact['confidence']
            
            # Check if we have properties for this relation
            if pred in self.domain_knowledge['relation_properties']:
                properties = self.domain_knowledge['relation_properties'][pred]
                
                # Apply transitivity
                if 'transitivity' in properties and properties['transitivity'] > 0:
                    trans_confidence = properties['transitivity']
                    # Find matching facts where object of fact1 == subject of fact2
                    for fact2 in facts:
                        if fact2['predicate'] == pred and fact2['subject'] == obj:
                            # Transitive relation: subj -> obj -> fact2[obj]
                            target_obj = fact2['object']
                            new_confidence = conf * fact2['confidence'] * trans_confidence
                            
                            if new_confidence > 0.5:  # Only add reasonably confident facts
                                new_fact = {
                                    'subject': subj,
                                    'predicate': pred,
                                    'object': target_obj,
                                    'confidence': new_confidence,
                                    'derived': True,
                                    'rule': 'transitivity'
                                }
                                new_facts.append(new_fact)
                                results['rules_applied'] += 1
                
                # Apply symmetry
                if 'symmetry' in properties and properties['symmetry'] > 0:
                    sym_confidence = properties['symmetry']
                    new_confidence = conf * sym_confidence
                    
                    if new_confidence > 0.5:
                        new_fact = {
                            'subject': obj,  # Swap subject and object
                            'predicate': pred,
                            'object': subj,
                            'confidence': new_confidence,
                            'derived': True,
                            'rule': 'symmetry'
                        }
                        new_facts.append(new_fact)
                        results['rules_applied'] += 1
                
                # Apply reflexivity (relation with self)
                if 'reflexivity' in properties and properties['reflexivity'] > 0:
                    refl_confidence = properties['reflexivity']
                    
                    # Add relation to self
                    new_fact = {
                        'subject': subj,
                        'predicate': pred,
                        'object': subj,
                        'confidence': refl_confidence,
                        'derived': True,
                        'rule': 'reflexivity'
                    }
                    new_facts.append(new_fact)
                    results['rules_applied'] += 1
        
        # Add new facts to results
        for fact in new_facts:
            results['facts'].append(fact)
            
            # Categorize by confidence
            if fact['confidence'] >= 0.8:
                results['certain'].append(fact)
            else:
                results['uncertain'].append(fact)

    def _apply_relation_implications(self, results):
        """Apply relation implications from domain knowledge"""
        if not hasattr(self, 'domain_knowledge') or 'relation_implications' not in self.domain_knowledge:
            return
            
        # Get existing facts
        facts = results['facts'].copy()
        
        # Track new facts
        new_facts = []
        
        # Apply implications to each relation
        for fact in facts:
            pred = fact['predicate']
            subj = fact['subject']
            obj = fact['object']
            conf = fact['confidence']
            
            # Check if we have implications for this relation
            if pred in self.domain_knowledge['relation_implications']:
                implications = self.domain_knowledge['relation_implications'][pred]
                
                # Create implied facts
                for implied_pred in implications:
                    # Handle negated implications
                    if implied_pred.startswith('not_'):
                        base_pred = implied_pred[4:]
                        # Check if there's an existing contradictory fact
                        contradicts = False
                        for existing in facts:
                            if (existing['predicate'] == base_pred and 
                                existing['subject'] == subj and 
                                existing['object'] == obj):
                                contradicts = True
                                break
                        
                        # Only add if it doesn't contradict existing facts
                        if not contradicts:
                            new_fact = {
                                'subject': subj,
                                'predicate': implied_pred,
                                'object': obj,
                                'confidence': conf * 0.7,  # Lower confidence for negative implications
                                'derived': True,
                                'rule': 'implication'
                            }
                            new_facts.append(new_fact)
                            results['rules_applied'] += 1
                    else:
                        # Positive implication
                        new_fact = {
                            'subject': subj,
                            'predicate': implied_pred,
                            'object': obj,
                            'confidence': conf * 0.8,  # Lower confidence for implications
                            'derived': True,
                            'rule': 'implication'
                        }
                        new_facts.append(new_fact)
                        results['rules_applied'] += 1
        
        # Add new facts to results
        for fact in new_facts:
            results['facts'].append(fact)
            
            # Categorize by confidence
            if fact['confidence'] >= 0.8:
                results['certain'].append(fact)
            else:
                results['uncertain'].append(fact)

    def _load_word_vectors(self, file_path):
        """Load word vectors from a text file (GloVe/Word2Vec format)"""
        vectors = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                word = tokens[0]
                vec = np.array(tokens[1:], dtype=np.float32)
                vectors[word] = vec
        
        # Store as a wrapper class for compatibility
        class WordVectors:
            def __init__(self, vectors):
                self.vectors = vectors
                self.vector_size = len(next(iter(vectors.values()))) if vectors else 0
            
            def __contains__(self, word):
                return word in self.vectors
            
            def __getitem__(self, word):
                return self.vectors[word]
            
            def most_similar(self, word, topn=5):
                """Find most similar words using cosine similarity"""
                if word not in self.vectors:
                    return []
                
                word_vec = self.vectors[word]
                similarities = []
                
                for other_word, other_vec in self.vectors.items():
                    if other_word != word:
                        # Cosine similarity
                        sim = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                        similarities.append((other_word, sim))
                
                # Sort by similarity and return top-n
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:topn]
        
        return WordVectors(vectors)
    
    def process_text_input(self, text, timesteps=20, add_special_tokens=True):
        """
        Process text input directly to spikes using the bidirectional processor.
        
        Args:
            text: Input text to process
            timesteps: Number of time steps for spike encoding
            add_special_tokens: Whether to add special tokens like BOS/EOS
            
        Returns:
            Spike pattern tensor
        """
        logging.info(f"Processing text input: '{text[:50]}...' if len(text) > 50 else text")
        
        # Check if bidirectional processor is available
        if not hasattr(self, 'bidirectional_processor') or self.bidirectional_processor is None:
            raise ValueError("Bidirectional processor not available")
        
        # Convert text to spikes using bidirectional processor
        spike_pattern = self.bidirectional_processor.text_to_spikes(
            text, timesteps=timesteps, add_special_tokens=add_special_tokens
        )
        
        # Return just the spike pattern - let each model handle the processing
        # This avoids size mismatch issues when different models process the same 
        # spike pattern but have different neuron counts
        return spike_pattern
    
    def process_text_sequence(self, text, timesteps=20, add_special_tokens=True):
        """
        Process text input as a sequence, maintaining token-level representation.
        
        Args:
            text: Input text to process
            timesteps: Number of time steps for spike encoding
            add_special_tokens: Whether to add special tokens like BOS/EOS
            
        Returns:
            Sequence of spike patterns, sequence of neuron activations
        """
        logging.info(f"Processing text sequence: '{text[:50]}...' if len(text) > 50 else text")
        
        # Convert text to sequence of spike patterns
        spike_sequence = self.bidirectional_processor.text_sequence_to_spike_sequence(
            text, timesteps=timesteps, add_special_tokens=add_special_tokens
        )
        
        # Process each spike pattern in the sequence
        sequence_length = spike_sequence.shape[0]
        activations_sequence = []
        
        for i in range(sequence_length):
            # Get the spike pattern for the current token
            token_spike_pattern = spike_sequence[i]  # Shape: [timesteps, neurons]
            
            # Convert to a single activation pattern by summing across timesteps
            activation_pattern = token_spike_pattern.sum(dim=0)
            
            # Normalize if needed
            if activation_pattern.max() > 0:
                activation_pattern = activation_pattern / activation_pattern.max()
            
            # Process the activation pattern
            activations = self.process_input(activation_pattern)
            activations_sequence.append(activations)
        
        return spike_sequence, activations_sequence
    
    def generate_text_output(self, spike_pattern, max_length=100, remove_special_tokens=True):
        """
        Generate text output from spike patterns using bidirectional processor.
        
        Args:
            spike_pattern: Spike pattern to decode (tensor or list)
            max_length: Maximum length of generated text
            remove_special_tokens: Whether to remove special tokens from output
            
        Returns:
            Generated text
        """
        # Check if bidirectional processor is available
        if not hasattr(self, 'bidirectional_processor') or self.bidirectional_processor is None:
            raise ValueError("Bidirectional processor not available")
        
        try:
            # Handle different spike pattern formats
            if isinstance(spike_pattern, list):
                # Convert list format to tensor
                # Assuming list of (time, neuron, strength) tuples
                max_time = 0
                max_neuron = 0
                for t, n, _ in spike_pattern:
                    max_time = max(max_time, t)
                    max_neuron = max(max_neuron, n)
                
                # Create tensor representation
                timesteps = max_time + 1
                neuron_count = max_neuron + 1
                output_spikes = torch.zeros((timesteps, neuron_count), device=self.device)
                
                for t, n, s in spike_pattern:
                    if t < timesteps and n < neuron_count:
                        output_spikes[t, n] = s
            
            elif torch.is_tensor(spike_pattern):
                # Use tensor directly with some preprocessing
                if len(spike_pattern.shape) == 3:
                    # If shape is [batch, time, neurons], take first batch
                    output_spikes = spike_pattern[0]
                else:
                    # Assume [time, neurons] format
                    output_spikes = spike_pattern
            else:
                raise ValueError(f"Unsupported spike pattern format: {type(spike_pattern)}")
            
            # Check if we need to resize the spike pattern for the decoder
            # The decoder expects the neuron dimension to match the vector_dim
            original_shape = output_spikes.shape
            
            # Sum across time dimension if we have a time dimension
            if len(output_spikes.shape) > 1:
                spike_sum = output_spikes.sum(dim=0)
            else:
                spike_sum = output_spikes
                
            # Resize to match vector_dim if necessary
            vector_dim = self.vector_dim if hasattr(self, 'vector_dim') else 300
            if spike_sum.shape[0] != vector_dim:
                # Resize using interpolation
                if spike_sum.shape[0] > vector_dim:
                    # Reduce size using average pooling
                    spike_sum = torch.nn.functional.adaptive_avg_pool1d(
                        spike_sum.unsqueeze(0).unsqueeze(0), 
                        vector_dim
                    ).squeeze(0).squeeze(0)
                else:
                    # Increase size using interpolation
                    spike_sum = torch.nn.functional.interpolate(
                        spike_sum.unsqueeze(0).unsqueeze(0), 
                        size=vector_dim,
                        mode='linear'
                    ).squeeze(0).squeeze(0)
            
            # Generate text using bidirectional processor
            generated_text = self.bidirectional_processor.spikes_to_text(
                spike_sum, max_length=max_length, remove_special_tokens=remove_special_tokens
            )
            
            return generated_text
            
        except Exception as e:
            logging.error(f"Error generating text output: {e}")
            return f"Error generating text: {e}"
    
    def generate_text_from_sequence(self, output_spike_sequence, remove_special_tokens=True):
        """
        Generate text from a sequence of spike patterns.
        
        Args:
            output_spike_sequence: Sequence of spike pattern tensors
            remove_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            Generated text
        """
        # Convert sequence of spike patterns to text
        generated_text = self.bidirectional_processor.spike_sequence_to_text(
            output_spike_sequence, remove_special_tokens=remove_special_tokens
        )
        
        logging.info(f"Generated text from sequence: '{generated_text[:50]}...' if len(generated_text) > 50 else generated_text")
        return generated_text
    
    def train_with_text(self, text_inputs, expected_outputs=None, epochs=5, learning_rate=0.01):
        """
        Train the network with text inputs and optional expected outputs.
        
        Args:
            text_inputs: List of text inputs for training
            expected_outputs: List of expected text outputs (for supervised learning) or None
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training metrics
        """
        logging.info(f"Training with {len(text_inputs)} text inputs over {epochs} epochs")
        
        metrics = {
            'loss_history': [],
            'accuracy_history': []
        }
        
        # Train tokenizer and embeddings if needed
        if hasattr(self.bidirectional_processor, 'train_embeddings'):
            self.bidirectional_processor.train_embeddings(text_inputs, epochs=3)
        
        # Main training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            for i, text_input in enumerate(text_inputs):
                # Convert text to spikes
                input_spikes, neuron_activations = self.process_text_input(text_input)
                
                # If supervised learning with expected outputs
                if expected_outputs is not None and i < len(expected_outputs):
                    expected_text = expected_outputs[i]
                    
                    # Convert expected text to spikes
                    expected_spikes = self.bidirectional_processor.text_to_spikes(expected_text)
                    
                    # Calculate error/loss between actual and expected spikes
                    error = torch.mean((neuron_activations - expected_spikes) ** 2).item()
                    epoch_loss += error
                    
                    # Apply learning mechanisms with error signal
                    active_neurons = torch.nonzero(neuron_activations > 0.5).flatten().tolist()
                    self._apply_learning_mechanisms(active_neurons, time.time())
                    
                    # For simple accuracy metric
                    generated_text = self.generate_text_output(neuron_activations)
                    if generated_text.strip() == expected_text.strip():
                        correct_predictions += 1
                else:
                    # Unsupervised learning
                    active_neurons = torch.nonzero(neuron_activations > 0.5).flatten().tolist()
                    self._apply_learning_mechanisms(active_neurons, time.time())
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(text_inputs) if expected_outputs else 0.0
            accuracy = correct_predictions / len(text_inputs) if expected_outputs else 0.0
            
            metrics['loss_history'].append(avg_loss)
            metrics['accuracy_history'].append(accuracy)
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return metrics