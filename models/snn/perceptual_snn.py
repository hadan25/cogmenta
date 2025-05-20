# perceptual_snn.py
import numpy as np
import snntorch as snn
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from snntorch import surrogate
from collections import defaultdict
from models.snn.enhanced_snn import EnhancedSpikingCore

class PerceptualSNN(EnhancedSpikingCore):
    """
    SNN specialized for perceptual processing (inspired by occipital and temporal lobes).
    
    This implementation focuses on processing perceptual inputs across different modalities,
    with specialized feature detectors that can learn and adapt to recognize patterns.
    Supports text modality by default, with extensibility for visual and auditory processing.
    
    Enhanced with hybrid training approach combining bio-inspired dynamics with supervised learning.
    """
    
    def __init__(self, neuron_count=500, topology_type="flexible", modality="text", num_classes=10, vector_dim=300, bidirectional_processor=None):
        """
        Initialize the perceptual SNN with specialized parameters.
        
        Args:
            neuron_count: Total number of neurons in the network
            topology_type: Type of network topology
            modality: Input modality ("text", "visual", or "auditory")
            num_classes: Number of output classes (default is 10 for digit classification)
            vector_dim: Dimension of the vector space for token embeddings
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
        """
        # Set perception mode before calling parent's __init__
        self.perception_mode = modality
        
        # Initialize feature detectors dictionary
        self.feature_detectors = {}
        
        # Number of output classes
        self.num_classes = num_classes
        
        # Initialize weights for neural network
        self.weights = np.zeros((neuron_count, neuron_count))
        self.output_weights = np.zeros((self.num_classes, neuron_count))
        self.hidden = np.zeros(neuron_count)
        
        # Initialize spike history list (overriding parent's defaultdict)
        self.spike_history = []
        self.temporal_window = 100  # ms
        
        # Initialize modality-specific feature specifications
        self.text_features = {
            "word_boundary": {"threshold": 0.6, "learning_rate": 0.03},
            "entity": {"threshold": 0.7, "learning_rate": 0.04},
            "relation": {"threshold": 0.65, "learning_rate": 0.04},
            "sentiment": {"threshold": 0.6, "learning_rate": 0.05},
            "question": {"threshold": 0.7, "learning_rate": 0.04},
            "concept": {"threshold": 0.75, "learning_rate": 0.03}
        }
        
        self.visual_features = {
            "edge": {"threshold": 0.5, "learning_rate": 0.05},
            "shape": {"threshold": 0.6, "learning_rate": 0.04},
            "color": {"threshold": 0.5, "learning_rate": 0.03},
            "object": {"threshold": 0.7, "learning_rate": 0.03},
            "spatial": {"threshold": 0.65, "learning_rate": 0.04}
        }
        
        self.auditory_features = {
            "pitch": {"threshold": 0.5, "learning_rate": 0.05},
            "rhythm": {"threshold": 0.6, "learning_rate": 0.04},
            "timbre": {"threshold": 0.6, "learning_rate": 0.03},
            "phoneme": {"threshold": 0.7, "learning_rate": 0.04},
            "word": {"threshold": 0.75, "learning_rate": 0.03}
        }
        
        # Call parent's __init__
        super().__init__(
            neuron_count=neuron_count, 
            topology_type=topology_type,
            model_type="perceptual",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )
        
        # Override with perception-optimized parameters
        self.connection_density = 0.2    # Higher local connectivity
        self.region_density = 0.8        # Very high within-region connectivity
        
        # Learning parameters for feature detectors
        self.feature_learning_rate = 0.05
        self.detector_plasticity_factor = 1.5  # Higher plasticity for feature detectors
        
        # Track feature detector performance
        self.detector_performance = defaultdict(list)
        
        # Input encoding parameters
        self.input_encoding_mapping = {}  # Will store input token/feature to neuron mappings
        self.temporal_encoding_window = 5  # For temporal encoding of sequential inputs
        
        # Flag to control parent plasticity during training
        self.disable_parent_plasticity = True
        self.training_mode = False
        
        # Initialize modality-specific settings
        self._init_modality_settings()
        
        # Initialize snntorch components for supervised learning
        self._setup_snntorch_components()
        
        # Training statistics
        self.training_iterations = 0
        
        # Device management
        self.device = torch.device('cpu')
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Device to move model to ('cpu' or 'cuda')
        """
        self.device = device
        
        # Move feature network to device if it exists
        if hasattr(self, 'feature_network'):
            self.feature_network = self.feature_network.to(device)
        
        # Move pattern network to device if it exists
        if hasattr(self, 'pattern_network'):
            self.pattern_network = self.pattern_network.to(device)
        
        return self
    
    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x: Input data dictionary containing spike information
            
        Returns:
            Model output
        """
        # Extract spike information
        spike_times = x['spikes']['times'].to(self.device)
        spike_units = x['spikes']['units'].to(self.device)
        spike_mask = x['spikes']['mask'].to(self.device)
        
        # Convert spikes to activation pattern
        batch_size = spike_times.size(0)
        activations = []
        
        for i in range(batch_size):
            # Get valid spikes for this sample
            valid_mask = spike_mask[i]
            times = spike_times[i][valid_mask]
            units = spike_units[i][valid_mask]
            
            # Convert to activation pattern
            activation = self._convert_spikes_to_activation(times, units)
            activations.append(activation)
        
        # Stack activations
        activations = torch.stack(activations)
        
        # Process through feature network
        if hasattr(self, 'feature_network'):
            features = self.feature_network(activations)
            return features
        else:
            return activations
    
    def _convert_spikes_to_activation(self, times, units):
        """
        Convert spike times and units to activation pattern.
        
        Args:
            times: Spike times
            units: Spike units
            
        Returns:
            Activation pattern tensor
        """
        # Create activation pattern
        activation = torch.zeros(self.neuron_count, device=self.device)
        
        # Convert spikes to activation
        for i in range(len(times)):
            time = times[i]
            unit = units[i]
            
            # Handle PyTorch tensor
            if torch.is_tensor(unit):
                if unit.device.type != 'cpu' and self.device == 'cpu':
                    unit = unit.cpu()
                # Try to convert single-element tensor to scalar
                if unit.numel() == 1:
                    try:
                        unit = unit.item()
                    except:
                        continue
            
            # Handle case where unit might be a numpy array
            if isinstance(unit, np.ndarray):
                try:
                    unit = unit.item()  # Convert single-element array to its value
                except (ValueError, TypeError):
                    continue  # Skip arrays that can't be converted to a scalar
                
            # Convert unit to integer if it's a string or numpy.str_
            if isinstance(unit, (str, np.str_, np.bytes_)):
                try:
                    unit = int(unit)
                except ValueError:
                    continue  # Skip non-integer string indices
            
            # Now we can safely compare with an integer
            if isinstance(unit, (int, np.integer)) and unit < self.neuron_count:
                # Add temporal encoding - earlier spikes get higher activation
                if torch.is_tensor(time):
                    if time.device.type != 'cpu' and self.device == 'cpu':
                        time = time.cpu()
                    if time.numel() == 1:
                        try:
                            time = time.item()
                        except:
                            continue
                temporal_factor = 1.0 - (time / 100.0)  # Normalize by time window
                activation[unit] = max(activation[unit], temporal_factor)
        
        return activation
    
    def train(self, mode=True):
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set to training mode
        """
        self.training_mode = mode
        if hasattr(self, 'feature_network'):
            self.feature_network.train(mode)
        if hasattr(self, 'pattern_network'):
            self.pattern_network.train(mode)
        return self
    
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        return self.train(False)
        
    def _init_modality_settings(self):
        """Initialize modality-specific parameters and processors"""
        if self.perception_mode == "text":
            # Simple tokenizer for text processing
            self.tokenizer = TextTokenizer()
            
        elif self.perception_mode == "visual":
            # Visual-specific settings can be added here
            pass
            
        elif self.perception_mode == "auditory":
            # Auditory-specific settings can be added here
            pass
        
        # Initialize with perception-specific settings
        self.feature_detectors = {}      # Will store specialized feature detection neurons
        
        # Learning parameters for feature detectors
        self.feature_learning_rate = 0.05
        self.detector_plasticity_factor = 1.5  # Higher plasticity for feature detectors
        
        # Track feature detector performance
        self.detector_performance = defaultdict(list)
        
        # Input encoding parameters
        self.input_encoding_mapping = {}  # Will store input token/feature to neuron mappings
        self.temporal_encoding_window = 5  # For temporal encoding of sequential inputs
        
        # Flag to control parent plasticity during training
        self.disable_parent_plasticity = True
        self.training_mode = False
        
        # Initialize snntorch components for supervised learning
        self._setup_snntorch_components()
        
        # Training statistics
        self.training_iterations = 0
    
    def _init_topology(self):
        """Override to create perception-specific regions"""
        # Call parent method to initialize basic topology
        super()._init_topology()
        
        # Enhance regions relevant to perception with more neurons
        if 'sensory' in self.regions:
            # Expand sensory region if needed
            sensory_size = len(self.regions['sensory']['neurons'])
            if sensory_size < (self.neuron_count * 0.4):  # Ensure at least 40% of neurons for sensory
                # Reallocate neurons from other regions to expand sensory
                self._reallocate_neurons_to_sensory()
        
        # Create additional perception-specialized regions
        self._create_perceptual_regions()
                
        # Initialize feature detector assemblies
        self._init_feature_detectors()
    
    def _create_perceptual_regions(self):
        """Create specialized perceptual processing regions"""
        # Get available neurons that aren't already assigned
        assigned_neurons = set()
        for region in self.regions.values():
            assigned_neurons.update(region['neurons'])
        
        available_neurons = list(set(range(self.neuron_count)) - assigned_neurons)
        
        # If no neurons available, borrow from non-essential regions
        if not available_neurons:
            # Borrow from regions that are less critical for perception
            borrow_regions = ['output', 'metacognition']
            borrowed_neurons = []
            
            for region_name in borrow_regions:
                if region_name in self.regions:
                    # Take up to 20% of neurons from these regions
                    region_neurons = self.regions[region_name]['neurons']
                    borrow_count = len(region_neurons) // 5
                    borrowed = region_neurons[:borrow_count]
                    borrowed_neurons.extend(borrowed)
                    
                    # Update the source region
                    self.regions[region_name]['neurons'] = region_neurons[borrow_count:]
            
            available_neurons = borrowed_neurons
        
        # Create new perceptual regions if we have neurons available
        if available_neurons:
            neurons_per_region = max(20, len(available_neurons) // 3)
            
            # Feature extraction region
            feature_neurons = available_neurons[:neurons_per_region]
            self.regions['feature_extraction'] = {
                'neurons': feature_neurons,
                'activation': 0.0,
                'recurrent': True,
                'plasticity_factor': 1.2  # Higher plasticity for feature learning
            }
            
            # Pattern integration region
            if len(available_neurons) > neurons_per_region:
                pattern_neurons = available_neurons[neurons_per_region:2*neurons_per_region]
                self.regions['pattern_integration'] = {
                    'neurons': pattern_neurons,
                    'activation': 0.0,
                    'recurrent': True,
                    'plasticity_factor': 1.1
                }
            
            # Perceptual prediction region
            if len(available_neurons) > 2*neurons_per_region:
                prediction_neurons = available_neurons[2*neurons_per_region:]
                self.regions['perceptual_prediction'] = {
                    'neurons': prediction_neurons,
                    'activation': 0.0,
                    'recurrent': True,
                    'plasticity_factor': 1.0
                }
                
        # Update region connectivity
        if 'feature_extraction' in self.regions:
            # Connect to appropriate regions
            for region_name in ['sensory', 'pattern_integration']:
                if region_name in self.regions:
                    # Get existing connections or initialize
                    if region_name not in self.region_connectivity:
                        self.region_connectivity[region_name] = []
                    if 'feature_extraction' not in self.region_connectivity:
                        self.region_connectivity['feature_extraction'] = []
                    
                    # Add bidirectional connections
                    if 'feature_extraction' not in self.region_connectivity[region_name]:
                        self.region_connectivity[region_name].append('feature_extraction')
                    if region_name not in self.region_connectivity['feature_extraction']:
                        self.region_connectivity['feature_extraction'].append(region_name)
                
        # Connect pattern integration to other regions
        if 'pattern_integration' in self.regions:
            for region_name in ['feature_extraction', 'perceptual_prediction', 'memory', 'higher_cognition']:
                if region_name in self.regions:
                    # Get existing connections or initialize
                    if region_name not in self.region_connectivity:
                        self.region_connectivity[region_name] = []
                    if 'pattern_integration' not in self.region_connectivity:
                        self.region_connectivity['pattern_integration'] = []
                    
                    # Add bidirectional connections
                    if 'pattern_integration' not in self.region_connectivity[region_name]:
                        self.region_connectivity[region_name].append('pattern_integration')
                    if region_name not in self.region_connectivity['pattern_integration']:
                        self.region_connectivity['pattern_integration'].append(region_name)
                
    def _reallocate_neurons_to_sensory(self):
        """Reallocate neurons from other regions to expand sensory region"""
        # Determine how many neurons to reallocate
        target_sensory_size = int(self.neuron_count * 0.4)  # 40% of neurons
        current_sensory_size = len(self.regions['sensory']['neurons'])
        neurons_needed = target_sensory_size - current_sensory_size
        
        if neurons_needed <= 0:
            return  # No reallocation needed
        
        # Identify donor regions (regions that can spare neurons)
        donor_regions = []
        for region_name, region in self.regions.items():
            if region_name != 'sensory':
                # Calculate how many neurons can be spared (keep at least 50%)
                max_donate = len(region['neurons']) // 2
                if max_donate > 0:
                    donor_regions.append((region_name, max_donate))
        
        # Sort donors by how many neurons they can spare (descending)
        donor_regions.sort(key=lambda x: x[1], reverse=True)
        
        # Reallocate neurons
        neurons_reallocated = 0
        for region_name, max_donate in donor_regions:
            if neurons_reallocated >= neurons_needed:
                break
                
            # Calculate how many to take from this region
            to_take = min(max_donate, neurons_needed - neurons_reallocated)
            
            # Take neurons from the donor region
            donor_neurons = self.regions[region_name]['neurons']
            reallocated_neurons = donor_neurons[:to_take]
            self.regions[region_name]['neurons'] = donor_neurons[to_take:]
            
            # Add these neurons to the sensory region
            self.regions['sensory']['neurons'].extend(reallocated_neurons)
            neurons_reallocated += to_take
            
            print(f"[PerceptualSNN] Reallocated {to_take} neurons from {region_name} to sensory region")
        
        print(f"[PerceptualSNN] Sensory region now has {len(self.regions['sensory']['neurons'])} neurons")
                
    def _init_feature_detectors(self):
        """
        Initialize specialized neuron assemblies for feature detection.
        Also registers these synapses for custom learning with parent class.
        """
        # Identify which features to initialize based on modality
        if self.perception_mode == "text":
            feature_specs = self.text_features
        elif self.perception_mode == "visual":
            feature_specs = self.visual_features
        elif self.perception_mode == "auditory":
            feature_specs = self.auditory_features
        else:
            # Default to text if unknown modality
            feature_specs = {"entity": {"threshold": 0.7, "learning_rate": 0.04}}
        
        # Get neurons from perceptual regions for feature detectors
        available_regions = ['sensory', 'feature_extraction', 'pattern_integration']
        detector_neurons = []
        
        for region_name in available_regions:
            if region_name in self.regions:
                # Allocate a portion of each region's neurons
                region_neurons = self.regions[region_name]['neurons']
                allocation_size = len(region_neurons) // 4  # Use 25% from each region
                detector_neurons.extend(region_neurons[:allocation_size])
                
                # Update region's neurons
                self.regions[region_name]['neurons'] = region_neurons[allocation_size:]
        
        # Create feature detectors
        features_count = len(feature_specs)
        synapse_count = 0
        
        if features_count > 0 and detector_neurons:
            neurons_per_feature = max(10, len(detector_neurons) // features_count)
            
            for i, (feature_name, params) in enumerate(feature_specs.items()):
                # Calculate start and end indices for this feature's neurons
                start_idx = i * neurons_per_feature
                end_idx = min((i + 1) * neurons_per_feature, len(detector_neurons))
                
                if start_idx < len(detector_neurons):
                    # Allocate neurons for this feature detector
                    feature_neurons = detector_neurons[start_idx:end_idx]
                    
                    # Create the feature detector
                    self.feature_detectors[feature_name] = {
                        'neurons': feature_neurons,
                        'threshold': params['threshold'],
                        'learning_rate': params['learning_rate'],
                        'confidence': 0.5,  # Initial confidence
                        'weights': None  # Will be initialized during learning
                    }
                    
                    print(f"[PerceptualSNN] Created feature detector '{feature_name}' with {len(feature_neurons)} neurons")
                    
                    # Register all detector neurons for specialized learning
                    for post in feature_neurons:
                        self.custom_learning_neurons.add(post)
            
        # Initialize feature detector weights
        self._initialize_detector_weights()
    
    def _initialize_detector_weights(self):
        """
        Initialize weights for feature detectors and register synapses
        for specialized learning with parent class.
        """
        synapse_count = 0
        
        for feature_name, detector in self.feature_detectors.items():
            detector_neurons = detector['neurons']
            
            # Create weight matrices for selected input neurons to connect to the detector
            # These will be adjusted during learning
            # For now, just initialize with small random weights
            
            # Get sensory neurons as potential inputs
            input_neurons = []
            if 'sensory' in self.regions:
                input_neurons.extend(self.regions['sensory']['neurons'])
            
            # Create sparse weight matrix connecting inputs to detector
            if input_neurons and detector_neurons:
                # Each detector neuron gets connected to a subset of input neurons
                weight_dict = {}
                for detector_neuron in detector_neurons:
                    # Connect to ~30% of input neurons with small weights
                    connect_count = max(1, len(input_neurons) // 3)
                    input_connections = random.sample(input_neurons, connect_count)
                    
                    # Initialize weights
                    weight_dict[detector_neuron] = {
                        input_neuron: 0.1 + 0.2 * random.random()  # Small positive weights
                        for input_neuron in input_connections
                    }
                    
                    # Register all synapses with the parent class for specialized learning
                    for input_neuron in input_connections:
                        self.register_specialized_synapse(input_neuron, detector_neuron)
                        synapse_count += 1
                
                # Store the weight connections
                detector['weights'] = weight_dict
        
        print(f"[PerceptualSNN] Registered {synapse_count} synapses for specialized learning")
    
    def _setup_snntorch_components(self):
        """Set up snntorch neural network components for supervised learning"""
        # Beta is the decay rate for the neuron's spike trace
        beta = 0.95
        
        # Use surrogate gradient for backpropagation through spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Get input size based on feature dimensions extracted from bio-inspired SNN
        perceptual_regions_size = self._get_perceptual_regions_size()
        
        # Ensure output size is at least 1 to avoid zero-element tensor warnings
        output_size = max(1, len(self.feature_detectors))
        
        # Define feature classifier network (predicts presence of features)
        self.feature_network = nn.Sequential(
            nn.Linear(perceptual_regions_size, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(64, output_size)
        )
        
        # Define pattern classifier network (for pattern recognition tasks)
        pattern_output_size = max(1, self.num_classes)  # Ensure at least 1 output
        self.pattern_network = nn.Sequential(
            nn.Linear(perceptual_regions_size, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(64, pattern_output_size)
        )
        
        # Define loss functions
        self.feature_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification for features
        self.pattern_loss_fn = nn.CrossEntropyLoss()   # Multi-class for patterns
        
        # Define optimizers
        self.feature_optimizer = optim.Adam(self.feature_network.parameters(), lr=0.01)
        self.pattern_optimizer = optim.Adam(self.pattern_network.parameters(), lr=0.01)
        
        # Training state
        self.training_stats = {
            'feature_losses': [],
            'pattern_losses': [],
            'accuracy': []
        }
    
    def _get_perceptual_regions_size(self):
        """Calculate the number of features in perceptual regions for SNN input size"""
        # Count neurons in perceptual processing regions for feature vector size
        perceptual_regions = ['sensory', 'feature_extraction', 'pattern_integration', 
                             'perceptual_prediction']
                             
        # Start with basic activation features
        feature_count = len(self.feature_detectors) * 3  # Basic + temporal features per detector
        
        # Add region activations
        feature_count += len([r for r in perceptual_regions if r in self.regions])
        
        # Add spike pattern features
        feature_count += 10  # Temporal spike pattern features
        
        # Ensure minimum feature size
        feature_count = max(50, feature_count)
        
        return feature_count
    
    def simulate_spiking(self, input_activation, timesteps=10):
        """
        Enhanced spiking simulation with feature detector integration.
        Modified to handle parent plasticity control during training.
        
        Args:
            input_activation: Initial activation pattern
            timesteps: Number of simulation steps
            
        Returns:
            List of spike events per timestep
        """
        # Store original plasticity settings if in training mode
        original_plasticity = {}
        if self.training_mode and self.disable_parent_plasticity:
            # Temporarily disable parent's plasticity mechanisms for controlled learning
            if hasattr(self, 'learning_module'):
                for key in self.learning_module:
                    if isinstance(self.learning_module[key], dict) and 'enabled' in self.learning_module[key]:
                        original_plasticity[key] = self.learning_module[key]['enabled']
                        self.learning_module[key]['enabled'] = False
        
        # Integrate feature detector weights into main synaptic weights
        self._apply_detector_weights_to_simulation()
        
        # Run standard simulation with integrated weights
        spike_patterns = super().simulate_spiking(input_activation, timesteps)
        
        # Restore original plasticity settings
        if self.training_mode and self.disable_parent_plasticity and original_plasticity:
            for key, enabled in original_plasticity.items():
                if key in self.learning_module and 'enabled' in self.learning_module[key]:
                    self.learning_module[key]['enabled'] = enabled
        
        # Track detector-specific activity for more detailed feature analysis
        detector_activities = {}
        for feature_name, detector in self.feature_detectors.items():
            detector_neurons = set(detector['neurons'])
            
            # Count spikes in each timestep for this detector
            timestep_activities = []
            for t, timestep_spikes in enumerate(spike_patterns):
                # Count detector neuron spikes in this timestep
                active_detector_neurons = [n for n, _ in timestep_spikes if n in detector_neurons]
                activation_ratio = len(active_detector_neurons) / len(detector_neurons) if detector_neurons else 0
                timestep_activities.append(activation_ratio)
            
            detector_activities[feature_name] = timestep_activities
        
        # Store for later analysis
        self.detector_timestep_activities = detector_activities
        
        return spike_patterns
    
    def process_input(self, input_activation):
        """
        Process spike data input through the SNN.
        
        Args:
            input_activation: Input activation pattern or spike data dictionary
                If dictionary, should contain 'spikes' key with 'times', 'units', and 'mask'
            
        Returns:
            Processing results
        """
        # Ensure spike_history is a list
        if not hasattr(self, 'spike_history') or not isinstance(self.spike_history, list):
            self.spike_history = []
            
        # Handle spike data input
        if isinstance(input_activation, dict) and 'spikes' in input_activation:
            # Convert spike data to activation pattern
            spike_data = input_activation['spikes']
            times = spike_data['times']
            units = spike_data['units']
            mask = spike_data['mask']
            
            # Create activation pattern from spikes
            activation = np.zeros(self.neuron_count)
            
            # Move tensors to CPU if they are on GPU
            if torch.is_tensor(times) and times.device.type != 'cpu':
                times_cpu = times.cpu()
            else:
                times_cpu = times
            
            if torch.is_tensor(units) and units.device.type != 'cpu':
                units_cpu = units.cpu()
            else:
                units_cpu = units
            
            if torch.is_tensor(mask) and mask.device.type != 'cpu':
                mask_cpu = mask.cpu()
            else:
                mask_cpu = mask
            
            for time, unit in zip(times_cpu[mask_cpu], units_cpu[mask_cpu]):
                # Convert unit to integer if it's a string or numpy.str_
                if isinstance(unit, (str, np.str_, np.bytes_)):
                    try:
                        unit = int(unit)
                    except ValueError:
                        continue  # Skip non-integer string indices
                
                # Now we can safely compare with an integer
                if isinstance(unit, (int, np.integer)) and unit < self.neuron_count:
                    # Add temporal encoding - earlier spikes get higher activation
                    temporal_factor = 1.0 - (time / self.temporal_window)
                    activation[unit] = max(activation[unit], temporal_factor)
            
            input_activation = activation
            
            # Store spike history for temporal processing
            self.spike_history.append({
                'times': times,
                'units': units,
                'mask': mask
            })
            
            # Keep only recent history
            if len(self.spike_history) > 10:
                self.spike_history.pop(0)
        
        # Process through parent class
        result = super().process_input(input_activation)
        
        # Add spike-specific processing results
        result['spike_history'] = self.spike_history[-1] if self.spike_history else None
        
        return result
    
    def train_perceptual_features(self, spike_data, labels=None, epochs=100):
        """
        Train feature detectors using spike data.
        
        Args:
            spike_data: Dictionary containing spike information
                Should have 'spikes' key with 'times', 'units', and 'mask'
            labels: Optional labels for supervised learning
            epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        # Initialize training metrics
        stats = {
            'loss': [],
            'accuracy': [],
            'feature_activations': []
        }
        
        # Enable training mode
        self.training = True
        
        for epoch in range(epochs):
            # Process input
            result = self.process_input(spike_data)
            
            # Extract features
            features = self._extract_features(result['spike_patterns'])
            
            # Update feature detectors
            if labels is not None:
                # Ensure labels are on the same device as the model
                if torch.is_tensor(labels) and hasattr(self, 'device'):
                    labels = labels.to(self.device)
                    
                # Supervised learning
                loss = self._update_supervised_features(features, labels)
                stats['loss'].append(loss)
            else:
                # Unsupervised learning
                self._update_unsupervised_features(features)
            
            # Record statistics
            stats['feature_activations'].append(features)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                if labels is not None:
                    print(f"Loss: {loss:.4f}")
        
        # Disable training mode
        self.training = False
        
        return stats
    
    def _extract_features(self, spike_patterns):
        """
        Extract features from spike patterns.
        
        Args:
            spike_patterns: List of spike patterns
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.neuron_count)
        
        # Count spikes per neuron
        for pattern in spike_patterns:
            for neuron_data in pattern:
                neuron = neuron_data[0]  # neuron id
                
                # Handle PyTorch tensors on CUDA
                if torch.is_tensor(neuron) and neuron.device.type != 'cpu':
                    neuron = neuron.cpu()
                    
                # Handle case where neuron might be a numpy array
                if isinstance(neuron, np.ndarray):
                    try:
                        neuron = neuron.item()  # Convert single-element array to its value
                    except (ValueError, TypeError):
                        continue  # Skip arrays that can't be converted to a scalar
                
                # Convert neuron to integer if it's a string or numpy.str_
                if isinstance(neuron, (str, np.str_, np.bytes_)):
                    try:
                        neuron = int(neuron)
                    except ValueError:
                        continue  # Skip non-integer string indices
                
                # Now we can safely compare with an integer
                if isinstance(neuron, (int, np.integer)) and neuron < self.neuron_count:
                    features[neuron] += 1
        
        # Normalize
        if np.max(features) > 0:
            features /= np.max(features)
        
        return features
    
    def _update_supervised_features(self, features, labels):
        """
        Update feature detectors using supervised learning.
        
        Args:
            features: Feature vector
            labels: Target labels
            
        Returns:
            Loss value
        """
        # Convert labels to one-hot encoding if needed
        if len(labels.shape) == 1:
            # Move tensor to CPU before converting to numpy
            if torch.is_tensor(labels):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = labels
            
            # Determine max label value to ensure we have enough classes
            max_label = np.max(labels_np)
            if max_label >= self.num_classes:
                # Update num_classes if needed
                old_num_classes = self.num_classes
                self.num_classes = max_label + 1
                print(f"[PerceptualSNN] Increasing number of classes from {old_num_classes} to {self.num_classes} based on data")
                
                # Resize output weights
                if self.output_weights.shape[0] < self.num_classes:
                    new_weights = np.zeros((self.num_classes, self.output_weights.shape[1]))
                    new_weights[:old_num_classes] = self.output_weights
                    self.output_weights = new_weights
            
            # Convert to one-hot encoding
            labels = np.eye(self.num_classes)[labels_np]
        
        # Forward pass
        predictions = self._forward_pass(features)
        
        # Calculate loss
        loss = self._calculate_loss(predictions, labels)
        
        # Backward pass
        self._backward_pass(features, labels, predictions)
        
        return loss
    
    def _update_unsupervised_features(self, features):
        """
        Update feature detectors using unsupervised learning.
        
        Args:
            features: Feature vector
        """
        # Apply STDP-like learning
        for neuron in range(self.neuron_count):
            if features[neuron] > 0:
                # Strengthen connections to active neurons
                self.weights[neuron] += 0.01 * features
                
                # Normalize weights
                if np.sum(self.weights[neuron]) > 0:
                    self.weights[neuron] /= np.sum(self.weights[neuron])
    
    def _forward_pass(self, features):
        """
        Forward pass through the network.
        
        Args:
            features: Input features
            
        Returns:
            Network predictions
        """
        # Linear layer
        hidden = np.dot(features, self.weights.T)
        
        # ReLU activation
        hidden = np.maximum(0, hidden)
        
        # Store hidden layer for backward pass
        self.hidden = hidden
        
        # Check if output_weights has the correct shape
        if self.output_weights.shape[0] != self.num_classes:
            # Resize output weights if num_classes has changed
            old_shape = self.output_weights.shape
            new_weights = np.zeros((self.num_classes, old_shape[1]))
            # Copy existing weights for classes that still exist
            new_weights[:min(old_shape[0], self.num_classes)] = self.output_weights[:min(old_shape[0], self.num_classes)]
            self.output_weights = new_weights
        
        # Output layer
        output = np.dot(hidden, self.output_weights.T)
        
        # Softmax
        exp_output = np.exp(output - np.max(output))
        predictions = exp_output / np.sum(exp_output)
        
        return predictions
    
    def _backward_pass(self, features, labels, predictions):
        """
        Backward pass through the network.
        
        Args:
            features: Input features
            labels: Target labels
            predictions: Network predictions
        """
        # Ensure predictions and labels have compatible shapes
        if len(predictions) != labels.shape[1]:
            # This can happen if num_classes changed
            print(f"[PerceptualSNN] Warning: Predictions shape {predictions.shape} does not match labels shape {labels.shape}")
            
            # Adjust predictions or labels to match
            if len(predictions) < labels.shape[1]:
                # Pad predictions with zeros
                padded_predictions = np.zeros(labels.shape[1])
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions
        
        # Calculate gradients
        output_grad = predictions - labels
        # Ensure output_weights has compatible shape for dot product
        if output_grad.shape[0] != self.output_weights.shape[0]:
            # Adjust output_grad to match output_weights shape
            if len(output_grad) > self.output_weights.shape[0]:
                output_grad = output_grad[:self.output_weights.shape[0]]
            else:
                # Pad output_grad
                padded_grad = np.zeros(self.output_weights.shape[0])
                padded_grad[:len(output_grad)] = output_grad
                output_grad = padded_grad
        
        hidden_grad = np.dot(output_grad, self.output_weights)
        hidden_grad[hidden_grad < 0] = 0  # ReLU gradient
        
        # Update weights
        # Fix: Properly handle dimensions for outer product
        for i in range(len(output_grad)):
            self.output_weights[i] -= 0.01 * output_grad[i] * self.hidden
        
        self.weights -= 0.01 * np.outer(hidden_grad, features)
    
    def _calculate_loss(self, predictions, labels):
        """
        Calculate cross-entropy loss.
        
        Args:
            predictions: Network predictions
            labels: Target labels
            
        Returns:
            Loss value
        """
        # Ensure predictions and labels have the same shape
        if len(predictions) != labels.shape[1]:
            if len(predictions) < labels.shape[1]:
                # Pad predictions with epsilon to avoid log(0)
                epsilon = 1e-15
                padded_predictions = np.full(labels.shape[1], epsilon)
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions
            else:
                # Truncate predictions
                predictions = predictions[:labels.shape[1]]
        
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Only calculate loss for available classes in the batch
        valid_indices = np.where(np.sum(labels, axis=0) > 0)[0]
        if len(valid_indices) == 0:
            return 0.0  # No valid classes, return zero loss
        
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[:, valid_indices]
        
        # Calculate cross-entropy loss
        loss = -np.sum(valid_labels * np.log(valid_predictions)) / len(valid_indices)
        return loss
    
    def _apply_detector_weights_to_simulation(self, spike_patterns=None):
        """
        Integrates detector weights into the main synaptic weight matrix
        for use in simulation. This ensures feature detector learning
        is properly reflected in network dynamics.
        
        Args:
            spike_patterns: Optional recent spike patterns for activity-based integration
            
        Returns:
            Success status
        """
        if not self.feature_detectors:
            return False
        
        updates_applied = 0
        
        # Process each feature detector
        for feature_name, detector in self.feature_detectors.items():
            detector_neurons = detector['neurons']
            
            # Skip if no weights defined
            if 'weights' not in detector or not detector['weights']:
                continue
                
            # For each detector neuron, update its incoming connections in the main weight matrix
            for detector_neuron in detector_neurons:
                if detector_neuron in detector['weights']:
                    # Get this neuron's weight dictionary
                    input_weights = detector['weights'][detector_neuron]
                    
                    # Update weights in the main synaptic matrix
                    for input_neuron, weight in input_weights.items():
                        # Only update non-zero weights to maintain sparsity
                        if weight > 0.01:
                            # Update the main weight matrix
                            self.synaptic_weights[input_neuron, detector_neuron] = weight
                            updates_applied += 1
        
        if updates_applied > 0:
            print(f"[PerceptualSNN] Applied {updates_applied} detector weight updates to main weight matrix")
        
        return updates_applied > 0
    
    def analyze_input(self, input_data, modality=None):
        """
        Comprehensive analysis of input data combining bio-inspired feature extraction
        with supervised network predictions. Uses the hybrid approach for more accurate results.
        
        Args:
            input_data: Raw input data
            modality: Input modality
            
        Returns:
            Dictionary with feature detection results and analysis
        """
        # Extract features using bio-inspired network
        bio_features = self.extract_features(input_data, modality)
        
        # Get network state after processing
        network_state = {
            'region_activations': {name: region['activation'] for name, region in self.regions.items()},
            'phi': self.phi,
            'integration': self.integration,
            'differentiation': self.differentiation
        }
        
        # Get bio-inspired feature vector for supervised networks
        result = {
            'spike_patterns': self.spike_patterns if hasattr(self, 'spike_patterns') else []
        }
        bio_inspired_features = self._extract_bio_inspired_features(result)
        
        # Use supervised networks for additional predictions if available
        supervised_features = {}
        if hasattr(self, 'feature_network'):
            # Convert to tensor
            input_tensor = torch.tensor(bio_inspired_features, dtype=torch.float32).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                feature_output = self.feature_network(input_tensor)
                feature_probs = torch.sigmoid(feature_output)[0]
                
                # Convert to dictionary
                feature_names = list(self.feature_detectors.keys())
                for i, feature_name in enumerate(feature_names):
                    if i < len(feature_probs):
                        supervised_features[feature_name] = {
                            'probability': feature_probs[i].item(),
                            'detected': feature_probs[i].item() > 0.5
                        }
        
        # Blend bio-inspired and supervised predictions
        combined_features = {}
        for feature_name in self.feature_detectors:
            bio_info = bio_features.get(feature_name, {})
            sup_info = supervised_features.get(feature_name, {})
            
            # Get bio-inspired detection info
            bio_detected = bio_info.get('detected', False)
            bio_confidence = bio_info.get('confidence', 0.0)
            
            # Get supervised detection info
            sup_detected = sup_info.get('detected', False)
            sup_confidence = sup_info.get('probability', 0.5)
            
            # Combine predictions with weighted blend
            # More weight to bio-inspired for lower training iterations, 
            # gradually shifting to supervised with more training
            bio_weight = max(0.3, min(0.7, 1.0 - (self.training_iterations / 100)))
            sup_weight = 1.0 - bio_weight
            
            # Calculate final detection and confidence
            combined_confidence = (bio_confidence * bio_weight) + (sup_confidence * sup_weight)
            combined_detected = combined_confidence > 0.5
            
            # Store combined result
            combined_features[feature_name] = {
                'detected': combined_detected,
                'confidence': combined_confidence,
                'bio_detected': bio_detected,
                'bio_confidence': bio_confidence,
                'sup_detected': sup_detected,
                'sup_confidence': sup_confidence,
                'weights': {
                    'bio': bio_weight,
                    'supervised': sup_weight
                }
            }
        
        # Identify the dominant features (highest activation)
        dominant_features = []
        for feature_name, detection in combined_features.items():
            if detection['detected']:
                dominant_features.append((feature_name, detection['confidence']))
        
        # Sort by confidence
        dominant_features.sort(key=lambda x: x[1], reverse=True)
        
        # Analyze feature combinations for semantic significance
        semantic_analysis = self._analyze_feature_combinations(combined_features, network_state)
        
        # Compile analysis results
        analysis_results = {
            'features': combined_features,
            'dominant_features': dominant_features[:3],  # Top 3 features
            'network_state': network_state,
            'semantic_analysis': semantic_analysis
        }
        
        return analysis_results
    
    def _analyze_feature_combinations(self, features, network_state):
        """
        Analyze combinations of detected features for semantic significance.
        Implements a perceptual version of abductive reasoning.
        """
        semantic_analysis = {
            'recognized_patterns': [],
            'confidence': 0.0,
            'novel_elements': 0.0,
            'interpretation': None
        }
        
        # Check for specific feature combinations that have semantic meaning
        detected_features = [f for f, data in features.items() if data['detected']]
        
        # 1. Check for entity + relation pattern (potential subject-verb-object)
        if 'entity' in detected_features and 'relation' in detected_features:
            pattern = {
                'name': 'subject_predicate',
                'confidence': features['entity']['confidence'] * features['relation']['confidence'],
                'description': "Potential subject-predicate relationship detected"
            }
            semantic_analysis['recognized_patterns'].append(pattern)
        
        # 2. Check for entity + sentiment pattern
        if 'entity' in detected_features and 'sentiment' in detected_features:
            pattern = {
                'name': 'entity_sentiment',
                'confidence': features['entity']['confidence'] * features['sentiment']['confidence'],
                'description': "Emotional association with entity detected"
            }
            semantic_analysis['recognized_patterns'].append(pattern)
        
        # 3. Check for question pattern
        if 'question' in detected_features:
            question_conf = features['question']['confidence']
            pattern = {
                'name': 'question',
                'confidence': question_conf,
                'description': "Interrogative pattern detected"
            }
            semantic_analysis['recognized_patterns'].append(pattern)
            
        # 4. Check for concept recognition
        if 'concept' in detected_features:
            pattern = {
                'name': 'abstract_concept',
                'confidence': features['concept']['confidence'],
                'description': "Abstract concept pattern detected"
            }
            semantic_analysis['recognized_patterns'].append(pattern)
        
        # Calculate overall semantic confidence based on patterns and network integration
        if semantic_analysis['recognized_patterns']:
            # Average the confidence of recognized patterns
            pattern_confidence = sum(p['confidence'] for p in semantic_analysis['recognized_patterns'])
            pattern_confidence /= len(semantic_analysis['recognized_patterns'])
            
            # Factor in network integration (phi)
            integration_factor = network_state.get('phi', 0.5)
            
            # Combined confidence
            semantic_analysis['confidence'] = pattern_confidence * (0.7 + 0.3 * integration_factor)
            
            # Generate interpretation based on detected patterns
            semantic_analysis['interpretation'] = self._generate_interpretation(
                semantic_analysis['recognized_patterns'],
                network_state
            )
        
        # Measure novelty as the deviation from commonly observed pattern combinations
        # This is a simple implementation - a real system would track historical patterns
        semantic_analysis['novel_elements'] = random.uniform(0.3, 0.7)  # Placeholder
        
        return semantic_analysis
    
    def _generate_interpretation(self, patterns, network_state):
        """Generate semantic interpretation based on detected patterns"""
        # Filter patterns by minimum confidence
        relevant_patterns = [p for p in patterns if p['confidence'] >= 0.6]
        
        if not relevant_patterns:
            return "No strong interpretable patterns detected"
            
        # Sort by confidence
        relevant_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Generate interpretation based on pattern combinations
        if len(relevant_patterns) == 1:
            # Single pattern interpretation
            return f"Detected {relevant_patterns[0]['name']} pattern ({relevant_patterns[0]['description']})"
        else:
            # Multiple pattern interpretation
            pattern_names = [p['name'] for p in relevant_patterns]
            primary_pattern = relevant_patterns[0]['name']
            secondary_patterns = pattern_names[1:]
            
            if 'question' in pattern_names and 'entity' in pattern_names:
                return "Question pattern about a specific entity detected"
            elif 'entity_sentiment' in pattern_names:
                if network_state.get('phi', 0) > 0.6:
                    return "Integrated emotional perception associated with entity"
                else:
                    return "Simple emotional association with entity detected"
            elif 'subject_predicate' in pattern_names:
                if 'abstract_concept' in pattern_names:
                    return "Abstract relational structure detected"
                else:
                    return "Concrete relational structure detected"
            else:
                # Default interpretation
                return f"Complex pattern with {primary_pattern} as primary element, with {', '.join(secondary_patterns)} as secondary elements"
    
    def evaluate_perception(self, test_data):
        """
        Evaluate perception performance on test data.
        
        Args:
            test_data: Test data (list of samples)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize metrics
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": defaultdict(lambda: defaultdict(int))
        }
        
        # Count correct predictions
        correct = 0
        total = 0
        
        # Process each test sample
        for sample in test_data:
            # Extract input and label
            if isinstance(sample, tuple) and len(sample) == 2:
                input_data, label = sample
            else:
                input_data = sample
                label = None
            
            # Process input
            features = self.analyze_input(input_data)
            prediction = features.get("prediction")
            
            # Update metrics if label is available
            if label is not None and prediction is not None:
                total += 1
                if prediction == label:
                    correct += 1
                
                # Update confusion matrix
                metrics["confusion_matrix"][label][prediction] += 1
        
        # Calculate accuracy
        if total > 0:
            metrics["accuracy"] = correct / total
        
        return metrics
    
    def train_perception(self, spike_data, category=None, learn_rate=0.01):
        """
        Train the perceptual SNN on a single sample.
        
        Args:
            spike_data: Input spike data
            category: Category label (optional)
            learn_rate: Learning rate
            
        Returns:
            Dictionary with training results
        """
        # Initialize result
        result = {
            "error": 0.0,
            "success": False
        }
        
        try:
            # Process spike data
            if isinstance(spike_data, dict) and "times" in spike_data and "units" in spike_data:
                # Direct spike format
                times = spike_data["times"]
                units = spike_data["units"]
                
                # Convert to activation pattern
                activation = self._convert_spikes_to_activation(times, units)
                
                # Extract features
                features = self._extract_features([activation])
                
                # Update feature detectors
                if category is not None:
                    # Supervised learning
                    error = self._update_supervised_features(features, [category])
                    # Convert tensor to float if needed
                    if hasattr(error, 'item'):
                        error = error.item()
                else:
                    # Unsupervised learning
                    error = self._update_unsupervised_features(features)
                    # Convert tensor to float if needed
                    if hasattr(error, 'item'):
                        error = error.item()
                
                # Apply detector weights to simulation
                self._apply_detector_weights_to_simulation()
                
                # Update result
                result["error"] = error
                result["success"] = error < 0.3
                
            else:
                # Process input data
                features = self.analyze_input(spike_data)
                
                # Extract prediction and confidence
                prediction = features.get("prediction")
                confidence = features.get("confidence", 0.0)
                
                # Convert tensor to float if needed
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                
                # Calculate error
                if category is not None and prediction is not None:
                    error = 0.0 if prediction == category else 1.0 - confidence
                else:
                    error = 0.5  # Default error for unsupervised learning
                
                # Update feature detectors based on error
                for detector_name, detector_info in self.feature_detectors.items():
                    detector_neurons = detector_info["neurons"]
                    detector_weights = detector_info["weights"]
                    
                    # Skip if no weights defined
                    if detector_weights is None:
                        continue
                    
                    # Get activation for this detector
                    detector_activation = features.get(detector_name, 0.0)
                    
                    # Convert tensor to float if needed
                    if hasattr(detector_activation, 'item'):
                        detector_activation = detector_activation.item()
                    
                    # Update detector weights
                    self._update_detector_weights(
                        detector_name,
                        detector_activation,
                        error,
                        learn_rate
                    )
                
                # Update result
                result["error"] = error
                result["success"] = error < 0.3
                
        except Exception as e:
            print(f"Error training perception: {e}")
            result["error"] = 1.0
            result["success"] = False
        
        return result
    
    def transfer_to_supervised(self):
        """
        Transfer learning from unsupervised to supervised mode.
        
        This method adapts the network to supervised learning after
        initial unsupervised pre-training.
        """
        # Generate synthetic examples from current knowledge
        feature_names = list(self.feature_detectors.keys())
        
        # No transfer if no features or no supervised networks
        if not feature_names or not hasattr(self, 'feature_network'):
            return {
                "success": False,
                "error": "No features or supervised networks available for transfer"
            }
        
        # Track learning metrics
        transfer_metrics = {
            "samples_processed": 0,
            "feature_loss_before": 0,
            "feature_loss_after": 0
        }
        
        # Create a training dataset from bio-inspired representations
        training_samples = []
        
        # Generate samples with different feature combinations
        for _ in range(20):  # Create 20 synthetic examples
            # Randomly select active features
            active_features = {}
            for feature in feature_names:
                active_features[feature] = random.random() > 0.5
            
            # Create simple input
            if self.perception_mode == "text":
                # Create simple text that might trigger these features
                text_parts = []
                if active_features.get('entity', False):
                    text_parts.append("Person")
                if active_features.get('relation', False):
                    text_parts.append("interacts with")
                if active_features.get('sentiment', False):
                    text_parts.append("happily")
                if active_features.get('question', False):
                    text_parts = ["Does"] + text_parts + ["?"]
                
                input_data = " ".join(text_parts) if text_parts else "Sample text"
            else:
                # Generic input for other modalities
                input_data = {"synthetic": True, "active_features": active_features}
            
            # Process through bio-inspired SNN
            input_activation = self.encode_input(input_data)
            result = super().process_input(input_activation)
            
            # Extract features
            bio_features = self._extract_bio_inspired_features(result)
            
            # Create target
            feature_target = np.zeros(len(feature_names))
            for i, feature in enumerate(feature_names):
                if active_features.get(feature, False):
                    feature_target[i] = 1.0
            
            # Add to training samples
            training_samples.append({
                'features': bio_features,
                'feature_target': feature_target,
                'active_features': active_features
            })
        
        # Measure loss before training
        for sample in training_samples:
            # Convert to tensors
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            feature_target = torch.tensor(sample['feature_target'], dtype=torch.float32).unsqueeze(0)
            
            # Compute loss
            with torch.no_grad():
                feature_output = self.feature_network(features)
                feature_loss = self.feature_loss_fn(feature_output, feature_target)
                transfer_metrics['feature_loss_before'] += feature_loss.item()
        
        # Train on the samples
        num_epochs = 5  # Multiple passes for better learning
        
        for epoch in range(num_epochs):
            # Shuffle samples
            random.shuffle(training_samples)
            
            for sample in training_samples:
                # Convert to tensors
                features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
                feature_target = torch.tensor(sample['feature_target'], dtype=torch.float32).unsqueeze(0)
                
                # Train feature network
                self.feature_optimizer.zero_grad()
                feature_output = self.feature_network(features)
                feature_loss = self.feature_loss_fn(feature_output, feature_target)
                feature_loss.backward()
                self.feature_optimizer.step()
                
                transfer_metrics['samples_processed'] += 1
        
        # Measure loss after training
        for sample in training_samples:
            # Convert to tensors
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            feature_target = torch.tensor(sample['feature_target'], dtype=torch.float32).unsqueeze(0)
            
            # Compute loss
            with torch.no_grad():
                feature_output = self.feature_network(features)
                feature_loss = self.feature_loss_fn(feature_output, feature_target)
                transfer_metrics['feature_loss_after'] += feature_loss.item()
        
        # Calculate improvements
        num_samples = len(training_samples)
        if num_samples > 0:
            transfer_metrics['feature_loss_before'] /= num_samples
            transfer_metrics['feature_loss_after'] /= num_samples
            
            transfer_metrics['improvement'] = transfer_metrics['feature_loss_before'] - transfer_metrics['feature_loss_after']
        
        return {
            "success": True,
            "metrics": transfer_metrics,
            "samples_transferred": len(training_samples)
        }
    
    def transfer_to_bio_inspired(self):
        """
        Transfer knowledge from supervised networks to bio-inspired network.
        This allows the bio-inspired network to benefit from the structured learning
        in the supervised networks using direct error-driven learning.
        
        Returns:
            Results of the knowledge transfer
        """
        # Need supervised networks for this transfer
        if not hasattr(self, 'feature_network'):
            return {
                "success": False,
                "error": "No supervised networks available for transfer"
            }
        
        # Get feature names
        feature_names = list(self.feature_detectors.keys())
        
        # Track learning metrics
        transfer_metrics = {
            "weights_updated": 0,
            "samples_processed": 0,
            "features_processed": []
        }
        
        # Create synthetic examples for knowledge transfer
        for _ in range(10):  # Generate multiple samples
            # Randomly select a subset of features to activate
            active_features = {}
            for feature in feature_names:
                active_features[feature] = random.random() > 0.5
            
            # Create input that might trigger these features
            if self.perception_mode == "text":
                # Create simple text input
                text_parts = []
                if active_features.get('entity', False):
                    text_parts.append("Person")
                if active_features.get('relation', False):
                    text_parts.append("interacts with")
                if active_features.get('sentiment', False):
                    text_parts.append("happily")
                if active_features.get('question', False):
                    text_parts = ["Does"] + text_parts + ["?"]
                
                input_data = " ".join(text_parts) if text_parts else "Sample text"
            else:
                # Generic input for other modalities
                input_data = {"synthetic": True, "active_features": active_features}
            
            # Encode the input
            input_activation = self.encode_input(input_data)
            
            # Process through bio-inspired SNN
            result = super().process_input(input_activation)
            
            # Extract features for supervised networks
            bio_features = self._extract_bio_inspired_features(result)
            input_tensor = torch.tensor(bio_features, dtype=torch.float32).unsqueeze(0)
            
            # Get supervised predictions
            with torch.no_grad():
                feature_output = self.feature_network(input_tensor)
                feature_probs = torch.sigmoid(feature_output)[0]
            
            # For each feature, update detector weights based on supervised output
            for i, feature_name in enumerate(feature_names):
                if i >= len(feature_probs):
                    continue
                    
                # Get supervised prediction
                sup_prediction = feature_probs[i].item()
                
                # Get bio-inspired activation
                if feature_name in self.feature_detectors:
                    detector = self.feature_detectors[feature_name]
                    detector_neurons = detector['neurons']
                    
                    # Calculate activation from result
                    bio_activation = 0
                    if hasattr(self, 'spike_patterns'):
                        bio_activation = self._calculate_detector_activation(detector_neurons, self.spike_patterns)
                    
                    # Calculate target activation
                    # Convert probability to desired activation level
                    target_activation = max(0.3, sup_prediction)
                    
                    # Calculate error
                    error = target_activation - bio_activation
                    
                    # Only update if error is significant
                    if abs(error) > 0.1:
                        # Apply direct error-driven learning
                        learning_rate = detector.get('learning_rate', 0.05)
                        weights_updated = self._update_detector_weights(
                            feature_name,
                            input_activation,
                            error,
                            learning_rate
                        )
                        
                        transfer_metrics['weights_updated'] += weights_updated
                        
                        if feature_name not in transfer_metrics['features_processed']:
                            transfer_metrics['features_processed'].append(feature_name)
            
            transfer_metrics['samples_processed'] += 1
        
        return {
            "success": True,
            "metrics": transfer_metrics
        }
    
    def _update_detector_weights(self, feature_name, input_activation, error, learning_rate):
        """
        Update weights for a specific feature detector using error-driven learning.
        
        Args:
            feature_name: Name of the feature detector to update
            input_activation: Input activation pattern
            error: Error signal (difference between target and actual activation)
            learning_rate: Learning rate for weight updates
            
        Returns:
            Number of weights updated
        """
        if feature_name not in self.feature_detectors:
            return 0
            
        detector = self.feature_detectors[feature_name]
        detector_neurons = detector['neurons']
        
        # Skip if error is too small
        if abs(error) < 0.05:
            return 0
            
        # Get active input neurons
        active_inputs = np.where(input_activation > 0.2)[0]
        
        # Update weights for detector neurons
        weights_updated = 0
        
        for detector_neuron in detector_neurons:
            # Get or initialize weights dictionary for this neuron
            if 'weights' not in detector:
                detector['weights'] = {}
            
            if detector_neuron not in detector['weights']:
                detector['weights'][detector_neuron] = {}
            
            neuron_weights = detector['weights'][detector_neuron]
            
            # Update weights from active inputs
            for input_neuron in active_inputs:
                # Skip if out of range
                if input_neuron >= self.neuron_count:
                    continue
                    
                # Get current weight or initialize
                current_weight = neuron_weights.get(input_neuron, 0.0)
                
                # Calculate weight change using error-driven rule
                input_strength = input_activation[input_neuron]
                weight_change = learning_rate * error * input_strength
                
                # Apply change
                new_weight = current_weight + weight_change
                
                # Bound weights for stability
                new_weight = max(-1.0, min(1.0, new_weight))
                
                # Update if significantly changed
                if abs(new_weight - current_weight) > 0.001:
                    neuron_weights[input_neuron] = new_weight
                    weights_updated += 1
                    
                    # Update main weight matrix
                    self.synaptic_weights[input_neuron, detector_neuron] = new_weight
        
        return weights_updated
    
    def encode_input(self, input_data, modality=None):
        """
        Encode input data into neural activation pattern.
        
        Args:
            input_data: Raw input data (text, image, audio)
            modality: Input modality (defaults to self.perception_mode)
            
        Returns:
            Neural activation pattern (numpy array)
        """
        if modality is None:
            modality = self.perception_mode
            
        activation = np.zeros(self.neuron_count)
        
        if modality == "text":
            # Try semantic encoding first if possible
            activation = self._encode_text_embedding(input_data)
            
            # If no activation (failed or not implemented), fall back to basic encoding
            if np.sum(activation) < 0.1:
                activation = self._encode_text(input_data)
        elif modality == "visual":
            activation = self._encode_visual(input_data)
        elif modality == "auditory":
            activation = self._encode_auditory(input_data)
        else:
            print(f"[PerceptualSNN] Warning: Unsupported modality '{modality}'")
            
        return activation
    
    def _encode_text(self, text):
        """
        Encode text input for perceptual analysis using standardized encoding.
        
        Args:
            text: Input text to encode
            
        Returns:
            Neural activation pattern (numpy array)
        """
        activation = np.zeros(self.neuron_count)
        input_layer_neurons = self.regions.get('sensory', {}).get('neurons', [])
        
        if not input_layer_neurons:
            print("[PerceptualSNN] Warning: Sensory region not defined for input encoding.")
            return activation
        
        # Use standardized bidirectional processor if available
        if hasattr(self, 'bidirectional_processor') and self.bidirectional_processor is not None:
            # Process text through bidirectional processor
            try:
                # Use the standardized processor to get spike patterns
                spike_pattern = self.bidirectional_processor.text_to_spikes(text, timesteps=10)
                
                # Convert spike pattern to activation (flatten over time dimension)
                if torch.is_tensor(spike_pattern):
                    # If spike pattern is [timesteps, neurons]
                    if len(spike_pattern.shape) == 2:
                        # Sum over time dimension
                        spike_sum = spike_pattern.sum(dim=0)
                        # Normalize by number of timesteps
                        normalized = spike_sum / spike_pattern.shape[0]
                        # Map to neurons in the input layer
                        for i, neuron_idx in enumerate(input_layer_neurons):
                            if i < normalized.shape[0]:
                                activation[neuron_idx] = normalized[i].item()
                    # If spike pattern is [batch, timesteps, neurons]
                    elif len(spike_pattern.shape) == 3:
                        # Take first batch, sum over time
                        spike_sum = spike_pattern[0].sum(dim=0)
                        # Normalize and map to neurons
                        normalized = spike_sum / spike_pattern.shape[1]
                        for i, neuron_idx in enumerate(input_layer_neurons):
                            if i < normalized.shape[0]:
                                activation[neuron_idx] = normalized[i].item()
                
                return activation
            except Exception as e:
                print(f"[PerceptualSNN] Error using bidirectional processor: {e}")
                # Fall back to older method
        
        # FALLBACK: Use legacy encoding for compatibility
        # Tokenize the text using simple split
        words = text.lower().split()
        token_count = len(words)
        
        # Map tokens to neurons in the input layer
        neurons_per_token = max(1, len(input_layer_neurons) // (token_count * 2)) 
        
        # Encode each token
        for i, word in enumerate(words):
            # Calculate activation based on position (earlier = higher)
            position_factor = 1.0 - (0.05 * min(i, self.temporal_encoding_window))
            
            # Get the range of neurons for this token
            start_idx = min(i * neurons_per_token, len(input_layer_neurons) - neurons_per_token)
            end_idx = min(start_idx + neurons_per_token, len(input_layer_neurons))
            
            # Activate neurons for this token
            for j in range(start_idx, end_idx):
                neuron_idx = input_layer_neurons[j]
                activation[neuron_idx] = 0.7 * position_factor
        
        return activation
    
    def _initialize_text_encoding_mapping(self, initial_tokens):
        """
        Initialize mapping from text tokens to input neurons.
        
        Args:
            initial_tokens: Initial set of tokens to prepare for
        """
        # Skip initialization if we're using bidirectional processor
        if hasattr(self, 'bidirectional_processor') and self.bidirectional_processor is not None:
            print("[PerceptualSNN] Using standardized bidirectional processor for text encoding")
            return
            
        # Otherwise, initialize legacy mapping
        input_layer_neurons = self.regions.get('sensory', {}).get('neurons', [])
        
        if not input_layer_neurons:
            print("[PerceptualSNN] Warning: Sensory region not defined for input encoding.")
            return
            
        # Determine number of neurons per token
        token_count = len(initial_tokens) if initial_tokens else 100
        neurons_per_token = max(1, len(input_layer_neurons) // token_count)
        
        # Create mapping from tokens to neuron indices
        self.input_encoding_mapping = {}
        
        for i, token in enumerate(initial_tokens):
            # Get the range of neurons for this token
            start_idx = min(i * neurons_per_token, len(input_layer_neurons) - neurons_per_token)
            end_idx = min(start_idx + neurons_per_token, len(input_layer_neurons))
            
            # Store neuron indices for this token
            self.input_encoding_mapping[token] = [input_layer_neurons[j] for j in range(start_idx, end_idx)]
            
        print(f"[PerceptualSNN] Initialized encoding mapping for {len(self.input_encoding_mapping)} tokens")
    
    def _encode_text_embedding(self, text, embedding_model=None):
        """
        Encodes text using semantic word embeddings and maps to SNN activation.
        
        Args:
            text: Input text to encode
            embedding_model: Optional embedding model (will use internal vector engine if available)
            
        Returns:
            Neural activation pattern (numpy array)
        """
        activation = np.zeros(self.neuron_count)
        input_layer_neurons = self.regions.get('sensory', {}).get('neurons', [])
        
        if not input_layer_neurons:
            print("[PerceptualSNN] Warning: Sensory region not defined for input encoding.")
            return activation
        
        # First try to use standardized bidirectional processor with vector support
        if hasattr(self, 'bidirectional_processor') and self.bidirectional_processor is not None:
            try:
                # Get token ids from text
                token_ids = self.bidirectional_processor.tokenizer.encode(text)
                # Get vectors for tokens
                vectors = self.bidirectional_processor.get_vectors_for_tokens(token_ids)
                
                # Apply model-specific projection
                vectors = vectors.to(self.device)
                
                # Map vectors to neuron activations (distributing across input layer)
                vector_dim = vectors.shape[1]
                neurons_per_dim = max(1, len(input_layer_neurons) // vector_dim)
                
                # Process each dimension of the vector
                for dim_idx in range(vector_dim):
                    # Calculate average value across tokens for this dimension
                    avg_value = vectors[:, dim_idx].mean().item()
                    
                    # Map to activation range (0 to 1)
                    act_value = (avg_value + 1) / 2.0 if avg_value < 1 else 1.0
                    
                    # Calculate neuron indices for this dimension
                    start_idx = min(dim_idx * neurons_per_dim, len(input_layer_neurons) - neurons_per_dim)
                    end_idx = min(start_idx + neurons_per_dim, len(input_layer_neurons))
                    
                    # Activate neuron group based on embedding dimension value
                    for neuron_offset, neuron_idx in enumerate(range(start_idx, end_idx)):
                        # Activation decays with position in group
                        offset_factor = 1.0 - (0.2 * neuron_offset / neurons_per_dim)
                        neuron_act = act_value * offset_factor
                        
                        # Add noise for robustness
                        neuron_act = neuron_act * (0.9 + 0.2 * random.random())
                        
                        # Set activation (using max to handle token overlaps)
                        if neuron_idx < len(input_layer_neurons):
                            neuron = input_layer_neurons[neuron_idx]
                            activation[neuron] = max(activation[neuron], neuron_act)
                
                return activation
            except Exception as e:
                print(f"[PerceptualSNN] Error encoding with bidirectional processor: {e}")
                # Fall back to the regular embedding method
        
        # Tokenize the text (using standardized tokenizer if available, otherwise fallback)
        if hasattr(self, 'bidirectional_processor') and hasattr(self.bidirectional_processor, 'tokenizer'):
            # Use standardized tokenizer
            token_ids = self.bidirectional_processor.tokenizer.encode(text)
            tokens = [self.bidirectional_processor.tokenizer.id_to_token.get(id, "") for id in token_ids]
        else:
            # Fallback to custom tokenizer
            if not hasattr(self, 'tokenizer'):
                self.tokenizer = TextTokenizer()
            tokens = self.tokenizer.tokenize(text)
        
        # Get embedding model - try different sources
        if embedding_model is None:
            # First try to use the bidirectional processor for embeddings
            if hasattr(self, 'bidirectional_processor') and hasattr(self.bidirectional_processor, 'get_vectors_for_tokens'):
                # Use bidirectional processor vector functionality
                token_ids = self.bidirectional_processor.tokenizer.encode(text)
                vectors = self.bidirectional_processor.get_vectors_for_tokens(token_ids)
                
                # Process similar to above method
                vector_dim = vectors.shape[1]
                neurons_per_dim = max(1, len(input_layer_neurons) // vector_dim)
                
                for dim_idx in range(vector_dim):
                    avg_value = vectors[:, dim_idx].mean().item()
                    act_value = (avg_value + 1) / 2.0 if avg_value < 1 else 1.0
                    
                    start_idx = min(dim_idx * neurons_per_dim, len(input_layer_neurons) - neurons_per_dim)
                    end_idx = min(start_idx + neurons_per_dim, len(input_layer_neurons))
                    
                    for neuron_offset, neuron_idx in enumerate(range(start_idx, end_idx)):
                        offset_factor = 1.0 - (0.2 * neuron_offset / neurons_per_dim)
                        neuron_act = act_value * offset_factor * (0.9 + 0.2 * random.random())
                        
                        if neuron_idx < len(input_layer_neurons):
                            neuron = input_layer_neurons[neuron_idx]
                            activation[neuron] = max(activation[neuron], neuron_act)
                
                return activation
            
            # Try other embedding sources
            if hasattr(self, 'vector_symbolic'):
                embedding_model = self.vector_symbolic
            elif hasattr(self, 'word_vectors'):
                embedding_model = self.word_vectors
        
        # LEGACY APPROACH: If we have embeddings available, use semantic encoding
        if embedding_model is not None and hasattr(embedding_model, 'get_vector'):
            # Get embedding dimension 
            if hasattr(embedding_model, 'vector_size'):
                embedding_dim = embedding_model.vector_size
            else:
                # Default embedding dimension if not directly available
                embedding_dim = 100
                
            # Map embedding dimensions to neuron groups
            neurons_per_dim = max(1, len(input_layer_neurons) // embedding_dim)
            
            # Process each token
            for i, token in enumerate(tokens):
                try:
                    # Get embedding vector for token
                    if hasattr(embedding_model, 'get_vector'):
                        embedding = embedding_model.get_vector(token)
                    elif token in embedding_model:
                        embedding = embedding_model[token]
                    else:
                        continue  # Skip tokens without embeddings
                        
                    # Map each embedding dimension to corresponding neurons
                    for dim_idx in range(min(embedding_dim, len(embedding))):
                        # Get value from embedding (typically -1 to 1)
                        value = embedding[dim_idx]
                        
                        # Map to activation range (0 to 1)
                        act_value = (value + 1) / 2.0
                        
                        # Calculate neuron indices for this dimension
                        start_idx = min(dim_idx * neurons_per_dim, len(input_layer_neurons) - neurons_per_dim)
                        end_idx = min(start_idx + neurons_per_dim, len(input_layer_neurons))
                        
                        # Add temporal encoding - earlier tokens get slightly lower activation
                        temporal_factor = 1.0 - (0.05 * min(i, self.temporal_encoding_window))
                        
                        # Activate neuron group based on embedding dimension value
                        for neuron_idx in range(start_idx, end_idx):
                            neuron_offset = (neuron_idx - start_idx) / neurons_per_dim  # 0 to 1 based on position
                            
                            # Activation decays with position in group (creates more unique patterns)
                            neuron_act = act_value * (1.0 - (0.5 * neuron_offset)) * temporal_factor
                            
                            # Add noise for robustness
                            neuron_act = neuron_act * (0.9 + 0.2 * random.random())
                            
                            # Set activation (using max to handle token overlaps)
                            activation[input_layer_neurons[neuron_idx]] = max(activation[input_layer_neurons[neuron_idx]], neuron_act)
                except Exception as e:
                    print(f"[PerceptualSNN] Error encoding token '{token}': {e}")
                    continue
        else:
            # Fall back to simpler encoding if no embedding model available
            return self._encode_text(text)
        
        return activation
    
    def _encode_visual(self, image_data):
        """Encode visual input into neural activation pattern"""
        # Placeholder for visual encoding
        activation = np.zeros(self.neuron_count)
        
        # Get sensory neurons for encoding
        sensory_neurons = self.regions.get('sensory', {}).get('neurons', [])
        if not sensory_neurons:
            return activation
            
        # Simple encoding based on image features
        # In a real implementation, this would process actual image data
        
        # Simulate edge detection
        if isinstance(image_data, dict) and 'edges' in image_data:
            edge_intensity = image_data['edges']
            if 'edge' in self.feature_detectors:
                edge_neurons = self.feature_detectors['edge']['neurons']
                for neuron_idx in edge_neurons:
                    activation[neuron_idx] = edge_intensity * (0.7 + 0.3 * random.random())
        
        # Simulate color processing
        if isinstance(image_data, dict) and 'colors' in image_data:
            colors = image_data['colors']
            color_neurons_start = len(sensory_neurons) // 4
            color_neurons_end = color_neurons_start + len(sensory_neurons) // 4
            color_neurons = sensory_neurons[color_neurons_start:color_neurons_end]
            
            for i, (color, intensity) in enumerate(colors.items()):
                idx = i % len(color_neurons)
                activation[color_neurons[idx]] = intensity * (0.6 + 0.4 * random.random())
        
        return activation
    
    def _encode_auditory(self, audio_data):
        """Encode auditory input into neural activation pattern"""
        # Placeholder for auditory encoding
        activation = np.zeros(self.neuron_count)
        
        # Get sensory neurons for encoding
        sensory_neurons = self.regions.get('sensory', {}).get('neurons', [])
        if not sensory_neurons:
            return activation
            
        # Simple encoding based on audio features
        # In a real implementation, this would process actual audio data
        
        # Simulate frequency analysis
        if isinstance(audio_data, dict) and 'frequencies' in audio_data:
            freqs = audio_data['frequencies']
            freq_neurons_start = 0
            freq_neurons_end = len(sensory_neurons) // 3
            freq_neurons = sensory_neurons[freq_neurons_start:freq_neurons_end]
            
            for i, (freq, amplitude) in enumerate(freqs.items()):
                idx = i % len(freq_neurons)
                activation[freq_neurons[idx]] = amplitude * (0.7 + 0.3 * random.random())
        
        return activation
    
    def visualize_feature_detectors(self):
        """
        Create a visual representation of feature detector activations.
        Useful for analysis and debugging.
        
        Returns:
            Dictionary with visualization data
        """
        visualization = {
            'features': {},
            'regions': {},
            'topology': self.topology_type
        }
        
        # Gather feature detector information
        for feature_name, detector in self.feature_detectors.items():
            neurons = detector['neurons']
            threshold = detector.get('threshold', 0.5)
            
            # Calculate connection statistics
            total_connections = 0
            connection_strengths = []
            
            if 'weights' in detector:
                for detector_neuron, weights in detector['weights'].items():
                    total_connections += len(weights)
                    connection_strengths.extend(weights.values())
            
            # Calculate statistics
            avg_strength = np.mean(connection_strengths) if connection_strengths else 0
            max_strength = np.max(connection_strengths) if connection_strengths else 0
            min_strength = np.min(connection_strengths) if connection_strengths else 0
            
            visualization['features'][feature_name] = {
                'neuron_count': len(neurons),
                'threshold': threshold,
                'total_connections': total_connections,
                'avg_strength': float(avg_strength),
                'max_strength': float(max_strength),
                'min_strength': float(min_strength),
                'learning_rate': detector.get('learning_rate', 0.05),
                'confidence': detector.get('confidence', 0.5)
            }
        
        # Gather region information
        for region_name, region in self.regions.items():
            neurons = region['neurons']
            
            visualization['regions'][region_name] = {
                'neuron_count': len(neurons),
                'activation': region.get('activation', 0.0),
                'plasticity_factor': region.get('plasticity_factor', 1.0)
            }
        
        # Add global network metrics
        if hasattr(self, 'phi'):
            visualization['phi'] = self.phi
        if hasattr(self, 'integration'):
            visualization['integration'] = self.integration
        if hasattr(self, 'differentiation'):
            visualization['differentiation'] = self.differentiation
        
        return visualization
    
    def process_text_input(self, text_input, timesteps=20):
        """
        Process text input using the standardized bidirectional processor.
        
        Args:
            text_input: Text input for perception
            timesteps: Number of timesteps for spike patterns
            
        Returns:
            Processed spike patterns for perceptual processing
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
                
                # Return the spike patterns directly
                return spike_patterns
            except Exception as e:
                print(f"[PerceptualSNN] Error processing text with bidirectional processor: {e}")
                # Fall back to legacy encoding
        
        # Legacy encoding if bidirectional processor is not available or error occurs
        # Convert text to activation pattern
        activation = self._encode_text(text_input)
        # Simulate spiking to get spike patterns
        spike_patterns = self.simulate_spiking(activation, timesteps=timesteps)
        
        return spike_patterns
    
    def generate_text_output(self, spike_patterns, max_length=100):
        """
        Generate text output from perceptual spike patterns using the standardized bidirectional processor.
        
        Args:
            spike_patterns: Spike patterns from perceptual process
            max_length: Maximum length of generated text
            
        Returns:
            Generated text from perceptual process
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
                print(f"[PerceptualSNN] Error generating text with bidirectional processor: {e}")
                # Fall back to legacy approach
        
        # Legacy approach for text generation
        # For demonstration purposes, generate a simple description
        feature_activations = {}
        
        # Extract feature activations
        for feature_name, detector in self.feature_detectors.items():
            neurons = detector['neurons']
            activation = 0.0
            
            # Get average activation for this feature
            if isinstance(spike_patterns, list):
                # For list format, count spikes for feature neurons
                feature_spikes = 0
                for t, neuron, strength in spike_patterns:
                    if neuron in neurons:
                        feature_spikes += strength
                activation = feature_spikes / (len(neurons) + 0.001)
            elif torch.is_tensor(spike_patterns):
                # For tensor format, get average activation
                feature_indices = [i for i, n in enumerate(neurons) if i < spike_patterns.shape[1]]
                if feature_indices:
                    activation = spike_patterns[:, feature_indices].sum().item() / len(feature_indices)
            
            feature_activations[feature_name] = activation
        
        # Generate text based on feature activations
        output_text = "Perceived "
        active_features = [f for f, a in feature_activations.items() if a > 0.4]
        
        if active_features:
            output_text += ", ".join(active_features)
        else:
            output_text += "no strong features"
        
        return output_text


class TextTokenizer:
    """Simple text tokenizer for preprocessing text input"""
    
    def __init__(self):
        """Initialize the tokenizer"""
        self.special_tokens = {
            '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}',
            '-', '_', '+', '=', '*', '/', '\\', '|', '<', '>', ', '#', '@', '%', '^', '&'
        }
    
    def tokenize(self, text):
        """
        Tokenize text into words and special tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Handle special tokens
        for special in self.special_tokens:
            text = text.replace(special, f' {special} ')
            
        # Split on whitespace
        tokens = text.split()
        
        # Handle contractions and possessives
        processed_tokens = []
        for token in tokens:
            if "'" in token:
                # Handle contractions like don't, I'll, etc.
                if token.lower() in ["don't", "can't", "won't", "isn't", "aren't", "haven't"]:
                    processed_tokens.append(token)
                elif token.endswith("'s"):
                    # Handle possessives
                    processed_tokens.append(token[:-2])
                    processed_tokens.append("'s")
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return processed_tokens