# decision_snn.py
import numpy as np
import random
import time
from collections import defaultdict
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    torch = None

from models.snn.enhanced_snn import EnhancedSpikingCore
from models.snn.bidirectional_encoding import BidirectionalProcessor, create_processor

class DecisionSNN(EnhancedSpikingCore):
    """SNN specialized for decision-making and action selection with hybrid learning"""
    
    def __init__(self, neuron_count=400, topology_type="scale_free", vector_dim=300, bidirectional_processor=None):
        """
        Initialize the decision SNN with specialized parameters.
        
        Args:
            neuron_count: Total number of neurons in the network
            topology_type: Type of network topology ("scale_free" is recommended for decision)
            vector_dim: Dimension of the vector space for token embeddings
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
        """
        # Initialize custom learning neurons set before parent init
        self.custom_learning_neurons = set()
        
        # Flag to control parent class plasticity
        self.disable_parent_plasticity = False

        # Training mode flag
        self.training_mode = False
        
        super().__init__(
            neuron_count=neuron_count,
            topology_type=topology_type,
            model_type="decision",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )

        # Initialize neuron biases
        self.neuron_biases = np.zeros(neuron_count)
        
        # Override with decision-optimized parameters
        self.connection_density = 0.25    # Higher density for complex decision integration
        self.region_density = 0.6         # Medium-high internal connectivity
        
        # Decision-specific attributes
        self.decision_threshold = 0.7     # Threshold for making a decision
        self.action_channels = {}         # Maps action names to neuron groups
        self.decision_context = {}        # Contextual factors affecting decisions
        self.recent_decisions = []        # History of recent decisions
        
        # Set up SNNTorch components for supervised learning
        self._setup_snntorch_components()
        
        # Register specialized decision synapses for hybrid learning
        self._register_decision_synapses()
    
    def _setup_snntorch_components(self):
        """Set up SNNTorch components for supervised learning"""
        try:
            self.torch_available = True if torch else False
            
            if not self.torch_available:
                print("[DecisionSNN] Warning: PyTorch not available - supervised learning disabled")
                return
            
            # Beta is the decay rate for the neuron's spiking trace
            beta = 0.95
            
            # Use surrogate gradient for backpropagation through spikes
            spike_grad = surrogate.fast_sigmoid(slope=25)
            
            # Get size of decision features to use as input size for supervised networks
            decision_features_size = 100  # Fixed size for stability
            
            # Number of action channels (classes for classification)
            num_actions = 7  # Based on action_types in _init_action_channels
            
            # Create a wrapper class to handle tuple inputs properly
            class TupleInputWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    # Print debug info about input type
                    print(f"[DecisionSNN] Debug - Input type: {type(x)}")
                    
                    # Handle different input types
                    if isinstance(x, tuple):
                        print(f"[DecisionSNN] Debug - Tuple input with {len(x)} elements")
                        if len(x) > 0:
                            # If first element is a tensor, use it
                            if torch.is_tensor(x[0]):
                                x = x[0]
                            # If it's a numpy array, convert to tensor
                            elif isinstance(x[0], np.ndarray):
                                try:
                                    x = torch.tensor(x[0], dtype=torch.float32)
                                except:
                                    print("[DecisionSNN] Warning: Could not convert numpy array to tensor")
                                    x = torch.zeros(100, dtype=torch.float32)
                            # If it's a list or other sequence, convert to tensor
                            elif isinstance(x[0], (list, tuple)):
                                try:
                                    x = torch.tensor(x[0], dtype=torch.float32)
                                except:
                                    print("[DecisionSNN] Warning: Could not convert sequence to tensor")
                                    x = torch.zeros(100, dtype=torch.float32)
                            # If it's a dict, extract features if possible
                            elif isinstance(x[0], dict):
                                try:
                                    # Try to extract features from the dict
                                    if 'features' in x[0]:
                                        x = torch.tensor(x[0]['features'], dtype=torch.float32)
                                    elif 'activations' in x[0]:
                                        x = torch.tensor(x[0]['activations'], dtype=torch.float32)
                                    else:
                                        # Use zeros as fallback
                                        x = torch.zeros(100, dtype=torch.float32)
                                except:
                                    print("[DecisionSNN] Warning: Could not extract features from dict")
                                    x = torch.zeros(100, dtype=torch.float32)
                            else:
                                # Try direct conversion or use zeros as fallback
                                try:
                                    x = torch.tensor(x[0], dtype=torch.float32)
                                except:
                                    print(f"[DecisionSNN] Warning: Unknown element type in tuple: {type(x[0])}")
                                    x = torch.zeros(100, dtype=torch.float32)
                        else:
                            # Empty tuple, use zeros
                            x = torch.zeros(100, dtype=torch.float32)
                    # If input is a numpy array
                    elif isinstance(x, np.ndarray):
                        try:
                            x = torch.tensor(x, dtype=torch.float32)
                        except:
                            print("[DecisionSNN] Warning: Could not convert numpy array to tensor")
                            x = torch.zeros(100, dtype=torch.float32)
                    # If input is a list or other sequence
                    elif isinstance(x, (list, tuple)):
                        try:
                            x = torch.tensor(x, dtype=torch.float32)
                        except:
                            print("[DecisionSNN] Warning: Could not convert sequence to tensor")
                            x = torch.zeros(100, dtype=torch.float32)
                    # If input is a dict
                    elif isinstance(x, dict):
                        try:
                            # Try to extract features from the dict
                            if 'features' in x:
                                x = torch.tensor(x['features'], dtype=torch.float32)
                            elif 'activations' in x:
                                x = torch.tensor(x['activations'], dtype=torch.float32)
                            else:
                                # Use zeros as fallback
                                x = torch.zeros(100, dtype=torch.float32)
                        except:
                            print("[DecisionSNN] Warning: Could not extract features from dict")
                            x = torch.zeros(100, dtype=torch.float32)
                    # If input is not already a tensor, try to convert it
                    elif not torch.is_tensor(x):
                        try:
                            x = torch.tensor(x, dtype=torch.float32)
                        except:
                            print(f"[DecisionSNN] Warning: Could not convert input type {type(x)} to tensor")
                            x = torch.zeros(100, dtype=torch.float32)
                    
                    # Ensure x has the correct shape
                    if len(x.shape) == 1:
                        x = x.unsqueeze(0)  # Add batch dimension if needed
                    
                    # Before passing to the model, check for NaN values
                    if torch.isnan(x).any():
                        print("[DecisionSNN] Warning: NaN values detected in tensor, replacing with zeros")
                        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                    
                    try:
                        return self.model(x)
                    except Exception as e:
                        print(f"[DecisionSNN] Error in model forward pass: {e}")
                        # Return a default output based on the expected shape
                        if 'action' in self.__class__.__name__.lower():
                            return torch.zeros(1, 7)  # 7 action classes
                        else:
                            return torch.zeros(1, 1)  # 1 confidence value
            
            # 1. Action predictor network
            # Predicts: Which action channel should be activated given current state
            # Output: Probability distribution over action channels
            action_base_network = nn.Sequential(
                nn.Linear(decision_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, num_actions),  # Multi-class classification
                nn.Softmax(dim=1)            # Output probabilities
            )
            
            # Wrap with our tuple handler
            self.action_network = TupleInputWrapper(action_base_network)
            
            # 2. Confidence predictor network
            # Predicts: Confidence level for the selected action
            # Output: Single value [0,1] representing confidence
            confidence_base_network = nn.Sequential(
                nn.Linear(decision_features_size, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, 32),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output confidence [0,1]
            )
            
            # Wrap with our tuple handler
            self.confidence_network = TupleInputWrapper(confidence_base_network)
            
            # Define loss functions
            self.action_loss_fn = nn.CrossEntropyLoss()  # Multi-class classification
            self.confidence_loss_fn = nn.MSELoss()       # Confidence regression
            
            # Define optimizers
            self.action_optimizer = optim.Adam(self.action_network.parameters(), lr=0.01)
            self.confidence_optimizer = optim.Adam(self.confidence_network.parameters(), lr=0.01)
            
            print("[DecisionSNN] SNNTorch components initialized successfully")
            
        except Exception as e:
            print(f"[DecisionSNN] Warning: Error initializing SNNTorch components - {e}")
            self.torch_available = False
    
    def _register_decision_synapses(self):
        """
        Register specialized synapses for decision operations.
        This ensures the parent's plasticity won't interfere with our custom learning.
        """
        synapse_count = 0
        
        # 1. Register synapses within decision regions
        decision_regions = ['decision', 'action', 'output']
        for region_name in decision_regions:
            if region_name not in self.regions:
                continue
                
            neurons = self.regions[region_name]['neurons']
            
            # Register a subset of within-region connections
            num_synapses = min(len(neurons) * len(neurons) // 4, 200)
            for _ in range(num_synapses):
                if len(neurons) >= 2:
                    pre, post = random.sample(neurons, 2)
                    self.register_specialized_synapse(pre, post)
                    synapse_count += 1
        
        # 2. Register synapses between action channels
        for channel_name, channel_neurons in self.action_channels.items():
            # Within-channel connections
            if len(channel_neurons) >= 2:
                num_synapses = min(len(channel_neurons) * len(channel_neurons) // 4, 50)
                for _ in range(num_synapses):
                    pre, post = random.sample(channel_neurons, 2)
                    self.register_specialized_synapse(pre, post)
                    synapse_count += 1
            
            # Cross-channel inhibitory connections (competition between actions)
            for other_channel, other_neurons in self.action_channels.items():
                if other_channel != channel_name and channel_neurons and other_neurons:
                    # Register inhibitory connections between channels
                    num_synapses = 20
                    for _ in range(num_synapses):
                        pre = random.choice(channel_neurons)
                        post = random.choice(other_neurons)
                        self.register_specialized_synapse(pre, post)
                        synapse_count += 1
        
        # 3. Register input → decision region connections
        if 'sensory' in self.regions and 'decision' in self.regions:
            sensory_neurons = self.regions['sensory']['neurons']
            decision_neurons = self.regions['decision']['neurons']
            
            num_synapses = min(len(sensory_neurons) * len(decision_neurons) // 10, 300)
            for _ in range(num_synapses):
                pre = random.choice(sensory_neurons)
                post = random.choice(decision_neurons)
                self.register_specialized_synapse(pre, post)
                synapse_count += 1
        
        print(f"[DecisionSNN] Registered {synapse_count} synapses for specialized decision learning")
        return synapse_count
    
    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the module.
        This method is required for model saving with torch.save().
        
        Returns:
            dict: A dictionary containing model parameters and persistent buffers.
        """
        # First create a state dictionary with network parameters (PyTorch modules)
        state = {}
        
        # Add all PyTorch neural networks if available
        if self.torch_available:
            if hasattr(self, 'action_network') and self.action_network is not None:
                state['action_network'] = self.action_network.state_dict()
            
            if hasattr(self, 'confidence_network') and self.confidence_network is not None:
                state['confidence_network'] = self.confidence_network.state_dict()
        
        # Add custom parameters specific to DecisionSNN
        # Include parent class attributes by calling super() if the parent has state_dict
        try:
            if hasattr(super(), 'state_dict') and callable(getattr(super(), 'state_dict')):
                parent_state = super().state_dict()
                # Update our state with parent's state
                for key, value in parent_state.items():
                    state[f'parent_{key}'] = value
        except Exception as e:
            print(f"[DecisionSNN] Warning: Could not get parent state dict: {e}")
        
        # Add SNN specific parameters
        state['synaptic_weights'] = self.synaptic_weights.copy() if hasattr(self, 'synaptic_weights') else None
        state['decision_threshold'] = self.decision_threshold if hasattr(self, 'decision_threshold') else 0.7
        state['neuron_biases'] = self.neuron_biases.copy() if hasattr(self, 'neuron_biases') else None
        
        # Include action channels if they exist
        if hasattr(self, 'action_channels'):
            # Convert to serializable format (e.g., lists instead of sets if needed)
            serializable_channels = {}
            for key, value in self.action_channels.items():
                if isinstance(value, set):
                    serializable_channels[key] = list(value)
                else:
                    serializable_channels[key] = value
            state['action_channels'] = serializable_channels
        
        # Include decision context if it exists
        if hasattr(self, 'decision_context'):
            state['decision_context'] = self.decision_context
            
        # Include regions (neurons by region)
        if hasattr(self, 'regions'):
            # Make a serializable copy of regions
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
            
        # Include specialized synapses for decision-making if they exist
        if hasattr(self, 'specialized_synapses'):
            # Convert to list for serialization if it's a set
            if isinstance(self.specialized_synapses, set):
                state['specialized_synapses'] = list(self.specialized_synapses)
            else:
                state['specialized_synapses'] = self.specialized_synapses
                
        # Include recent decisions history
        if hasattr(self, 'recent_decisions'):
            state['recent_decisions'] = self.recent_decisions
            
        # Add any optimizers if they exist
        if hasattr(self, 'action_optimizer') and self.action_optimizer is not None:
            state['action_optimizer'] = self.action_optimizer.state_dict()
            
        if hasattr(self, 'confidence_optimizer') and self.confidence_optimizer is not None:
            state['confidence_optimizer'] = self.confidence_optimizer.state_dict()
            
        return state
    
    def load_state_dict(self, state_dict):
        """
        Copies parameters and buffers from state_dict into this module.
        This method is required for loading models saved with torch.save().
        
        Args:
            state_dict (dict): A dictionary containing model parameters and persistent buffers.
        """
        # Load PyTorch neural networks if available
        if self.torch_available:
            if 'action_network' in state_dict and hasattr(self, 'action_network'):
                self.action_network.load_state_dict(state_dict['action_network'])
                
            if 'confidence_network' in state_dict and hasattr(self, 'confidence_network'):
                self.confidence_network.load_state_dict(state_dict['confidence_network'])
        
        # Load parent class attributes
        parent_keys = [k for k in state_dict.keys() if k.startswith('parent_')]
        if parent_keys and hasattr(super(), 'load_state_dict') and callable(getattr(super(), 'load_state_dict')):
            try:
                # Extract parent state
                parent_state = {k[7:]: state_dict[k] for k in parent_keys}  # Remove 'parent_' prefix
                super().load_state_dict(parent_state)
            except Exception as e:
                print(f"[DecisionSNN] Warning: Could not load parent state dict: {e}")
        
        # Load SNN specific parameters
        if 'synaptic_weights' in state_dict and state_dict['synaptic_weights'] is not None:
            self.synaptic_weights = state_dict['synaptic_weights'].copy()
            
        if 'decision_threshold' in state_dict:
            self.decision_threshold = state_dict['decision_threshold']
            
        if 'neuron_biases' in state_dict and state_dict['neuron_biases'] is not None:
            self.neuron_biases = state_dict['neuron_biases'].copy()
        
        # Load action channels if they exist
        if 'action_channels' in state_dict and hasattr(self, 'action_channels'):
            # Convert lists back to sets if needed
            for key, value in state_dict['action_channels'].items():
                if isinstance(self.action_channels.get(key, []), set):
                    self.action_channels[key] = set(value)
                else:
                    self.action_channels[key] = value
        
        # Load decision context if it exists
        if 'decision_context' in state_dict and hasattr(self, 'decision_context'):
            self.decision_context = state_dict['decision_context']
            
        # Load regions (neurons by region)
        if 'regions' in state_dict and hasattr(self, 'regions'):
            # Convert lists back to sets if needed
            for region_name, region_data in state_dict['regions'].items():
                if region_name not in self.regions:
                    self.regions[region_name] = {}
                
                for k, v in region_data.items():
                    # Check if original is a set
                    if k in self.regions[region_name] and isinstance(self.regions[region_name][k], set):
                        self.regions[region_name][k] = set(v)
                    else:
                        self.regions[region_name][k] = v
                        
        # Load specialized synapses
        if 'specialized_synapses' in state_dict and hasattr(self, 'specialized_synapses'):
            if isinstance(self.specialized_synapses, set):
                # Convert list back to set
                self.specialized_synapses = set(state_dict['specialized_synapses'])
            else:
                self.specialized_synapses = state_dict['specialized_synapses']
                
        # Load recent decisions history
        if 'recent_decisions' in state_dict and hasattr(self, 'recent_decisions'):
            self.recent_decisions = state_dict['recent_decisions']
            
        # Load optimizers if they exist
        if 'action_optimizer' in state_dict and hasattr(self, 'action_optimizer'):
            self.action_optimizer.load_state_dict(state_dict['action_optimizer'])
            
        if 'confidence_optimizer' in state_dict and hasattr(self, 'confidence_optimizer'):
            self.confidence_optimizer.load_state_dict(state_dict['confidence_optimizer'])
            
        return self

    def _create_supervised_biases(self, input_activation):
        """
        Create neuron biases based on supervised network predictions to guide bio-inspired activity.
        
        Args:
            input_activation: Input activation pattern
            
        Returns:
            Bias values for each neuron
        """
        if not self.torch_available:
            return np.zeros(self.neuron_count)
        
        # First run the network to get current state
        try:
            temp_result = super().process_input(input_activation)
            
            # Handle case where result is a tuple or None
            if temp_result is None:
                print("[DecisionSNN] Warning: process_input returned None in _create_supervised_biases")
                return np.zeros(self.neuron_count)
                
            # Handle if result is a tuple
            if isinstance(temp_result, tuple):
                # Try to extract the relevant data from the tuple
                if len(temp_result) > 0 and isinstance(temp_result[0], dict):
                    temp_result = temp_result[0]
                else:
                    # Fall back to zeros if we can't extract useful data
                    print("[DecisionSNN] Warning: Received tuple result in _create_supervised_biases")
                    return np.zeros(self.neuron_count)
                    
            features = self._extract_decision_features_for_snntorch(temp_result)
        except Exception as e:
            print(f"[DecisionSNN] Error in _create_supervised_biases: {e}")
            return np.zeros(self.neuron_count)
        
        # Get supervised predictions
        try:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = self.action_network(features_tensor).squeeze(0).numpy()
        except Exception as e:
            print(f"[DecisionSNN] Error getting predictions in _create_supervised_biases: {e}")
            return np.zeros(self.neuron_count)
        
        # Create biases based on predicted probabilities
        biases = np.zeros(self.neuron_count)
        action_types = list(self.action_channels.keys())
        
        # Apply biases to action channels based on predicted probabilities
        for i, action_type in enumerate(action_types):
            if action_type in self.action_channels:
                channel_neurons = self.action_channels[action_type]
                prob = action_probs[i] if i < len(action_probs) else 0.0
                
                # Apply bias proportional to probability
                bias_strength = prob * 0.5  # Scale factor to prevent overwhelming bio-inspired dynamics
                
                for neuron in channel_neurons:
                    biases[neuron] = bias_strength
        
        return biases

    def process_input(self, input_activation):
        """
        Process input with optional supervised biasing during training.
        
        Args:
            input_activation: Input activation pattern
            
        Returns:
            Processing results from parent class
        """
        if hasattr(self, 'training_mode') and self.training_mode:
            # During training, apply supervised biases
            if hasattr(self, 'neuron_biases') and np.any(self.neuron_biases):
                # Create a copy to avoid modifying the original
                biased_activation = input_activation.copy()
                
                # Add biases
                biased_activation += self.neuron_biases
                
                # Ensure values stay in reasonable range
                biased_activation = np.clip(biased_activation, 0, 1)
                
                # Run with biased input
                result = super().process_input(biased_activation)
                
                # Clear biases after use
                self.neuron_biases = np.zeros(self.neuron_count)
                
                return result
        
        # Otherwise, run normally
        return super().process_input(input_activation)
    
    def _extract_decision_features_for_snntorch(self, network_state):
        """
        Extract decision-relevant features from the SNN state for supervised learning.
        
        Args:
            network_state: Current network state after processing input
            
        Returns:
            Feature vector for SNNTorch networks (100-dimensional)
        """
        # Handle case where network_state is a tuple or None
        if network_state is None:
            # Return default zeros if no state
            return np.zeros(100)
        
        # Handle if network_state is a tuple (sometimes returned by process_input)
        if isinstance(network_state, tuple):
            # Try to extract the relevant data from the tuple
            if len(network_state) > 0 and isinstance(network_state[0], dict):
                network_state = network_state[0]
            else:
                # Fall back to zeros if we can't extract useful data
                print("[DecisionSNN] Warning: Received tuple network state, using default features")
                return np.zeros(100)
        
        # Ensure network_state is a dictionary
        if not isinstance(network_state, dict):
            print(f"[DecisionSNN] Warning: Unexpected network_state type {type(network_state)}, using default features")
            return np.zeros(100)
            
        # Now proceed with normal feature extraction
        features = []
        
        # 1. Decision region activations
        decision_regions = ['decision', 'action', 'output']
        region_activations = []
        
        for region in decision_regions:
            if region in self.regions:
                activation = self.regions[region]['activation']
                region_activations.append(activation)
            else:
                region_activations.append(0.0)
        
        features.extend(region_activations)
        
        # 2. Action channel activations (crucial for decision making)
        channel_features = []
        action_types = [
            "factual_response", "hypothesis_selection", "memory_retrieval",
            "clarification_request", "verification_check", 
            "uncertainty_expression", "confident_assertion"
        ]
        
        for action_type in action_types:
            if action_type in self.action_channels:
                neurons = self.action_channels[action_type]
                active_neurons = network_state.get('active_neurons', set())
                
                # Calculate activation ratio for this channel
                active_count = len(set(neurons).intersection(active_neurons))
                activation_ratio = active_count / len(neurons) if neurons else 0
                channel_features.append(activation_ratio)
                
                # Also add average membrane potential
                if len(neurons) > 0:
                    # Handle missing membrane_potentials
                    if 'membrane_potentials' in network_state and network_state['membrane_potentials'] is not None:
                        try:
                            avg_potential = np.mean(network_state['membrane_potentials'][neurons])
                        except (IndexError, KeyError, TypeError):
                            avg_potential = 0.0
                        channel_features.append(avg_potential)
                    else:
                        channel_features.append(0.0)
                else:
                    channel_features.append(0.0)
            else:
                channel_features.extend([0.0, 0.0])
        
        features.extend(channel_features)
        
        # 3. Integration metrics (for consciousness-based decision making)
        features.append(network_state.get('phi', 0.0))
        features.append(network_state.get('integration', 0.0))
        features.append(network_state.get('differentiation', 0.0))
        
        # 4. Cross-region influence features
        if 'affective' in self.regions and 'decision' in self.regions:
            # Emotional influence on decision
            affective_activation = self.regions['affective']['activation']
            decision_activation = self.regions['decision']['activation']
            emotional_influence = affective_activation * decision_activation
            features.append(emotional_influence)
        else:
            features.append(0.0)
        
        if 'memory' in self.regions and 'decision' in self.regions:
            # Memory influence on decision
            memory_activation = self.regions['memory']['activation']
            decision_activation = self.regions['decision']['activation']
            memory_influence = memory_activation * decision_activation
            features.append(memory_influence)
        else:
            features.append(0.0)
        
        # 5. Decision confidence from bio-inspired network
        channel_activations = []
        for action_type in action_types:
            if action_type in self.action_channels:
                neurons = self.action_channels[action_type]
                active_count = len(set(neurons).intersection(network_state.get('active_neurons', set())))
                activation = active_count / len(neurons) if neurons else 0
                channel_activations.append(activation)
        
        if channel_activations:
            # Bio-inspired confidence based on distribution of activations
            max_activation = max(channel_activations)
            mean_activation = np.mean(channel_activations)
            
            # Higher confidence if one channel clearly dominates
            bio_confidence = max_activation - mean_activation
            features.append(bio_confidence)
        else:
            features.append(0.0)
        
        # Pad to reach target dimension
        current_dim = len(features)
        target_dim = 100
        
        if current_dim < target_dim:
            features.extend([0.0] * (target_dim - current_dim))
        elif current_dim > target_dim:
            features = features[:target_dim]
        
        # Ensure we return a properly shaped numpy array
        return np.array(features, dtype=np.float32)
    
    def train_decision(self, input_state_activation, target_action_label, confidence_target=None, learn_rate=0.02):
        """
        Train decision making using hybrid learning approach.
        
        TRAINING FLOW:
        1. Process input through bio-inspired network to get initial state
        2. Extract features and run supervised networks to get predictions
        3. Calculate supervised losses and errors
        4. Create supervised biases to guide bio-inspired dynamics
        5. Re-process input with biases to get guided network state
        6. Apply custom error-driven plasticity based on supervised error
        
        Args:
            input_state_activation: Input activation pattern representing current state
            target_action_label: Target action (index 0-6 for action channels)
            confidence_target: Optional target confidence level (0-1)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Training results with losses and accuracy
        """
        # Set training mode and control parent plasticity
        self.training_mode = True
        self.disable_parent_plasticity = True
        
        # PHASE 1: Initial processing to get features for supervised networks
        try:
            initial_result = self.process_input(input_state_activation)
            features = self._extract_decision_features_for_snntorch(initial_result)
        except Exception as e:
            print(f"[DecisionSNN] Error in initial processing: {e}")
            # Provide fallback features
            features = np.zeros(100, dtype=np.float32)
        
        # PHASE 2: Supervised learning with SNNTorch
        if self.torch_available:
            try:
                # Convert features to tensor with proper error handling
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor([target_action_label], dtype=torch.long)
                
                # Train action prediction network
                self.action_optimizer.zero_grad()
                action_output = self.action_network(features_tensor)
                action_loss = self.action_loss_fn(action_output, target_tensor)
                action_loss.backward()
                self.action_optimizer.step()
                
                # Get predicted action and its probability
                predicted_probs = action_output.detach().numpy()[0]
                predicted_action = np.argmax(predicted_probs)
                predicted_confidence = predicted_probs[predicted_action]
                
                # Train confidence network if target provided
                confidence_loss = 0.0
                if confidence_target is not None:
                    confidence_tensor = torch.tensor([[confidence_target]], dtype=torch.float32)
                    
                    self.confidence_optimizer.zero_grad()
                    confidence_output = self.confidence_network(features_tensor)
                    confidence_loss = self.confidence_loss_fn(confidence_output, confidence_tensor)
                    confidence_loss.backward()
                    self.confidence_optimizer.step()
                    
                    predicted_confidence = confidence_output.item()
            except Exception as e:
                print(f"[DecisionSNN] Error in supervised learning phase: {e}")
                # Set default values for error case
                action_loss = 0.0
                confidence_loss = 0.0
                predicted_action = random.randint(0, 6)
                predicted_confidence = 0.5
        else:
            # No supervised learning available
            action_loss = 0.0
            confidence_loss = 0.0
            predicted_action = random.randint(0, 6)
            predicted_confidence = 0.5
        
        # PHASE 3: Create supervised biases to guide bio-inspired activity
        try:
            self.neuron_biases = self._create_supervised_biases(input_state_activation)
        except Exception as e:
            print(f"[DecisionSNN] Error creating supervised biases: {e}")
            self.neuron_biases = np.zeros(self.neuron_count)
        
        # PHASE 4: Re-process with biases to get guided network state
        try:
            guided_result = self.process_input(input_state_activation)
        except Exception as e:
            print(f"[DecisionSNN] Error in guided processing: {e}")
            guided_result = {}
        
        # PHASE 5: Calculate error signals for bio-inspired learning
        action_error = 1.0 if predicted_action == target_action_label else -1.0
        
        if confidence_target is not None:
            confidence_error = confidence_target - predicted_confidence
        else:
            # Use action correctness as proxy for confidence
            confidence_error = action_error
        
        # Combined error for driving plasticity
        combined_error = action_error * (0.7 + 0.3 * abs(confidence_error))
        
        # PHASE 6: Apply custom error-driven plasticity
        try:
            weights_updated = self._update_decision_weights(
                input_state_activation, 
                target_action_label,
                combined_error,
                learn_rate
            )
        except Exception as e:
            print(f"[DecisionSNN] Error in weight update: {e}")
            weights_updated = 0
        
        # PHASE 7: Apply threshold plasticity
        threshold_updates = 0
        
        try:
            # Get target action channel neurons
            action_types = list(self.action_channels.keys())
            if target_action_label < len(action_types):
                target_channel_name = action_types[target_action_label]
                target_neurons = self.action_channels.get(target_channel_name, [])
                
                # Update thresholds based on action error
                if target_neurons:
                    # If we got the action wrong, make target neurons more excitable
                    if predicted_action != target_action_label:
                        error_signal = 0.3  # Positive -> decrease threshold
                        threshold_updates += self._apply_threshold_plasticity(
                            target_neurons,
                            error_signal,
                            learning_rate=learn_rate * 0.5
                        )
                        
                        # Also make incorrectly chosen action neurons less excitable
                        if predicted_action < len(action_types):
                            incorrect_channel = action_types[predicted_action]
                            incorrect_neurons = self.action_channels.get(incorrect_channel, [])
                            if incorrect_neurons:
                                error_signal = -0.2  # Negative -> increase threshold
                                threshold_updates += self._apply_threshold_plasticity(
                                    incorrect_neurons,
                                    error_signal,
                                    learning_rate=learn_rate * 0.3
                                )
                    else:
                        # Correct action - adjust based on confidence
                        if confidence_target is not None:
                            confidence_error = confidence_target - predicted_confidence
                            # If confidence is too low, make neurons slightly more excitable
                            if confidence_error > 0.2:
                                error_signal = 0.1
                            else:
                                error_signal = -0.05  # Slight increase in threshold for stability
                            
                            threshold_updates += self._apply_threshold_plasticity(
                                target_neurons,
                                error_signal,
                                learning_rate=learn_rate * 0.3
                            )
        except Exception as e:
            print(f"[DecisionSNN] Error in threshold plasticity: {e}")
            threshold_updates = 0
        
        # Reset flags
        self.disable_parent_plasticity = False
        self.training_mode = False
        
        return {
            'action_loss': float(action_loss) if isinstance(action_loss, (int, float)) else action_loss.item(),
            'confidence_loss': float(confidence_loss) if isinstance(confidence_loss, (int, float)) else confidence_loss.item(),
            'predicted_action': predicted_action,
            'target_action': target_action_label,
            'predicted_confidence': predicted_confidence,
            'action_error': action_error,
            'weights_updated': weights_updated,
            'threshold_updates': threshold_updates,  # Add this to output
            'accuracy': 1.0 if predicted_action == target_action_label else 0.0
        }
    
    def _update_decision_weights(self, input_activation, target_action, error, learn_rate):
        """
        Update synaptic weights based on decision error using custom plasticity rules.
        
        PLASTICITY RULES:
        1. Input → Target Channel: Strengthen/weaken based on error
        - Positive error: Strengthen connections from active inputs to target action
        - Negative error: Weaken connections to incorrectly chosen action
        
        2. Within-Channel Coherence: Strengthen internal connections for correct actions
        - Makes action channels more self-sustaining when correct
        
        3. Cross-Channel Competition: Increase inhibition when wrong action chosen
        - Creates winner-take-all dynamics between action channels
        
        Args:
            input_activation: Input pattern that led to decision
            target_action: Correct action index that should have been chosen
            error: Combined error signal (positive=correct, negative=incorrect)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Number of synaptic weights updated
        """
        weights_updated = 0
        
        # Get active input neurons (sensory evidence)
        active_inputs = np.where(input_activation > 0.3)[0]
        
        # Get target action channel neurons
        action_types = list(self.action_channels.keys())
        if target_action >= len(action_types):
            return 0
            
        target_channel_name = action_types[target_action]
        target_neurons = self.action_channels.get(target_channel_name, [])
        
        if not target_neurons:
            return 0
        
        # 1. Update input → target action channel connections
        # These connections encode which inputs should trigger which actions
        for input_neuron in active_inputs:
            # Sample subset of target neurons for efficiency
            target_sample = random.sample(target_neurons, min(10, len(target_neurons)))
            
            for target_neuron in target_sample:
                # Only update registered synapses
                if (input_neuron, target_neuron) not in self.specialized_learning_synapses:
                    continue
                
                current_weight = self.synaptic_weights[input_neuron, target_neuron]
                
                # Weight change proportional to:
                # - error magnitude (how wrong/right we were)
                # - input activation (how strong this input was)
                weight_change = learn_rate * error * input_activation[input_neuron]
                
                # Apply bounds to prevent extreme weights
                new_weight = max(-1.0, min(1.0, current_weight + weight_change))
                
                if abs(new_weight - current_weight) > 0.001:
                    self.synaptic_weights[input_neuron, target_neuron] = new_weight
                    weights_updated += 1
        
        # 2. Update within-channel connections for target action
        # Only strengthen internal coherence for correct decisions
        if error > 0:  # Correct decision
            target_sample = random.sample(target_neurons, min(20, len(target_neurons)))
            
            for i, pre in enumerate(target_sample):
                for post in random.sample(target_neurons, min(5, len(target_neurons))):
                    if pre != post and (pre, post) in self.specialized_learning_synapses:
                        current_weight = self.synaptic_weights[pre, post]
                        
                        # Strengthen recurrent connections within correct action channel
                        # This makes the channel more likely to sustain its activity
                        weight_change = learn_rate * error * 0.5
                        new_weight = min(1.0, current_weight + weight_change)
                        
                        if abs(new_weight - current_weight) > 0.001:
                            self.synaptic_weights[pre, post] = new_weight
                            weights_updated += 1
        
        # 3. Update inhibitory connections between channels
        # Only increase competition when wrong action was chosen
        if error < 0:  # Incorrect decision
            # Find which channel was incorrectly activated
            bio_activations = []
            for action_type in action_types:
                neurons = self.action_channels.get(action_type, [])
                active_count = len(set(neurons).intersection(self.active_neurons_cache))
                activation = active_count / len(neurons) if neurons else 0
                bio_activations.append(activation)
            
            if bio_activations:
                incorrect_action = np.argmax(bio_activations)
                
                if incorrect_action != target_action:
                    incorrect_neurons = self.action_channels.get(action_types[incorrect_action], [])
                    
                    # Strengthen inhibition from target to incorrect channel
                    # This creates mutual inhibition for winner-take-all dynamics
                    target_sample = random.sample(target_neurons, min(10, len(target_neurons)))
                    incorrect_sample = random.sample(incorrect_neurons, min(10, len(incorrect_neurons)))
                    
                    for target_neuron in target_sample:
                        for incorrect_neuron in incorrect_sample:
                            if (target_neuron, incorrect_neuron) in self.specialized_learning_synapses:
                                current_weight = self.synaptic_weights[target_neuron, incorrect_neuron]
                                
                                # Make connection more inhibitory (more negative)
                                # This suppresses the incorrect channel when target is active
                                weight_change = learn_rate * abs(error) * 0.5
                                new_weight = max(-1.0, current_weight - weight_change)
                                
                                if abs(new_weight - current_weight) > 0.001:
                                    self.synaptic_weights[target_neuron, incorrect_neuron] = new_weight
                                    weights_updated += 1
        
        return weights_updated
    
    def _reallocate_neurons_to_decision(self, target_region, count):
        """Reallocate neurons from other regions to decision region"""
        # Target size for decision regions
        target_total_size = int(self.neuron_count * 0.3)
        current_size = len(self.regions[target_region]['neurons'])
        
        if current_size >= target_total_size:
            return  # Already sufficient
        
        # Find donor regions
        donor_candidates = []
        for region_name, region in self.regions.items():
            if region_name not in ['decision', 'action', 'output']:
                # Non-decision regions can donate neurons
                max_donate = len(region['neurons']) // 2  # Can donate up to half
                if max_donate > 0:
                    donor_candidates.append((region_name, max_donate))
        
        # Sort by donation capacity
        donor_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Reallocate neurons
        neurons_needed = target_total_size - current_size
        
        for donor_name, max_donate in donor_candidates:
            if neurons_needed <= 0:
                break
                
            # Calculate how many to take
            to_take = min(max_donate, neurons_needed)
            
            # Take neurons from donor
            donor_neurons = self.regions[donor_name]['neurons']
            taken_neurons = donor_neurons[:to_take]
            self.regions[donor_name]['neurons'] = donor_neurons[to_take:]
            
            # Add to target region
            self.regions[target_region]['neurons'].extend(taken_neurons)
            neurons_needed -= to_take
            
            print(f"[DecisionSNN] Reallocated {to_take} neurons from {donor_name} to {target_region}")

    # Add a method to process text input using the standardized processor
    def process_text_decision(self, text_input, timesteps=20):
        """
        Process text input for decision-making using the standardized bidirectional processor.
        
        Args:
            text_input: Text input for decision
            timesteps: Number of timesteps for spike patterns
            
        Returns:
            Processed spike patterns for decision making
        """
        # Use the inherited process_text_input method from EnhancedSpikingCore
        spike_patterns = self.process_text_input(text_input, timesteps)
        return spike_patterns
    
    # Add a method to generate text output from decision spikes
    def generate_decision_text(self, spike_patterns, max_length=100):
        """
        Generate text output from decision spike patterns using the standardized bidirectional processor.
        
        Args:
            spike_patterns: Spike patterns from decision process
            max_length: Maximum length of generated text
            
        Returns:
            Generated text from decision process
        """
        # Use the standardized parent method for text generation
        try:
            text_output = super().generate_text_output(
                spike_patterns,
                max_length=max_length,
                remove_special_tokens=True
            )
            return text_output
        except Exception as e:
            print(f"[DecisionSNN] Error generating text: {e}")
            return f"Error generating decision text: {e}"