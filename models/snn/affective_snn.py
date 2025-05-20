# affective_snn.py
from collections import defaultdict
from models.snn.enhanced_snn import EnhancedSpikingCore
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate

class AffectiveSNN(EnhancedSpikingCore):
    """
    SNN specialized for emotional processing and learning (inspired by limbic system).
    
    This implementation uses learned neural representations for affective states rather
    than hardcoded dimensions. Emotions emerge from patterns of neural activity through
    learning and experience, similar to how biological systems process emotions.
    """
    
    def __init__(self, neuron_count=400, topology_type="flexible", vector_dim=300, bidirectional_processor=None):
        """
        Initialize the affective SNN with specialized parameters.
        
        Args:
            neuron_count: Total number of neurons in the network
            topology_type: Type of network topology
            vector_dim: Dimension of the vector space for token embeddings
            bidirectional_processor: Optional BidirectionalProcessor instance or None to create a new one
        """
        super().__init__(
            neuron_count=neuron_count, 
            topology_type=topology_type,
            model_type="affective",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )
        
        # Configure for affective processing (higher connectivity and plasticity)
        self.connection_density = 0.2     # Higher density for emotional processing
        self.region_density = 0.7         # High internal connectivity
        self.plasticity_rate = 0.03       # Enhanced plasticity for emotional learning
        
        # Emotion representation attributes 
        self.emotion_assemblies = {}      # Neural assemblies that learn to represent emotions
        self.valence_arousal_mapping = {} # Mapping of emotions to valence-arousal space (for decoding)
        
        # State tracking
        self.current_affective_state = None
        
        # Setup affective-specific regions and assemblies
        self._configure_affective_regions()
        self._initialize_emotion_assemblies()
        
        # Initialize snntorch specific components
        self._setup_snntorch_components()
        
        # Initialize training counter
        self.training_iterations = 0
    
    def _initialize_emotion_assemblies(self):
        """
        Initialize neural assemblies for basic emotions.
        These assemblies will learn to represent specific emotions through experience.
        
        Improved to properly register specialized synapses with parent class.
        """
        # Basic emotions in valence-arousal space for representation and decoding
        # These values help interpret the learned representations but aren't hardcoded into neurons
        self.valence_arousal_mapping = {
            'joy': (0.8, 0.7),         # Positive valence, high arousal
            'sadness': (-0.7, -0.5),   # Negative valence, low arousal
            'anger': (-0.8, 0.8),      # Negative valence, high arousal
            'fear': (-0.6, 0.9),       # Negative valence, high arousal
            'disgust': (-0.7, 0.3),    # Negative valence, medium arousal
            'surprise': (0.1, 0.8),    # Neutral/slight positive valence, high arousal
            'trust': (0.6, -0.2),      # Positive valence, low-medium arousal
            'anticipation': (0.5, 0.5)  # Positive valence, medium arousal
        }
        
        # Create initial neural assemblies for each emotion
        # These will be refined through learning
        self._create_emotion_assemblies()
        
        # Register all synapses connecting to these assemblies for specialized learning
        synapse_count = 0
        for emotion, assembly in self.emotion_assemblies.items():
            # For each neuron in the assembly
            for post in assembly['neurons']:
                # Register synapses from sensory and language regions
                for region_name in ['sensory', 'language']:
                    if region_name in self.regions:
                        # Register a subset of connections to prevent overspecialization
                        region_neurons = self.regions[region_name]['neurons']
                        # Sample a subset of neurons proportional to emotion assembly size
                        sample_size = min(len(region_neurons), len(assembly['neurons']) * 10)
                        input_neurons = random.sample(region_neurons, sample_size)
                        
                        # Register each synapse with the parent using the correct method
                        for pre in input_neurons:
                            self.register_specialized_synapse(pre, post)
                            synapse_count += 1
        
        print(f"[AffectiveSNN] Registered {synapse_count} synapses for specialized learning")
    
    def simulate_spiking(self, input_activation, timesteps=10):
        """
        Override to incorporate supervised learning biases while letting parent handle plasticity.
        
        Removed the global disabling of parent plasticity in favor of synapse-specific control.
        
        Args:
            input_activation: Initial activation pattern
            timesteps: Number of simulation steps
            
        Returns:
            List of spike events per timestep
        """
        # If in training mode, apply temporary biases to the network
        if hasattr(self, 'training_mode') and self.training_mode and hasattr(self, 'emotion_network') and hasattr(self, 'va_network'):
            # Create biases from supervised networks
            biased_activation = self._create_supervised_biases(input_activation)
            
            # Run the parent's simulation (plasticity will skip registered synapses automatically)
            spike_patterns = super().simulate_spiking(biased_activation, timesteps)
        else:
            # Run standard simulation (plasticity will skip registered synapses automatically)
            spike_patterns = super().simulate_spiking(input_activation, timesteps)
        
        # Store spike patterns for later use
        self.spike_patterns = spike_patterns
        
        return spike_patterns
    
    def process_input(self, input_activation):
        """
        Process input through the SNN with our custom learning approach.
        Simplified to rely on synapse registration with parent.
        
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
            
            # Run standard processing (registered synapses will be skipped by parent)
            result = super().process_input(biased_activation)
            
            # Clear biases after use (they're only for this simulation)
            self.neuron_biases = np.zeros(self.neuron_count)
        else:
            # Run standard processing (registered synapses will be skipped by parent)
            result = super().process_input(input_activation)
        
        # Integrate our learned weights from emotion assemblies
        # This ensures the network has the latest weights for simulation
        self._integrate_emotion_weights()
        
        # Add spike patterns to result for state decoding
        result['spike_patterns'] = self.spike_patterns if hasattr(self, 'spike_patterns') else []
        
        return result
    
    def train_emotion_assembly(self, input_features, emotion, learn_rate=None):
        """
        Train a specific emotion assembly using error-driven learning.
        Simplified to rely on proper synapse registration.
        
        Args:
            input_features: Input features dict
            emotion: Target emotion to activate
            learn_rate: Learning rate for weight updates
            
        Returns:
            Training metrics
        """
        # PHASE 1: PREPARE INPUTS AND TARGETS
        # Encode the input features
        input_activation = self.encode_affective_input(input_features)
        
        # Set training mode to enable supervised guidance
        self.training_mode = True
        
        # Process through bio-inspired SNN
        result = self.process_input(input_activation)
        spike_patterns = result.get('spike_patterns', [])
        
        # Extract bio-inspired features for supervised networks
        bio_inspired_features = self._extract_bio_inspired_features(result)
        input_tensor = torch.tensor(bio_inspired_features, dtype=torch.float32).unsqueeze(0)
        
        # Create supervised learning targets
        # Target for emotion classification (one-hot encoding)
        emotion_index = list(self.valence_arousal_mapping.keys()).index(emotion)
        emotion_target = torch.zeros(1, len(self.valence_arousal_mapping))
        emotion_target[0, emotion_index] = 1.0
        
        # Target for valence-arousal prediction
        if emotion in self.valence_arousal_mapping:
            target_valence, target_arousal = self.valence_arousal_mapping[emotion]
        else:
            # Fallback to input features if mapping not available
            target_valence = input_features.get('sentiment', 0.0)
            target_arousal = input_features.get('intensity', 0.5)
            
        va_target = torch.tensor([[target_valence, target_arousal]], dtype=torch.float32)
        
        # Set target activation for bio-inspired learning
        target_activation = 0.9  # High target for correct emotion
        
        # PHASE 2: SUPERVISED LEARNING COMPONENT
        # Train emotion classifier network
        self.emotion_optimizer.zero_grad()
        emotion_output = self.emotion_network(input_tensor)
        emotion_loss = self.emotion_loss_fn(emotion_output, torch.argmax(emotion_target, dim=1))
        emotion_loss.backward()
        self.emotion_optimizer.step()
        
        # Train valence-arousal network
        self.va_optimizer.zero_grad()
        va_output = self.va_network(input_tensor)
        va_loss = self.va_loss_fn(va_output, va_target)
        va_loss.backward()
        self.va_optimizer.step()
        
        # Record supervised training stats
        self.training_stats['emotion_losses'].append(emotion_loss.item())
        self.training_stats['va_losses'].append(va_loss.item())
        
        # Calculate accuracy for monitoring
        predicted_emotion = torch.argmax(emotion_output, dim=1).item()
        accuracy = 1.0 if predicted_emotion == emotion_index else 0.0
        self.training_stats['accuracy'].append(accuracy)
        
        # PHASE 3: BIO-INSPIRED LEARNING COMPONENT
        # Get actual activation from the bio-inspired network
        actual_activation = self._get_assembly_activation(emotion, spike_patterns)
        
        # Calculate error signal for bio-inspired learning
        bio_error = target_activation - actual_activation
        
        # Calculate combined error signal with supervised component
        supervised_error_weight = 0.3  # Weight for supervised error component
        max_expected_loss = 5.0  # Typical scale of loss values
        
        # Convert and normalize the error components
        emotion_error_component = (emotion_loss.item() / max_expected_loss) * supervised_error_weight
        va_error_component = (va_loss.item() / max_expected_loss) * supervised_error_weight
        
        # Combine error signals
        combined_error = bio_error - emotion_error_component - va_error_component
        
        # Ensure combined error is in a reasonable range
        combined_error = max(-1.0, min(1.0, combined_error))
        
        # PHASE 4: DIRECT WEIGHT UPDATES
        # Use provided learning rate or get from assembly
        if learn_rate is None:
            learn_rate = self.emotion_assemblies[emotion].get('learning_rate', 0.05)
            
        # Update weights using our custom error-driven approach
        weights_updated = self._update_assembly_weights(emotion, input_activation, combined_error, learn_rate)
        
        # Also train valence-arousal regions with the same input based on target dimensions
        va_weights_updated = self._train_valence_arousal_regions(
            input_activation, 
            target_valence, 
            target_arousal, 
            learn_rate=learn_rate
        )
        
        # PHASE 5: UPDATE ASSEMBLY PROPERTIES
        # Adjust confidence based on prediction accuracy
        current_confidence = self.emotion_assemblies[emotion].get('confidence', 0.5)
        
        # Increase confidence if error is small, decrease if error is large
        confidence_change = (1 - abs(bio_error)) * 0.02  # Small adjustments based on error
        
        # Apply confidence update with bounds
        new_confidence = current_confidence + confidence_change
        self.emotion_assemblies[emotion]['confidence'] = min(0.9, max(0.1, new_confidence))
        
        # Get all assembly activations for tracking
        assembly_activations = {}
        for em in self.emotion_assemblies:
            assembly_activations[em] = self._get_assembly_activation(em, spike_patterns)
        
        # Disable training mode
        self.training_mode = False
        
        # Update training iteration counter
        self.training_iterations += 1
        
        # Return detailed results
        return {
            "success": True,
            "emotion": emotion,
            "target_activation": target_activation,
            "actual_activation": actual_activation,
            "bio_error": bio_error,
            "combined_error": combined_error,
            "emotion_loss": emotion_loss.item(),
            "va_loss": va_loss.item(),
            "accuracy": accuracy,
            "weights_updated": weights_updated + va_weights_updated,
            "confidence": self.emotion_assemblies[emotion]['confidence'],
            "assembly_activations": assembly_activations
        }

    def _setup_snntorch_components(self):
        """Set up snntorch neural network components for supervised learning"""
        # Beta is the decay rate for the neuron's spike trace
        beta = 0.95
        
        # Use surrogate gradient for backpropagation through spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Define the supervised learning networks:
        
        # Get size of affective regions or assemblies to use as input size for supervised networks
        affective_regions_size = self._get_affective_regions_size()
        
        # 1. Emotion classifier network (maps encoded inputs to emotion labels)
        self.emotion_network = nn.Sequential(
            nn.Linear(affective_regions_size, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(64, len(self.valence_arousal_mapping))
        )
        
        # 2. Valence-arousal network (maps encoded inputs to continuous values)
        self.va_network = nn.Sequential(
            nn.Linear(affective_regions_size, 128), 
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Linear(64, 2)  # 2 outputs: valence and arousal
        )
        
        # Define loss functions
        self.emotion_loss_fn = nn.CrossEntropyLoss()  # For classification
        self.va_loss_fn = nn.MSELoss()  # For regression
        
        # Define optimizers
        self.emotion_optimizer = optim.Adam(self.emotion_network.parameters(), lr=0.01)
        self.va_optimizer = optim.Adam(self.va_network.parameters(), lr=0.01)
        
        # Training state
        self.training_mode = False
        self.training_stats = {
            'emotion_losses': [],
            'va_losses': [],
            'accuracy': []
        }
    
    def _get_affective_regions_size(self):
        """Calculate the number of neurons in affective regions for SNN input size"""
        total_size = 0
        
        # Count neurons in affective processing regions
        for region_name in ['affective', 'emotional', 'limbic', 
                           'valence_processing', 'arousal_processing', 'emotion_integration']:
            if region_name in self.regions:
                total_size += len(self.regions[region_name]['neurons'])
        
        # Ensure we have a minimum size
        if total_size < 50:
            total_size = self.neuron_count  # Fallback to using all neurons
            
        return total_size
        
    def _configure_affective_regions(self):
        """Configure SNN regions specifically for affective processing"""
        # Ensure primary affective regions exist with sufficient neurons
        affective_regions = ['affective', 'emotional', 'limbic']
        region_found = False
        
        for region_name in affective_regions:
            if region_name in self.regions:
                region_found = True
                # Ensure affective regions have sufficient neurons
                current_size = len(self.regions[region_name]['neurons'])
                target_size = int(self.neuron_count * 0.5)  # Allocate 50% to emotional processing
                
                if current_size < target_size:
                    # Reallocate neurons from non-affective regions
                    self._reallocate_neurons_to_affective(region_name, target_size - current_size)
        
        # If no affective regions found, create one
        if not region_found:
            self._create_affective_region()
            
        # Create specialized sub-regions for emotional processing
        self._create_affective_subregions()
    
    def _create_affective_region(self):
        """Create affective region if none exists"""
        # Allocate half of all neurons to new affective region
        available_neurons = list(range(self.neuron_count))
        region_size = self.neuron_count // 2
        
        # Remove neurons already assigned to other regions
        for region in self.regions.values():
            for neuron in region['neurons']:
                if neuron in available_neurons:
                    available_neurons.remove(neuron)
        
        # Create new region with available neurons
        self.regions['affective'] = {
            'neurons': available_neurons[:region_size],
            'activation': 0.0,
            'recurrent': True,
            'plasticity_factor': 1.2  # Higher plasticity for emotional learning
        }
    
    def _create_affective_subregions(self):
        """Create specialized subregions for different emotional functions"""
        # First check if we have an affective or emotional region
        main_region = None
        for region_name in ['affective', 'emotional', 'limbic']:
            if region_name in self.regions:
                main_region = region_name
                break
                
        if not main_region:
            return  # No affective region to subdivide
            
        # Get neurons from the main affective region
        affective_neurons = self.regions[main_region]['neurons']
        if len(affective_neurons) < 100:  # Need enough neurons to subdivide
            return
            
        # Create subregions by dividing the available neurons
        neurons_per_subregion = len(affective_neurons) // 3
        
        # 1. Valence processing subregion (positive/negative affect)
        valence_neurons = affective_neurons[:neurons_per_subregion]
        
        # 2. Arousal processing subregion (intensity of affect)
        arousal_neurons = affective_neurons[neurons_per_subregion:2*neurons_per_subregion]
        
        # 3. Emotion integration subregion (combines valence and arousal)
        integration_neurons = affective_neurons[2*neurons_per_subregion:]
        
        # Create the subregions
        self.regions['valence_processing'] = {
            'neurons': valence_neurons,
            'activation': 0.0,
            'recurrent': True,
            'plasticity_factor': 1.2
        }
        
        self.regions['arousal_processing'] = {
            'neurons': arousal_neurons,
            'activation': 0.0,
            'recurrent': True,
            'plasticity_factor': 1.2
        }
        
        self.regions['emotion_integration'] = {
            'neurons': integration_neurons,
            'activation': 0.0,
            'recurrent': True,
            'plasticity_factor': 1.3  # Higher plasticity for integration
        }
        
        # Update the main affective region to include only integration neurons
        self.regions[main_region]['neurons'] = integration_neurons
    
    def _reallocate_neurons_to_affective(self, target_region, count):
        """
        Reallocate neurons from other regions to affective region.
        This allows dynamic adaptation of neural resources based on task demands.
        
        Args:
            target_region: Target affective region name
            count: Number of neurons to reallocate
            
        Returns:
            Number of neurons actually reallocated
        """
        # Find regions that can donate neurons (exclude affective regions)
        affective_regions = ['affective', 'emotional', 'limbic', 
                            'valence_processing', 'arousal_processing', 'emotion_integration']
        donor_regions = [r for r in self.regions if r not in affective_regions]
        
        if not donor_regions:
            return 0
        
        # Prioritize donor regions based on activity levels and importance
        prioritized_donors = []
        for region_name in donor_regions:
            region = self.regions[region_name]
            
            # Calculate recent activity level
            recent_activity = region.get('activation', 0.0)
            
            # Calculate importance score (lower means more available for donation)
            # Regions with more neurons and less activity are preferred donors
            importance_score = recent_activity * 2.0  # Activity is a strong factor
            
            # Prioritize regions based on function (spare critical regions)
            if region_name in ['sensory', 'memory']:
                importance_score += 0.3  # These are somewhat important
            elif region_name in ['higher_cognition', 'output']:
                importance_score += 0.6  # These are very important
            
            # Add to prioritized list
            prioritized_donors.append((region_name, importance_score))
        
        # Sort donors by importance (lowest first - best donors)
        prioritized_donors.sort(key=lambda x: x[1])
        
        # Calculate how many neurons each region can donate
        neuron_shares = {}
        total_donor_neurons = 0
        
        for region_name, _ in prioritized_donors:
            region = self.regions[region_name]
            region_neurons = region['neurons']
            
            # Calculate maximum donation based on region size and importance
            region_size = len(region_neurons)
            
            # Ensure regions maintain minimum viable size
            min_size = 20  # Minimum neurons a region should keep
            max_donation = max(0, region_size - min_size)
            
            # Limit donation to a percentage of neurons
            max_percent = 0.3  # Maximum 30% of neurons can be donated
            max_donation = min(max_donation, int(region_size * max_percent))
            
            if max_donation > 0:
                neuron_shares[region_name] = max_donation
                total_donor_neurons += max_donation
        
        # If not enough neurons available, take what we can get
        if total_donor_neurons < count:
            count = total_donor_neurons
            
        if count == 0:
            return 0  # Nothing to reallocate
        
        # Allocate neurons from each region proportionally
        neurons_reallocated = 0
        reallocated_neurons = []
        
        remaining_count = count
        for region_name, _ in prioritized_donors:
            if region_name not in neuron_shares or remaining_count <= 0:
                continue
                
            max_donation = neuron_shares[region_name]
            
            # Calculate proportional share
            if total_donor_neurons > 0:
                region_contribution = min(max_donation, 
                                        int((max_donation / total_donor_neurons) * count))
            else:
                region_contribution = 0
                
            if region_contribution <= 0:
                continue
                
            # Take neurons from region
            region_neurons = self.regions[region_name]['neurons']
            
            # Take preferentially from the end of the list to maintain 
            # structural integrity of the region
            donor_neurons = region_neurons[-region_contribution:]
            
            # Update region's neurons
            self.regions[region_name]['neurons'] = region_neurons[:-region_contribution]
            
            # Add to reallocated collection
            reallocated_neurons.extend(donor_neurons)
            neurons_reallocated += region_contribution
            remaining_count -= region_contribution
            
            # Update connectivity structures
            self._update_connectivity_after_reallocation(region_name, donor_neurons)
        
        # Add reallocated neurons to target region
        if target_region in self.regions:
            self.regions[target_region]['neurons'].extend(reallocated_neurons)
            
            # Initialize weights for new connections based on region identity
            self._initialize_weights_for_reallocated_neurons(target_region, reallocated_neurons)
        
        print(f"[AffectiveSNN] Reallocated {neurons_reallocated} neurons to {target_region}")
        return neurons_reallocated

    def _update_connectivity_after_reallocation(self, source_region, reallocated_neurons):
        """
        Update connectivity structures after neurons are reallocated.
        Ensures the network remains functional after neuron reassignment.
        
        Args:
            source_region: Region neurons were taken from
            reallocated_neurons: List of neurons that were reallocated
        """
        # Update region connectivity if necessary
        if hasattr(self, 'region_connectivity'):
            # No direct changes needed to region_connectivity structure
            # as it works with region names, not neuron indices
            pass
        
        # If we have specialized assemblies, update them
        if hasattr(self, 'assemblies'):
            for assembly_name, assembly in self.assemblies.items():
                # Remove relocated neurons from assemblies
                if 'neurons' in assembly:
                    assembly['neurons'] = [n for n in assembly['neurons'] 
                                        if n not in reallocated_neurons]
        
        # Update concept mappings if necessary
        for concept, neurons in self.concept_mappings.items():
            # Remove relocated neurons from concept mappings
            self.concept_mappings[concept] = [n for n in neurons 
                                            if n not in reallocated_neurons]

    def _initialize_weights_for_reallocated_neurons(self, target_region, reallocated_neurons):
        """Initialize weights for newly reallocated neurons in their new region"""
        if not reallocated_neurons:
            return
            
        # Find appropriate connection targets based on target region
        target_neurons = []
        
        if target_region == 'affective':
            # Connect to sensory and memory regions
            for region_name in ['sensory', 'memory', 'higher_cognition']:
                if region_name in self.regions:
                    target_neurons.extend(self.regions[region_name]['neurons'][:50])  # Limit to 50 neurons
        elif target_region == 'valence_processing':
            # Connect to sensory and affective regions
            for region_name in ['sensory', 'affective']:
                if region_name in self.regions:
                    target_neurons.extend(self.regions[region_name]['neurons'][:50])
        elif target_region == 'arousal_processing':
            # Connect to sensory and affective regions
            for region_name in ['sensory', 'affective']:
                if region_name in self.regions:
                    target_neurons.extend(self.regions[region_name]['neurons'][:50])
        elif target_region == 'emotion_integration':
            # Connect to valence, arousal, and higher cognition
            for region_name in ['valence_processing', 'arousal_processing', 'higher_cognition']:
                if region_name in self.regions:
                    target_neurons.extend(self.regions[region_name]['neurons'][:50])
        
        # Initialize weights
        if target_neurons:
            # For each reallocated neuron, connect to a subset of target neurons
            for neuron in reallocated_neurons:
                # Connect to ~20% of target neurons
                connection_count = max(5, len(target_neurons) // 5)
                connected_targets = random.sample(target_neurons, min(connection_count, len(target_neurons)))
                
                for target in connected_targets:
                    # Initialize with small random weights
                    weight = 0.1 * random.uniform(-1, 1)
                    self.synaptic_weights[neuron, target] = weight
                    self.synaptic_weights[target, neuron] = weight * 0.8  # Slightly weaker reverse connection

    def _create_emotion_assemblies(self):
        """Create learnable neural assemblies for emotions using available neurons"""
        # Find suitable regions for emotion assemblies
        assembly_regions = ['emotion_integration', 'affective', 'emotional', 'limbic']
        assembly_neurons = []
        
        # Collect neurons from eligible regions
        for region_name in assembly_regions:
            if region_name in self.regions:
                # Use a portion of each region's neurons
                region_neurons = self.regions[region_name]['neurons']
                contribution = min(len(region_neurons) // 2, 30)  # Up to half, max 30
                assembly_neurons.extend(region_neurons[:contribution])
        
        # If no suitable regions, use neurons from any available region
        if not assembly_neurons:
            for region in self.regions.values():
                contribution = min(len(region['neurons']) // 4, 20)  # Up to 1/4, max 20
                assembly_neurons.extend(region['neurons'][:contribution])
        
        # Create assemblies for each emotion
        emotions = list(self.valence_arousal_mapping.keys())
        neurons_per_emotion = max(10, len(assembly_neurons) // len(emotions))
        
        for i, emotion in enumerate(emotions):
            # Take a slice of neurons for this emotion
            start_idx = i * neurons_per_emotion
            end_idx = min(start_idx + neurons_per_emotion, len(assembly_neurons))
            
            if start_idx < len(assembly_neurons):
                neurons = assembly_neurons[start_idx:end_idx]
                
                # Create assembly with initial random weights
                self.emotion_assemblies[emotion] = {
                    'neurons': neurons,
                    'weights': {},  # Will be learned
                    'baseline_activation': 0.1,  # Initial activation threshold
                    'learning_rate': 0.05,
                    'confidence': 0.5  # Initialized with moderate confidence
                }
                
                # Initialize incoming weights
                self._initialize_assembly_weights(emotion)
    
    def _initialize_assembly_weights(self, emotion):
        """Initialize weights for an emotion assembly with small random values"""
        if emotion not in self.emotion_assemblies:
            return
            
        assembly = self.emotion_assemblies[emotion]
        neurons = assembly['neurons']
        
        # Find potential input neurons
        input_regions = ['sensory', 'feature_extraction', 'perception']
        input_neurons = []
        
        for region in input_regions:
            if region in self.regions:
                input_neurons.extend(self.regions[region]['neurons'])
        
        # If no input regions, use other non-emotional regions
        if not input_neurons:
            for region_name, region in self.regions.items():
                if region_name not in ['valence_processing', 'arousal_processing', 
                                     'emotion_integration', 'affective', 'emotional', 'limbic']:
                    input_neurons.extend(region['neurons'])
        
        # Initialize random weights for each neuron in the assembly
        weights = {}
        for neuron in neurons:
            # Connect to a subset of input neurons
            num_connections = min(50, len(input_neurons) // 2)
            connected_inputs = random.sample(input_neurons, num_connections)
            
            # Create weight dictionary for this neuron
            neuron_weights = {}
            for input_neuron in connected_inputs:
                # Small initial weights, positive or negative
                weight = 0.1 * random.uniform(-1, 1)
                neuron_weights[input_neuron] = weight
            
            weights[neuron] = neuron_weights
            
        # Store weights in the assembly
        self.emotion_assemblies[emotion]['weights'] = weights
    
    def encode_affective_input(self, input_features):
        """
        Encode affective input features into neural activation pattern using structured encoding.
        Improved implementation with smoother neural population coding for better learning.
        
        Args:
            input_features: Dictionary of affective features
                
        Returns:
            Neural activation pattern (numpy array)
        """
        # Initialize activation pattern
        activation = np.zeros(self.neuron_count)
        
        # Get neurons from input-receiving regions
        input_layer_neurons = []
        for region_name in ['sensory', 'valence_processing', 'arousal_processing']:
            if region_name in self.regions:
                input_layer_neurons.extend(self.regions[region_name]['neurons'])
        
        if not input_layer_neurons:
            print("[AffectiveSNN] Warning: No input neurons available for encoding.")
            return activation
            
        input_size = len(input_layer_neurons)
        
        # Define sections of the input pattern for different feature types
        sections = {
            'sentiment': (0, int(input_size * 0.2)),  # 20% for sentiment value
            'intensity': (int(input_size * 0.2), int(input_size * 0.4)),  # 20% for intensity value
            'keywords': (int(input_size * 0.4), int(input_size * 0.7)),  # 30% for keyword presence
            'detected_emotions': (int(input_size * 0.7), input_size)  # 30% for detected discrete emotions
        }
        
        # 1. Encode sentiment value with smoother population coding
        if 'sentiment' in input_features:
            sentiment = input_features['sentiment']  # -1.0 to 1.0
            start, end = sections['sentiment']
            sentiment_range = end - start
            
            # Create a Gaussian bump centered at the position corresponding to the sentiment value
            # Map sentiment from [-1,1] to position in the neuron array
            sentiment_pos = start + int((sentiment + 1) / 2.0 * sentiment_range)
            
            # Width of activation corresponds to ~10% of the range (smoother activation)
            width = max(3, sentiment_range // 10)  # Ensure minimum width for small networks
            
            # Apply Gaussian activation around the center position
            for i in range(start, end):
                if start <= i < end:  # Ensure we stay within section bounds
                    neuron_idx = input_layer_neurons[i]
                    # Apply Gaussian function centered at sentiment_pos
                    distance = abs(i - sentiment_pos)
                    # Smoother activation with wider Gaussian (division by width/2)
                    activation[neuron_idx] = np.exp(-0.5 * (distance / (width/2))**2)
                
        # 2. Encode intensity with smoother population coding
        if 'intensity' in input_features:
            intensity = input_features['intensity']  # 0.0 to 1.0
            start, end = sections['intensity']
            intensity_range = end - start
            
            # Create a Gaussian bump centered at the position corresponding to the intensity value
            intensity_pos = start + int(intensity * intensity_range)
            
            # Width of activation - wider for smoother gradients
            width = max(3, intensity_range // 10)
            
            # Apply Gaussian activation around the center position
            for i in range(start, end):
                if start <= i < end:  # Ensure we stay within section bounds
                    neuron_idx = input_layer_neurons[i]
                    # Apply Gaussian function centered at intensity_pos
                    distance = abs(i - intensity_pos)
                    activation[neuron_idx] = np.exp(-0.5 * (distance / (width/2))**2)
        
        # 3. Encode emotional keywords with consistent mapping
        if 'emotional_keywords' in input_features and input_features['emotional_keywords']:
            keywords = input_features['emotional_keywords']
            start, end = sections['keywords']
            keyword_neurons_available = end - start
            
            # Use consistent hash mapping for keywords
            for keyword in keywords:
                # Create a more stable hash mapping that preserves similarity
                # Similar keywords should activate similar neurons
                keyword_hash = hash(keyword) % keyword_neurons_available
                base_position = start + keyword_hash
                
                # Activate a cluster of neurons around the base position
                width = 5  # Fixed width for keyword activation
                for offset in range(-width, width+1):
                    pos = base_position + offset
                    if start <= pos < end:  # Ensure we stay within section bounds
                        neuron_idx = input_layer_neurons[pos]
                        # Gaussian activation, stronger at center, weaker at edges
                        activation[neuron_idx] = 0.8 * np.exp(-0.5 * (offset / (width/2))**2)
        
        # 4. Encode detected emotions with consistent mapping and similarity preservation
        if 'detected_emotions' in input_features and input_features['detected_emotions']:
            detected = input_features['detected_emotions']  # emotion name -> confidence
            start, end = sections['detected_emotions']
            emotion_range = end - start
            
            # Assign consistent positions for standard emotions
            standard_emotions = {
                'joy': 0.1,       # Position within section (0.0-1.0)
                'sadness': 0.2,
                'anger': 0.3,
                'fear': 0.4,
                'disgust': 0.5,
                'surprise': 0.6,
                'trust': 0.7,
                'anticipation': 0.8
            }
            
            for emotion, confidence in detected.items():
                # Get position in range - use standard mapping if available, otherwise hash
                if emotion in standard_emotions:
                    relative_pos = standard_emotions[emotion]
                else:
                    # Consistent hash for unknown emotions
                    relative_pos = (hash(emotion) % 100) / 100.0
                    
                # Convert to absolute position in neuron array
                emotion_pos = start + int(relative_pos * emotion_range)
                
                # Width scales with confidence - higher confidence = wider activation
                width = max(2, int(5 * confidence))
                
                # Apply Gaussian activation around the center position
                for offset in range(-width, width+1):
                    pos = emotion_pos + offset
                    if start <= pos < end:  # Ensure we stay within section bounds
                        neuron_idx = input_layer_neurons[pos]
                        # Gaussian activation scaled by confidence
                        # Stronger activation for higher confidence
                        activation[neuron_idx] = confidence * np.exp(-0.5 * (offset / (width/2))**2)
        
        return activation
    
    def evaluate_affective_state(self, input_features):
        """
        Process affective input features and determine emotional state.
        Integrates both supervised learning and bio-inspired dynamics.
        
        Args:
            input_features: Dictionary of affective features
            
        Returns:
            Dictionary with affective state evaluation
        """
        # Encode input features into neural activation pattern
        input_activation = self.encode_affective_input(input_features)
        
        # Process through the bio-inspired SNN to get neural dynamics
        result = super().process_input(input_activation)
        spike_patterns = result.get('spike_patterns', [])
        
        # Extract bio-inspired neural activity for supervised network input
        bio_inspired_features = self._extract_bio_inspired_features(result)
        
        # Convert to tensor for supervised networks
        input_tensor = torch.tensor(bio_inspired_features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            # Get emotion predictions
            emotion_output = self.emotion_network(input_tensor)
            emotion_probs = torch.softmax(emotion_output, dim=1)[0]
            
            # Get valence-arousal predictions
            va_output = self.va_network(input_tensor)[0]
            predicted_valence = va_output[0].item()
            predicted_arousal = va_output[1].item()
        
        # Combine supervised and bio-inspired outputs for a hybrid decoding
        affective_state = self._hybrid_decode_affective_state(
            spike_patterns, 
            emotion_probs,
            predicted_valence, 
            predicted_arousal
        )
        
        # Store current state for influence_processing
        self.current_affective_state = affective_state
        
        return affective_state

    def _extract_bio_inspired_features(self, result):
        """
        Extract relevant features from bio-inspired SNN state for supervised learning.
        Enhanced to capture more meaningful temporal and activation patterns.
        
        Args:
            result: Result from bio-inspired SNN processing
            
        Returns:
            Feature vector for supervised networks
        """
        # Get relevant affective regions
        affective_regions = ['affective', 'emotional', 'limbic', 
                            'valence_processing', 'arousal_processing', 'emotion_integration']
        
        # Initialize feature vector
        feature_vector = []
        
        # Get spike patterns for temporal feature extraction
        spike_patterns = result.get('spike_patterns', [])
        
        # 1. REGION ACTIVATION FEATURES
        # Get overall region activations (first-order features)
        for region_name in affective_regions:
            if region_name in self.regions:
                region_activation = self.regions[region_name].get('activation', 0.0)
                feature_vector.append(region_activation)
        
        # 2. EMOTION ASSEMBLY FEATURES
        # Get both activation level and temporal pattern features from emotion assemblies
        for emotion in self.emotion_assemblies:
            # Basic activation level
            assembly_activity = self._get_assembly_activation(emotion, spike_patterns)
            feature_vector.append(assembly_activity)
            
            # Add temporal features - when does this assembly activate?
            if spike_patterns:
                # Early activation (first third of simulation)
                early_steps = len(spike_patterns) // 3
                early_activation = self._get_assembly_activation_in_range(emotion, spike_patterns, 0, early_steps)
                feature_vector.append(early_activation)
                
                # Late activation (last third of simulation)
                late_start = 2 * len(spike_patterns) // 3
                late_activation = self._get_assembly_activation_in_range(emotion, spike_patterns, late_start, len(spike_patterns))
                feature_vector.append(late_activation)
        
        # 3. NEURAL ACTIVITY PATTERN FEATURES
        # Collect all neurons involved in affective processing
        affective_neurons = []
        for region_name in affective_regions:
            if region_name in self.regions:
                affective_neurons.extend(self.regions[region_name]['neurons'])
        
        # Calculate activation features from spike patterns
        if affective_neurons and spike_patterns:
            # Track neuron spike counts and timing
            neuron_spike_counts = {}
            first_spike_times = {}
            
            # Process spike patterns
            for t, time_step in enumerate(spike_patterns):
                for neuron_idx, _ in time_step:
                    if neuron_idx in affective_neurons:
                        # Count spikes
                        neuron_spike_counts[neuron_idx] = neuron_spike_counts.get(neuron_idx, 0) + 1
                        
                        # Record first spike time if not already recorded
                        if neuron_idx not in first_spike_times:
                            first_spike_times[neuron_idx] = t
            
            # Calculate features from spike patterns
            
            # A. Average spike rate across regions
            for region_name in affective_regions:
                if region_name in self.regions:
                    region_neurons = self.regions[region_name]['neurons']
                    if region_neurons:
                        # Calculate average spike rate for this region
                        region_total = sum(neuron_spike_counts.get(n, 0) for n in region_neurons)
                        avg_spike_rate = region_total / (len(region_neurons) * len(spike_patterns))
                        feature_vector.append(avg_spike_rate)
            
            # B. Synchrony measures between regions (important for emotional integration)
            region_pairs = [
                ('valence_processing', 'emotion_integration'),
                ('arousal_processing', 'emotion_integration'),
                ('affective', 'higher_cognition')
            ]
            
            for region1, region2 in region_pairs:
                if region1 in self.regions and region2 in self.regions:
                    # Calculate synchrony as temporal correlation of activity
                    synchrony = self._calculate_region_synchrony(
                        self.regions[region1]['neurons'],
                        self.regions[region2]['neurons'],
                        spike_patterns
                    )
                    feature_vector.append(synchrony)
            
            # C. First spike timing features (captures rapid emotional responses)
            # Average first spike time for valence and arousal regions
            for region_name in ['valence_processing', 'arousal_processing']:
                if region_name in self.regions:
                    region_neurons = self.regions[region_name]['neurons']
                    active_neurons = [n for n in region_neurons if n in first_spike_times]
                    
                    if active_neurons:
                        avg_first_spike = sum(first_spike_times[n] for n in active_neurons) / len(active_neurons)
                        # Normalize by total simulation time
                        normalized_time = avg_first_spike / len(spike_patterns)
                        feature_vector.append(normalized_time)
                    else:
                        # No neurons spiked - maximum latency
                        feature_vector.append(1.0)
        
        # If feature vector is empty (unlikely), use a fallback
        if not feature_vector:
            # Use valence and arousal from region activations as minimal features
            valence = self.regions.get('valence_processing', {}).get('activation', 0.0)
            arousal = self.regions.get('arousal_processing', {}).get('activation', 0.0)
            feature_vector = [valence, arousal]
        
        # Pad or truncate to expected size
        affective_regions_size = self._get_affective_regions_size()
        
        if len(feature_vector) < affective_regions_size:
            # Pad with zeros
            feature_vector.extend([0.0] * (affective_regions_size - len(feature_vector)))
        elif len(feature_vector) > affective_regions_size:
            # Truncate
            feature_vector = feature_vector[:affective_regions_size]
        
        return np.array(feature_vector)

    def _get_assembly_activation_in_range(self, emotion, spike_patterns, start_idx, end_idx):
        """Calculate activation level for an emotion assembly within a specific time range"""
        if emotion not in self.emotion_assemblies or start_idx >= end_idx:
            return 0.0
            
        assembly = self.emotion_assemblies[emotion]
        neurons = set(assembly['neurons'])
        
        # Only consider spike patterns in the specified range
        relevant_patterns = spike_patterns[start_idx:end_idx]
        if not relevant_patterns:
            return 0.0
        
        # Count active neurons
        active_count = 0
        for time_step in relevant_patterns:
            for neuron_idx, _ in time_step:
                if neuron_idx in neurons:
                    active_count += 1
        
        # Normalize by size and time steps
        denominator = len(neurons) * len(relevant_patterns) if neurons and relevant_patterns else 1
        activation = active_count / denominator
        
        # Apply baseline threshold
        threshold = assembly.get('baseline_activation', 0.1)
        normalized_activation = max(0, (activation - threshold) / (1 - threshold)) if threshold < 1 else 0
        
        return normalized_activation

    def _calculate_region_synchrony(self, neurons1, neurons2, spike_patterns):
        """Calculate neuronal firing synchrony between two sets of neurons"""
        if not neurons1 or not neurons2 or not spike_patterns:
            return 0.0
        
        # Convert neurons to sets for efficient lookup
        neurons1_set = set(neurons1)
        neurons2_set = set(neurons2)
        
        # Track activity over time
        activity1 = np.zeros(len(spike_patterns))
        activity2 = np.zeros(len(spike_patterns))
        
        # Calculate activity per timestep
        for t, time_step in enumerate(spike_patterns):
            # Count active neurons in each set
            active1 = sum(1 for neuron_idx, _ in time_step if neuron_idx in neurons1_set)
            active2 = sum(1 for neuron_idx, _ in time_step if neuron_idx in neurons2_set)
            
            # Normalize by number of neurons in each set
            activity1[t] = active1 / len(neurons1)
            activity2[t] = active2 / len(neurons2)
        
        # Calculate correlation coefficient if there's variance in the data
        if np.var(activity1) > 0 and np.var(activity2) > 0:
            corrcoef = np.corrcoef(activity1, activity2)[0, 1]
            # Convert correlation (-1 to 1) to synchrony measure (0 to 1)
            synchrony = (corrcoef + 1) / 2
        else:
            # No variance - check if both are always active or inactive
            if np.mean(activity1) > 0 and np.mean(activity2) > 0:
                synchrony = 1.0  # Both regions always active
            else:
                synchrony = 0.0  # At least one region never active
        
        return synchrony

    def _hybrid_decode_affective_state(self, spike_patterns, emotion_probs, predicted_valence, predicted_arousal):
        """
        Hybrid decoding that combines supervised learning results with bio-inspired dynamics.
        Uses confidence-based dynamic blending instead of fixed blend factors.
        
        Args:
            spike_patterns: Spike patterns from bio-inspired network
            emotion_probs: Emotion probabilities from supervised network
            predicted_valence: Valence prediction from supervised network
            predicted_arousal: Arousal prediction from supervised network
            
        Returns:
            Dictionary representing the current affective state
        """
        # 1. Get bio-inspired emotion activations
        bio_emotion_activations = {}
        
        for emotion, assembly in self.emotion_assemblies.items():
            assembly_activity = self._get_assembly_activation(emotion, spike_patterns)
            if assembly_activity > 0:
                bio_emotion_activations[emotion] = assembly_activity
        
        # 2. Get supervised emotion activations
        supervised_emotion_activations = {}
        emotions = list(self.valence_arousal_mapping.keys())
        
        for i, emotion in enumerate(emotions):
            activation = emotion_probs[i].item()
            if activation > 0.1:  # Only include reasonably confident predictions
                supervised_emotion_activations[emotion] = activation
        
        # 3. Get valence/arousal from bio-inspired dynamics
        bio_valence, bio_arousal = self._decode_valence_arousal_from_spikes(spike_patterns)
        
        # 4. DYNAMIC CONFIDENCE-BASED BLENDING
        # Calculate confidence metrics for each system
        
        # Bio-inspired system confidence - based on:
        # - Overall activation levels
        # - Consistency of activations (one dominant emotion vs spread across many)
        # - Integration metric (phi value if available)
        bio_dominant_emotion = None
        bio_max_activation = 0.0
        
        for emotion, activation in bio_emotion_activations.items():
            if activation > bio_max_activation:
                bio_max_activation = activation
                bio_dominant_emotion = emotion
        
        # Contrast between highest and average activations
        if bio_emotion_activations:
            average_activation = sum(bio_emotion_activations.values()) / len(bio_emotion_activations)
            activation_contrast = bio_max_activation - average_activation
        else:
            activation_contrast = 0.0
        
        # Integration metric (from IIT-inspired calculations)
        integration_metric = hasattr(self, 'phi') and self.phi or 0.5
        
        # Combined bio-inspired confidence
        bio_confidence = (
            0.4 * bio_max_activation +      # Higher activation = more confidence
            0.3 * activation_contrast +     # Clear dominant emotion = more confidence
            0.3 * integration_metric        # Higher integration = more confidence
        )
        
        # Supervised system confidence - based on:
        # - Softmax probability of top prediction
        # - Contrast between top and second predictions
        # - Recent training accuracy
        sup_max_prob = max(emotion_probs).item() if len(emotion_probs) > 0 else 0.0
        
        # Calculate contrast between top two predictions
        if len(emotion_probs) > 1:
            sorted_probs, _ = torch.sort(emotion_probs, descending=True)
            sup_prob_contrast = sorted_probs[0].item() - sorted_probs[1].item()
        else:
            sup_prob_contrast = 0.0
        
        # Recent training accuracy (if available)
        if hasattr(self, 'training_stats') and 'accuracy' in self.training_stats and self.training_stats['accuracy']:
            recent_accuracy = sum(self.training_stats['accuracy'][-10:]) / min(10, len(self.training_stats['accuracy']))
        else:
            recent_accuracy = 0.5  # Default value
        
        # Combined supervised confidence
        sup_confidence = (
            0.5 * sup_max_prob +        # Higher probability = more confidence
            0.3 * sup_prob_contrast +   # Clear winner = more confidence
            0.2 * recent_accuracy       # Better accuracy = more confidence
        )
        
        # Normalize confidences to sum to 1.0
        total_confidence = bio_confidence + sup_confidence
        if total_confidence > 0:
            bio_weight = bio_confidence / total_confidence
            sup_weight = sup_confidence / total_confidence
        else:
            # Default weights if confidence calculation fails
            bio_weight = 0.5
            sup_weight = 0.5
        
        # Apply minimum thresholds to ensure both systems contribute
        bio_weight = max(0.3, min(0.7, bio_weight))
        sup_weight = 1.0 - bio_weight
        
        # Blend emotions
        emotion_activations = {}
        all_emotions = set(list(bio_emotion_activations.keys()) + list(supervised_emotion_activations.keys()))
        
        for emotion in all_emotions:
            bio_value = bio_emotion_activations.get(emotion, 0)
            sup_value = supervised_emotion_activations.get(emotion, 0)
            
            # Weighted average
            blended_value = (bio_value * bio_weight) + (sup_value * sup_weight)
            if blended_value > 0.1:  # Threshold to reduce noise
                emotion_activations[emotion] = blended_value
        
        # Blend valence/arousal
        blended_valence = (bio_valence * bio_weight) + (predicted_valence * sup_weight)
        blended_arousal = (bio_arousal * bio_weight) + (predicted_arousal * sup_weight)
        
        # Find dominant emotion
        dominant_emotion = None
        max_activation = 0.0
        
        for emotion, activation in emotion_activations.items():
            if activation > max_activation:
                max_activation = activation
                dominant_emotion = emotion
        
        # Calculate overall intensity
        emotional_intensity = max_activation if emotion_activations else 0.0
        dimensional_intensity = (abs(blended_valence) + blended_arousal) / 2
        overall_intensity = 0.7 * emotional_intensity + 0.3 * dimensional_intensity
        
        # Create affective state result with confidence information
        return {
            "emotions": emotion_activations,
            "valence": blended_valence,
            "arousal": blended_arousal,
            "dominant_emotion": dominant_emotion,
            "intensity": overall_intensity,
            "confidence": max_activation,  # Use max emotion activation as confidence
            "bio_confidence": bio_confidence,
            "supervised_confidence": sup_confidence,
            "blend_weights": {
                "bio_weight": bio_weight,
                "supervised_weight": sup_weight
            }
        }

    def _decode_valence_arousal_from_spikes(self, spike_patterns):
        """Helper method to decode valence and arousal from spike patterns"""
        # Get valence and arousal region neurons for decoding
        valence_neurons = self.regions.get('valence_processing', {}).get('neurons', [])
        arousal_neurons = self.regions.get('arousal_processing', {}).get('neurons', [])
        
        # Prepare activity maps (for position-weighted decoding)
        valence_neuron_activity = {}
        arousal_neuron_activity = {}
        
        # Record activity for each neuron in these regions
        for time_step in spike_patterns:
            for neuron_idx, _ in time_step:
                if neuron_idx in valence_neurons:
                    valence_neuron_activity[neuron_idx] = valence_neuron_activity.get(neuron_idx, 0) + 1
                elif neuron_idx in arousal_neurons:
                    arousal_neuron_activity[neuron_idx] = arousal_neuron_activity.get(neuron_idx, 0) + 1
        
        # Decode valence using position-weighted activity (learned representation)
        decoded_valence = 0.0
        if valence_neurons and valence_neuron_activity:
            # Sort neurons by their index
            sorted_valence_neurons = sorted(list(valence_neurons))
            total_activity = sum(valence_neuron_activity.values())
            
            if total_activity > 0:
                # Position-based decoding - calculate center of mass
                valence_sum = 0.0
                
                for i, neuron in enumerate(sorted_valence_neurons):
                    if neuron in valence_neuron_activity:
                        # Position in range (0 to 1)
                        relative_pos = i / max(1, len(sorted_valence_neurons) - 1)
                        # Map to valence range (-1 to 1)
                        valence_value = 1.0 - 2.0 * relative_pos
                        # Weight by activity
                        activity = valence_neuron_activity[neuron]
                        valence_sum += valence_value * activity
                
                # Compute weighted average
                decoded_valence = valence_sum / total_activity
        
        # Decode arousal using position-weighted activity (learned representation)
        decoded_arousal = 0.0
        if arousal_neurons and arousal_neuron_activity:
            # Sort neurons by their index
            sorted_arousal_neurons = sorted(list(arousal_neurons))
            total_activity = sum(arousal_neuron_activity.values())
            
            if total_activity > 0:
                # Position-based decoding - calculate center of mass
                arousal_sum = 0.0
                
                for i, neuron in enumerate(sorted_arousal_neurons):
                    if neuron in arousal_neuron_activity:
                        # Position in range (0 to 1)
                        relative_pos = i / max(1, len(sorted_arousal_neurons) - 1)
                        # Map to arousal range (0 to 1)
                        arousal_value = 1.0 - relative_pos
                        # Weight by activity
                        activity = arousal_neuron_activity[neuron]
                        arousal_sum += arousal_value * activity
                
                # Compute weighted average
                decoded_arousal = arousal_sum / total_activity
        
        return decoded_valence, decoded_arousal
    
    def _update_assembly_weights(self, emotion, input_activation, error, learn_rate):
        """
        Update weights in emotion assembly using our custom error-driven learning.
        This is our primary mechanism for updating synaptic weights.
        
        Args:
            emotion: Emotion assembly to update
            input_activation: Input activation pattern
            error: Error signal (difference between target and actual activation)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Number of weights updated
        """
        if emotion not in self.emotion_assemblies:
            return 0
            
        assembly = self.emotion_assemblies[emotion]
        assembly_neurons = assembly['neurons']
        weights = assembly['weights']
        
        # Skip if error is too small to be meaningful
        if abs(error) < 0.05 or not weights:
            return 0
            
        weights_updated = 0
        
        # Find active input neurons - using a threshold to focus on strongly active inputs
        active_inputs = np.where(input_activation > 0.3)[0]
        
        # Register these neurons as managed by our custom learning rule
        # This helps prevent conflicts with parent's plasticity
        for neuron in assembly_neurons:
            self.custom_learning_neurons.add(neuron)
        
        # Update weights for each neuron in the assembly
        for neuron in assembly_neurons:
            if neuron not in weights:
                weights[neuron] = {}
                
            neuron_weights = weights[neuron]
            
            # Update weights from active inputs
            for input_neuron in active_inputs:
                # Skip if outside range
                if input_neuron >= self.neuron_count:
                    continue
                    
                # Get current weight or initialize
                current_weight = neuron_weights.get(input_neuron, 0.0)
                
                # Compute weight change based on error and input activity
                input_strength = input_activation[input_neuron]
                weight_change = learn_rate * error * input_strength
                
                # Apply change
                new_weight = current_weight + weight_change
                
                # Apply weight bounds to maintain stability
                new_weight = max(-1.0, min(1.0, new_weight))
                
                # Update weight if changed significantly
                if abs(new_weight - current_weight) > 0.001:
                    neuron_weights[input_neuron] = new_weight
                    weights_updated += 1
                    
                    # Also update main synaptic weight matrix
                    if hasattr(self, 'synaptic_weights'):
                        self.synaptic_weights[input_neuron, neuron] = new_weight
        
        return weights_updated

    def _train_valence_arousal_regions(self, input_activation, target_valence, target_arousal, learn_rate=0.05):
        """
        Train valence and arousal regions using our custom error-driven weight updates.
        
        Args:
            input_activation: Input activation pattern
            target_valence: Target valence value (-1.0 to 1.0)
            target_arousal: Target arousal value (0.0 to 1.0)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Number of weights updated
        """
        valence_neurons = self.regions.get('valence_processing', {}).get('neurons', [])
        arousal_neurons = self.regions.get('arousal_processing', {}).get('neurons', [])
        
        if not valence_neurons or not arousal_neurons:
            return 0
        
        # Register these neurons as managed by our custom learning rules
        for neuron in valence_neurons:
            self.custom_learning_neurons.add(neuron)
        for neuron in arousal_neurons:
            self.custom_learning_neurons.add(neuron)
        
        # Run simulation to get current region activity with plasticity disabled
        result = self.process_input(input_activation)  # Using our overridden method
        
        # Get current region activations (bio-inspired)
        valence_region_activation = self.regions.get('valence_processing', {}).get('activation', 0.0)
        arousal_region_activation = self.regions.get('arousal_processing', {}).get('activation', 0.0)
        
        # Map target valence and arousal to region activation space (0-1)
        target_valence_activation = (target_valence + 1.0) / 2.0  # Map -1,1 to 0,1
        target_arousal_activation = target_arousal  # Assuming arousal is already 0-1
        
        # Calculate direct error signals
        valence_error = target_valence_activation - valence_region_activation
        arousal_error = target_arousal_activation - arousal_region_activation
        
        weights_updated = 0
        
        # Only update weights if errors are significant
        if abs(valence_error) > 0.05 or abs(arousal_error) > 0.05:
            # Get active input neurons
            active_inputs = np.where(input_activation > 0.2)[0]
            
            # Update weights in valence region if error is significant
            if abs(valence_error) > 0.05:
                for valence_neuron in valence_neurons:
                    for input_neuron in active_inputs:
                        if input_neuron < self.neuron_count and valence_neuron < self.neuron_count:
                            # Apply weight change based on error and input activity
                            current_weight = self.synaptic_weights[input_neuron, valence_neuron]
                            
                            # Learning rule: delta_w = learning_rate * error * input_activity
                            input_activity = input_activation[input_neuron]
                            weight_change = learn_rate * valence_error * input_activity
                            
                            new_weight = current_weight + weight_change
                            
                            # Clamp weights to reasonable range
                            new_weight = max(-1.0, min(1.0, new_weight))
                            
                            # Only count as update if significant change
                            if abs(new_weight - current_weight) > 0.001:
                                self.synaptic_weights[input_neuron, valence_neuron] = new_weight
                                weights_updated += 1
            
            # Update weights in arousal region if error is significant
            if abs(arousal_error) > 0.05:
                for arousal_neuron in arousal_neurons:
                    for input_neuron in active_inputs:
                        if input_neuron < self.neuron_count and arousal_neuron < self.neuron_count:
                            # Apply weight change based on error and input activity
                            current_weight = self.synaptic_weights[input_neuron, arousal_neuron]
                            
                            # Same learning rule as valence
                            input_activity = input_activation[input_neuron]
                            weight_change = learn_rate * arousal_error * input_activity
                            
                            new_weight = current_weight + weight_change
                            
                            # Clamp weights
                            new_weight = max(-1.0, min(1.0, new_weight))
                            
                            # Only count as update if significant change
                            if abs(new_weight - current_weight) > 0.001:
                                self.synaptic_weights[input_neuron, arousal_neuron] = new_weight
                                weights_updated += 1
        
        return weights_updated
    
    def _update_assembly_weights(self, emotion, input_activation, error, learn_rate):
        """
        Update weights in emotion assembly using error-driven learning.
        
        Args:
            emotion: Emotion assembly to update
            input_activation: Input activation pattern
            error: Error signal (difference between target and actual activation)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Number of weights updated
        """
        if emotion not in self.emotion_assemblies:
            return 0
            
        assembly = self.emotion_assemblies[emotion]
        assembly_neurons = assembly['neurons']
        weights = assembly['weights']
        
        # Skip if no error or no weights
        if abs(error) < 0.05 or not weights:
            return 0
            
        weights_updated = 0
        
        # Find active input neurons - using a threshold to focus on strongly active inputs
        active_inputs = np.where(input_activation > 0.3)[0]
        
        # Update weights for each neuron in the assembly
        for neuron in assembly_neurons:
            if neuron not in weights:
                weights[neuron] = {}
                
            neuron_weights = weights[neuron]
            
            # Update weights from active inputs
            for input_neuron in active_inputs:
                # Skip if outside range
                if input_neuron >= self.neuron_count:
                    continue
                    
                # Get current weight or initialize
                current_weight = neuron_weights.get(input_neuron, 0.0)
                
                # Compute weight change based on error and input activity
                # Key part: weight change is proportional to:
                #  1. Learning rate
                #  2. Error signal (from combined bio + supervised error)
                #  3. Input activation strength (stronger inputs → larger changes)
                input_strength = input_activation[input_neuron]
                weight_change = learn_rate * error * input_strength
                
                # Apply change
                new_weight = current_weight + weight_change
                
                # Apply weight bounds to maintain stability
                new_weight = max(-1.0, min(1.0, new_weight))
                
                # Update weight if changed significantly
                if abs(new_weight - current_weight) > 0.001:
                    neuron_weights[input_neuron] = new_weight
                    weights_updated += 1
                    
                    # CRITICAL: Also update main synaptic weight matrix
                    # This ensures learning affects the bio-inspired dynamics
                    if hasattr(self, 'synaptic_weights'):
                        self.synaptic_weights[input_neuron, neuron] = new_weight
        
        return weights_updated
    
    def _train_valence_arousal_regions(self, input_activation, target_valence, target_arousal, learn_rate=0.05):
        """
        Train valence and arousal regions to respond to target sentiment/intensity values.
        Now integrates with supervised learning signals.
        
        Args:
            input_activation: Input activation pattern
            target_valence: Target valence value (-1.0 to 1.0)
            target_arousal: Target arousal value (0.0 to 1.0)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Number of weights updated
        """
        valence_neurons = self.regions.get('valence_processing', {}).get('neurons', [])
        arousal_neurons = self.regions.get('arousal_processing', {}).get('neurons', [])
        
        if not valence_neurons or not arousal_neurons:
            return 0
        
        # Run simulation to get current region activity
        result = super().process_input(input_activation)
        
        # Extract features for supervised networks
        bio_inspired_features = self._extract_bio_inspired_features(result)
        input_tensor = torch.tensor(bio_inspired_features, dtype=torch.float32).unsqueeze(0)
        
        # Run through supervised VA network to get predictions
        with torch.no_grad():
            va_output = self.va_network(input_tensor)[0]
            predicted_valence = va_output[0].item()
            predicted_arousal = va_output[1].item()
        
        # Use both bio-inspired and supervised errors for training
        # Get current region activations (bio-inspired)
        valence_region_activation = self.regions.get('valence_processing', {}).get('activation', 0.0)
        arousal_region_activation = self.regions.get('arousal_processing', {}).get('activation', 0.0)
        
        # Map target valence and arousal to region activation space (0-1)
        target_valence_activation = (target_valence + 1.0) / 2.0  # Map -1,1 to 0,1
        target_arousal_activation = target_arousal  # Assuming arousal is already 0-1
        
        # Calculate bio-inspired errors
        bio_valence_error = target_valence_activation - valence_region_activation
        bio_arousal_error = target_arousal_activation - arousal_region_activation
        
        # Calculate supervised errors
        supervised_valence_error = target_valence - predicted_valence
        supervised_arousal_error = target_arousal - predicted_arousal
        
        # Blend errors (50/50 blend)
        valence_error = 0.5 * bio_valence_error + 0.5 * supervised_valence_error * 0.5  # Scale supervised error
        arousal_error = 0.5 * bio_arousal_error + 0.5 * supervised_arousal_error * 0.5  # Scale supervised error
        
        weights_updated = 0
        
        # Update valence region weights if significant error
        if abs(valence_error) > 0.05:
            # Get active input neurons
            active_inputs = np.where(input_activation > 0.2)[0]
            
            # Update weights to valence region neurons
            for valence_neuron in valence_neurons:
                for input_neuron in active_inputs:
                    if input_neuron < self.neuron_count and valence_neuron < self.neuron_count:
                        # Apply weight change based on error and input activity
                        current_weight = self.synaptic_weights[input_neuron, valence_neuron]
                        
                        # Learning rule: delta_w = learning_rate * error * input_activity
                        input_activity = input_activation[input_neuron]
                        weight_change = learn_rate * valence_error * input_activity
                        
                        new_weight = current_weight + weight_change
                        
                        # Clamp weights
                        self.synaptic_weights[input_neuron, valence_neuron] = max(-1.0, min(1.0, new_weight))
                        weights_updated += 1
        
        # Update arousal region weights if significant error
        if abs(arousal_error) > 0.05:
            # Get active input neurons
            active_inputs = np.where(input_activation > 0.2)[0]
            
            # Update weights to arousal region neurons
            for arousal_neuron in arousal_neurons:
                for input_neuron in active_inputs:
                    if input_neuron < self.neuron_count and arousal_neuron < self.neuron_count:
                        # Apply weight change based on error and input activity
                        current_weight = self.synaptic_weights[input_neuron, arousal_neuron]
                        
                        # Learning rule: delta_w = learning_rate * error * input_activity
                        input_activity = input_activation[input_neuron]
                        weight_change = learn_rate * arousal_error * input_activity
                        
                        new_weight = current_weight + weight_change
                        
                        # Clamp weights
                        self.synaptic_weights[input_neuron, arousal_neuron] = max(-1.0, min(1.0, new_weight))
                        weights_updated += 1
        
        return weights_updated
    
    def _get_assembly_activation(self, emotion, spike_patterns):
        """Calculate activation level for an emotion assembly from spike patterns"""
        if emotion not in self.emotion_assemblies:
            return 0.0
            
        assembly = self.emotion_assemblies[emotion]
        neurons = set(assembly['neurons'])
        
        # Count active neurons
        active_count = 0
        for time_step in spike_patterns:
            for neuron_idx, _ in time_step:
                if neuron_idx in neurons:
                    active_count += 1
        
        # Normalize by size and time steps
        denominator = len(neurons) * len(spike_patterns) if neurons and spike_patterns else 1
        activation = active_count / denominator
        
        # Apply baseline threshold
        threshold = assembly.get('baseline_activation', 0.1)
        normalized_activation = max(0, (activation - threshold) / (1 - threshold)) if threshold < 1 else 0
        
        return normalized_activation
    
    def _create_supervised_biases(self, input_activation):
        """
        Create network biases based on supervised network predictions 
        to help guide the bio-inspired network.
        """
        # Run a temporary simulation to get features for supervised networks
        temp_result = super().process_input(input_activation)
        bio_inspired_features = self._extract_bio_inspired_features(temp_result)
        input_tensor = torch.tensor(bio_inspired_features, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions from supervised networks
        with torch.no_grad():
            # Emotion classification 
            emotion_output = self.emotion_network(input_tensor)
            emotion_probs = torch.softmax(emotion_output, dim=1)[0]
            max_emotion_idx = torch.argmax(emotion_probs).item()
            max_emotion = list(self.valence_arousal_mapping.keys())[max_emotion_idx]
            
            # Valence-arousal prediction
            va_output = self.va_network(input_tensor)
            predicted_valence = va_output[0, 0].item()
            predicted_arousal = va_output[0, 1].item()
        
        # Create a copy of the input activation
        biased_activation = input_activation.copy()
        
        # Apply biases to relevant neurons based on supervised predictions
        if hasattr(self, 'neuron_biases'):
            self.neuron_biases = np.zeros(self.neuron_count)
        else:
            self.neuron_biases = np.zeros(self.neuron_count)
            
        # Bias emotion assembly neurons for predicted emotion
        if max_emotion in self.emotion_assemblies:
            assembly = self.emotion_assemblies[max_emotion]
            for neuron in assembly['neurons']:
                # Apply small bias to increase likelihood of firing
                self.neuron_biases[neuron] = -0.1  # Negative bias lowers threshold
                
        # Add biases to input activation
        biased_activation += self.neuron_biases
        
        # Ensure values stay in reasonable range
        biased_activation = np.clip(biased_activation, 0, 1)
        
        return biased_activation

    def _bias_network_dynamics(self, emotion, valence, arousal):
        """
        Apply biases to the network based on supervised learning predictions
        to guide the bio-inspired dynamics.
        
        Args:
            emotion: Predicted dominant emotion
            valence: Predicted valence value
            arousal: Predicted arousal value
        """
        # 1. Bias emotion assembly neurons
        if emotion in self.emotion_assemblies:
            assembly = self.emotion_assemblies[emotion]
            for neuron in assembly['neurons']:
                # Slightly lower the threshold for these neurons to make them more likely to fire
                # This is a temporary effect for this simulation only
                if hasattr(self, 'neuron_biases'):
                    self.neuron_biases[neuron] = -0.1  # Negative bias lowers threshold
                else:
                    # Initialize if not exists
                    self.neuron_biases = np.zeros(self.neuron_count)
                    self.neuron_biases[neuron] = -0.1
        
        # 2. Bias valence region based on predicted valence
        if 'valence_processing' in self.regions:
            valence_neurons = self.regions['valence_processing']['neurons']
            valence_position = (valence + 1) / 2  # Map from [-1,1] to [0,1]
            
            # Apply bias to neurons in the region based on position
            for i, neuron in enumerate(sorted(valence_neurons)):
                rel_pos = i / max(1, len(valence_neurons) - 1)
                # Stronger bias for neurons closer to the predicted valence position
                bias_strength = -0.1 * max(0, 1 - abs(rel_pos - valence_position) * 10)
                
                if hasattr(self, 'neuron_biases'):
                    self.neuron_biases[neuron] = bias_strength
                else:
                    self.neuron_biases = np.zeros(self.neuron_count)
                    self.neuron_biases[neuron] = bias_strength
        
        # 3. Bias arousal region based on predicted arousal
        if 'arousal_processing' in self.regions:
            arousal_neurons = self.regions['arousal_processing']['neurons']
            
            # Apply bias to neurons in the region based on position
            for i, neuron in enumerate(sorted(arousal_neurons)):
                rel_pos = 1 - (i / max(1, len(arousal_neurons) - 1))  # Invert mapping
                # Stronger bias for neurons closer to the predicted arousal position
                bias_strength = -0.1 * max(0, 1 - abs(rel_pos - arousal) * 10)
                
                if hasattr(self, 'neuron_biases'):
                    self.neuron_biases[neuron] = bias_strength
                else:
                    self.neuron_biases = np.zeros(self.neuron_count)
                    self.neuron_biases[neuron] = bias_strength
        
    def _integrate_emotion_weights(self):
        """
        Integrate learned emotion assembly weights into the main synaptic weight matrix.
        This ensures consistent weight updates across the network.
        """
        updates_applied = 0
        
        # Process each emotion assembly
        for emotion, assembly in self.emotion_assemblies.items():
            # Skip if no weights defined
            if 'weights' not in assembly or not assembly['weights']:
                continue
                
            # Get assembly neurons and their weights
            for neuron, input_weights in assembly['weights'].items():
                # Add neuron to custom learning set to prevent conflicts
                self.custom_learning_neurons.add(neuron)
                
                # Update connections in the main weight matrix
                for input_neuron, weight in input_weights.items():
                    # Only update weights above threshold to maintain sparsity
                    if abs(weight) > 0.05:
                        # Set this connection in the main weight matrix
                        self.synaptic_weights[input_neuron, neuron] = weight
                        updates_applied += 1
        
        return updates_applied
    
    def evaluate_emotional_understanding(self, test_data):
        """
        Evaluate the network's emotional understanding capabilities.
        
        Args:
            test_data: List of (text, emotion_label, valence, arousal) tuples
            
        Returns:
            Evaluation metrics
        """
        self.training_mode = False  # Ensure we're in inference mode
        
        metrics = {
            "emotion_accuracy": 0,
            "valence_mse": 0,
            "arousal_mse": 0,
            "confusion_matrix": defaultdict(lambda: defaultdict(int))
        }
        
        for text, true_emotion, true_valence, true_arousal in test_data:
            # Process through network
            result = self.process_input(text)
            
            # Get predicted emotion (highest activated assembly)
            predicted_emotion = self.identify_emotion(text)['dominant_emotion']
            
            # Get predicted valence/arousal
            va_result = self.decode_valence_arousal()
            pred_valence = va_result['valence']
            pred_arousal = va_result['arousal']
            
            # Update confusion matrix
            metrics["confusion_matrix"][true_emotion][predicted_emotion] += 1
            
            # Calculate accuracy (binary for emotion)
            if predicted_emotion == true_emotion:
                metrics["emotion_accuracy"] += 1
            
            # Calculate MSE for valence/arousal
            metrics["valence_mse"] += (true_valence - pred_valence) ** 2
            metrics["arousal_mse"] += (true_arousal - pred_arousal) ** 2
        
        # Normalize metrics
        total_samples = len(test_data)
        metrics["emotion_accuracy"] /= total_samples
        metrics["valence_mse"] /= total_samples
        metrics["arousal_mse"] /= total_samples
        
        return metrics
    
    def learn_from_experience(self, data_batch, epochs=1):
        """
        Learn from a batch of emotional text experiences.
        
        Args:
            data_batch: List of (text, emotion_label, valence, arousal) tuples
            epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        if not data_batch:
            return {"error": "No training data provided"}
        
        metrics = {
            "epochs": epochs,
            "samples": len(data_batch),
            "emotion_loss": [],
            "va_loss": [],
            "assembly_activations": defaultdict(list),
            "learning_curve": []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_emotion_loss = 0
            epoch_va_loss = 0
            
            # Process each training sample
            for text, emotion, valence, arousal in data_batch:
                # Train emotion assembly
                train_result = self.train_emotion_assembly(text, emotion)
                epoch_emotion_loss += train_result['emotion_loss']
                epoch_va_loss += train_result['va_loss']
                
                # Track assembly activations
                for emotion_name, activation in train_result['assembly_activations'].items():
                    metrics["assembly_activations"][emotion_name].append(activation)
                
                # Update training iteration counter
                self.training_iterations += 1
            
            # Calculate average losses for this epoch
            avg_emotion_loss = epoch_emotion_loss / len(data_batch)
            avg_va_loss = epoch_va_loss / len(data_batch)
            
            # Store metrics
            metrics["emotion_loss"].append(avg_emotion_loss)
            metrics["va_loss"].append(avg_va_loss)
            metrics["learning_curve"].append(1.0 - (avg_emotion_loss + avg_va_loss) / 2)
            
            print(f"Epoch {epoch+1}/{epochs}: Emotion Loss: {avg_emotion_loss:.4f}, VA Loss: {avg_va_loss:.4f}")
        
        # Evaluate on the data set to measure progress
        evaluation = self.evaluate_emotional_understanding(data_batch)
        metrics.update(evaluation)
        
        return metrics
    
    def influence_processing(self, target_component):
        """
        Send affective signals to influence processing in target component.
        Converts neural affective representation into an influence signal.
        
        Args:
            target_component: Component to influence with affective state
            
        Returns:
            Success status
        """
        # Get current affective state
        if not hasattr(self, 'current_affective_state') or not self.current_affective_state:
            # No current state - perform evaluation with neutral input
            neutral_features = {
                'sentiment': 0.0,
                'intensity': 0.1,
                'emotional_keywords': [],
                'detected_emotions': {}
            }
            self.current_affective_state = self.evaluate_affective_state(neutral_features)
        
        # Create influence signal based on current affective state
        influence_signal = {
            "type": "affective_influence",
            "valence": self.current_affective_state["valence"],
            "arousal": self.current_affective_state["arousal"],
            "dominant_emotion": self.current_affective_state["dominant_emotion"],
            "intensity": self.current_affective_state["intensity"],
            "learning_modulation": self._calculate_learning_modulation(),
            "emotion_activations": self.current_affective_state["emotions"].copy()
        }
        
        # Send influence signal to target component
        if hasattr(target_component, 'receive_affective_influence'):
            return target_component.receive_affective_influence(influence_signal)
        elif hasattr(target_component, 'modulate_learning_rate') and "learning_modulation" in influence_signal:
            # Apply learning rate modulation if available
            return target_component.modulate_learning_rate(influence_signal["learning_modulation"])
        elif hasattr(target_component, 'receive_modulation'):
            # Generic modulation interface
            return target_component.receive_modulation(influence_signal)
        
        return False
    
    def _calculate_learning_modulation(self):
        """Calculate how current emotional state should modulate learning"""
        # High arousal typically enhances learning
        arousal_factor = max(0.5, min(2.0, 1.0 + self.current_affective_state["arousal"]))
        
        # Extreme valence (both positive and negative) enhances memory formation
        valence_factor = max(0.5, min(2.0, 1.0 + abs(self.current_affective_state["valence"]) * 0.5))
        
        # Emotion-specific modulation
        emotion_factor = 1.0
        emotions = self.current_affective_state["emotions"]
        
        # Different emotions have different effects on learning
        if "fear" in emotions and emotions["fear"] > 0.5:
            # Fear increases learning for threat-relevant information
            emotion_factor = 1.5
        elif "joy" in emotions and emotions["joy"] > 0.6:
            # Joy enhances exploratory learning
            emotion_factor = 1.3
        elif "surprise" in emotions and emotions["surprise"] > 0.7:
            # Surprise enhances attention and learning
            emotion_factor = 1.4
        
        # Combine factors with different weights
        return 0.4 * arousal_factor + 0.3 * valence_factor + 0.3 * emotion_factor
    
    def receive_affective_influence(self, influence_signal):
        """
        Receive influence signal from another affective component.
        This allows for emotional contagion and modulation.
        
        Args:
            influence_signal: Influence signal dictionary
            
        Returns:
            Processing result
        """
        # Extract influence components
        valence = influence_signal.get("valence", 0.0)
        arousal = influence_signal.get("arousal", 0.0)
        emotion_activations = influence_signal.get("emotion_activations", {})
        
        # Generate input features from influence signal
        input_features = {
            'sentiment': valence,
            'intensity': arousal,
            'emotional_keywords': [],
            'detected_emotions': emotion_activations
        }
        
        # Process the influence signal
        result = self.evaluate_affective_state(input_features)
        
        # Modulate network state based on influence
        intensity = influence_signal.get("intensity", 0.5)
        self._modulate_network_state(intensity)
        
        return {
            "success": True,
            "influence_effect": result,
            "modulation_strength": intensity
        }
        
    def _modulate_network_state(self, intensity):
        """
        Modulate network activation state based on emotional intensity.
        Higher intensity heightens neural sensitivity.
        
        Args:
            intensity: Activation intensity (0.0 to 1.0)
        """
        # Scale spike threshold based on emotional intensity
        # Higher intensity (arousal) lowers threshold, making neurons more excitable
        baseline_threshold = 0.5  # Default threshold
        modulation_range = 0.3    # Max adjustment
        
        # Adjust threshold (higher intensity = lower threshold)
        new_threshold = baseline_threshold - (intensity * modulation_range)
        self.spike_threshold = max(0.2, min(0.7, new_threshold))
        
        # Also modulate decay rate (higher intensity = slower decay)
        baseline_decay = 0.9
        decay_modulation = 0.1
        
        # Adjust decay (higher intensity = higher decay rate = less persistence)
        new_decay = baseline_decay + (intensity * decay_modulation)
        self.decay_rate = max(0.85, min(0.98, new_decay))
    
    def get_emotion_space_mapping(self):
        """
        Get a mapping of emotions in valence-arousal space.
        Useful for visualization and analysis.
        
        Returns:
            Dictionary mapping emotions to locations in valence-arousal space
        """
        # Calculate learned locations based on neural representations
        # and mapping to valence-arousal space
        emotion_mapping = {}
        
        for emotion, assembly in self.emotion_assemblies.items():
            # Use predefined mapping if available
            if emotion in self.valence_arousal_mapping:
                base_valence, base_arousal = self.valence_arousal_mapping[emotion]
                
                # Adjust based on confidence and learning
                confidence = assembly.get('confidence', 0.5)
                
                # More confident mappings stay closer to original position
                confidence_factor = 0.5 + (confidence * 0.5)
                
                # Add small learned variation (more variation with less confidence)
                learned_variation = (1 - confidence) * 0.2
                valence_adjust = random.uniform(-learned_variation, learned_variation)
                arousal_adjust = random.uniform(-learned_variation, learned_variation)
                
                # Calculate final position
                valence = base_valence * confidence_factor + valence_adjust
                arousal = base_arousal * confidence_factor + arousal_adjust
                
                # Store mapping
                emotion_mapping[emotion] = {
                    'valence': valence,
                    'arousal': arousal,
                    'confidence': confidence,
                    'neurons': len(assembly['neurons'])
                }
        
        return emotion_mapping
        
    def transfer_knowledge_to_supervised(self):
        """
        Transfer knowledge from the bio-inspired network to the supervised networks.
        This allows the supervised networks to learn from the emergent representations
        in the bio-inspired network.
        
        Returns:
            Results of the knowledge transfer
        """
        # Get all emotions
        emotions = list(self.valence_arousal_mapping.keys())
        
        # No transfer if no emotions or no supervised networks
        if not emotions or not hasattr(self, 'emotion_network') or not hasattr(self, 'va_network'):
            return {
                "success": False,
                "error": "No emotions or supervised networks available for transfer"
            }
        
        # Track learning metrics
        transfer_metrics = {
            "samples_processed": 0,
            "emotion_loss_before": 0,
            "emotion_loss_after": 0,
            "va_loss_before": 0,
            "va_loss_after": 0
        }
        
        # Create a training dataset from bio-inspired representations
        training_samples = []
        
        # For each emotion, generate features and targets
        for emotion in emotions:
            if emotion not in self.emotion_assemblies:
                continue
                
            # Get the valence-arousal mapping
            if emotion not in self.valence_arousal_mapping:
                continue
                
            valence, arousal = self.valence_arousal_mapping[emotion]
            
            # Create simple input features
            input_features = {
                'sentiment': valence,
                'intensity': arousal,
                'emotional_keywords': [emotion],
                'detected_emotions': {emotion: 0.9}
            }
            
            # Process through SNN
            input_activation = self.encode_affective_input(input_features)
            result = super().process_input(input_activation)
            
            # Extract bio-inspired features
            bio_features = self._extract_bio_inspired_features(result)
            
            # Create targets
            emotion_index = emotions.index(emotion)
            emotion_target = np.zeros(len(emotions))
            emotion_target[emotion_index] = 1.0
            
            va_target = np.array([valence, arousal])
            
            # Add to training samples
            training_samples.append({
                'features': bio_features,
                'emotion_target': emotion_target,
                'va_target': va_target,
                'emotion': emotion
            })
        
        # Measure loss before training
        for sample in training_samples:
            # Convert to tensors
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            emotion_target = torch.tensor(sample['emotion_target'], dtype=torch.float32).unsqueeze(0)
            va_target = torch.tensor(sample['va_target'], dtype=torch.float32).unsqueeze(0)
            
            # Compute loss
            with torch.no_grad():
                emotion_output = self.emotion_network(features)
                emotion_loss = self.emotion_loss_fn(emotion_output, torch.argmax(emotion_target, dim=1))
                transfer_metrics['emotion_loss_before'] += emotion_loss.item()
                
                va_output = self.va_network(features)
                va_loss = self.va_loss_fn(va_output, va_target)
                transfer_metrics['va_loss_before'] += va_loss.item()
        
        # Train on the samples
        num_epochs = 5  # Multiple passes for better learning
        
        for epoch in range(num_epochs):
            # Shuffle samples
            random.shuffle(training_samples)
            
            for sample in training_samples:
                # Convert to tensors
                features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
                emotion_target = torch.tensor(sample['emotion_target'], dtype=torch.float32).unsqueeze(0)
                va_target = torch.tensor(sample['va_target'], dtype=torch.float32).unsqueeze(0)
                
                # Train emotion network
                self.emotion_optimizer.zero_grad()
                emotion_output = self.emotion_network(features)
                emotion_loss = self.emotion_loss_fn(emotion_output, torch.argmax(emotion_target, dim=1))
                emotion_loss.backward()
                self.emotion_optimizer.step()
                
                # Train VA network
                self.va_optimizer.zero_grad()
                va_output = self.va_network(features)
                va_loss = self.va_loss_fn(va_output, va_target)
                va_loss.backward()
                self.va_optimizer.step()
                
                transfer_metrics['samples_processed'] += 1
        
        # Measure loss after training
        for sample in training_samples:
            # Convert to tensors
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            emotion_target = torch.tensor(sample['emotion_target'], dtype=torch.float32).unsqueeze(0)
            va_target = torch.tensor(sample['va_target'], dtype=torch.float32).unsqueeze(0)
            
            # Compute loss
            with torch.no_grad():
                emotion_output = self.emotion_network(features)
                emotion_loss = self.emotion_loss_fn(emotion_output, torch.argmax(emotion_target, dim=1))
                transfer_metrics['emotion_loss_after'] += emotion_loss.item()
                
                va_output = self.va_network(features)
                va_loss = self.va_loss_fn(va_output, va_target)
                transfer_metrics['va_loss_after'] += va_loss.item()
        
        # Calculate improvements
        num_samples = len(training_samples)
        if num_samples > 0:
            transfer_metrics['emotion_loss_before'] /= num_samples
            transfer_metrics['emotion_loss_after'] /= num_samples
            transfer_metrics['va_loss_before'] /= num_samples
            transfer_metrics['va_loss_after'] /= num_samples
            
            transfer_metrics['emotion_improvement'] = transfer_metrics['emotion_loss_before'] - transfer_metrics['emotion_loss_after']
            transfer_metrics['va_improvement'] = transfer_metrics['va_loss_before'] - transfer_metrics['va_loss_after']
        
        return {
            "success": True,
            "metrics": transfer_metrics,
            "samples_transferred": len(training_samples)
        }

    def transfer_supervised_to_bio(self):
        """
        Transfer knowledge from supervised networks to bio-inspired network.
        This allows the bio-inspired network to benefit from the structured learning
        in the supervised networks using a direct error-driven learning approach.
        
        Returns:
            Results of the knowledge transfer
        """
        # Need supervised networks for this transfer
        if not hasattr(self, 'emotion_network') or not hasattr(self, 'va_network'):
            return {
                "success": False,
                "error": "No supervised networks available for transfer"
            }
        
        # Get all emotions
        emotions = list(self.valence_arousal_mapping.keys())
        
        # Track learning metrics
        transfer_metrics = {
            "weights_updated": 0,
            "samples_processed": 0,
            "emotions_processed": []
        }
        
        # For each emotion, use supervised predictions to guide bio-inspired learning
        for emotion in emotions:
            if emotion not in self.emotion_assemblies:
                continue
                
            # Get the valence-arousal mapping
            if emotion not in self.valence_arousal_mapping:
                continue
                
            valence, arousal = self.valence_arousal_mapping[emotion]
            
            # Create input features with some variation
            for _ in range(3):  # Multiple samples per emotion for robustness
                # Add variations to valence/arousal for better generalization
                var_valence = valence + random.uniform(-0.1, 0.1)
                var_arousal = arousal + random.uniform(-0.1, 0.1)
                
                # Keep within valid ranges
                var_valence = max(-1.0, min(1.0, var_valence))
                var_arousal = max(0.0, min(1.0, var_arousal))
                
                input_features = {
                    'sentiment': var_valence,
                    'intensity': var_arousal,
                    'emotional_keywords': [emotion],
                    'detected_emotions': {emotion: 0.9}
                }
                
                # Encode input
                input_activation = self.encode_affective_input(input_features)
                
                # Process through bio-inspired SNN
                result = super().process_input(input_activation)
                
                # Extract features for supervised networks
                bio_features = self._extract_bio_inspired_features(result)
                input_tensor = torch.tensor(bio_features, dtype=torch.float32).unsqueeze(0)
                
                # Get supervised predictions
                with torch.no_grad():
                    emotion_output = self.emotion_network(input_tensor)
                    emotion_probs = torch.softmax(emotion_output, dim=1)[0]
                    
                    va_output = self.va_network(input_tensor)[0]
                    predicted_valence = va_output[0].item()
                    predicted_arousal = va_output[1].item()
                
                # Get target emotion index 
                emotions_list = list(self.valence_arousal_mapping.keys())
                target_emotion_idx = emotions_list.index(emotion)
                predicted_emotion_idx = torch.argmax(emotion_probs).item()
                
                # Calculate error for correct emotion activation
                target_confidence = emotion_probs[target_emotion_idx].item()
                error = 1.0 - target_confidence  # Error is how far we are from perfect confidence
                
                # DIRECT ERROR-DRIVEN LEARNING APPROACH
                # Instead of using reinforcement learning, we directly update weights based on error
                
                # 1. Update emotion assembly weights directly
                learn_rate = 0.03
                assembly = self.emotion_assemblies[emotion]
                target_activation = 0.9  # High target for correct emotion
                
                # Get actual activation
                actual_activation = self._get_assembly_activation(emotion, result.get('spike_patterns', []))
                
                # Calculate activation error
                activation_error = target_activation - actual_activation
                
                # Update weights using our direct error-driven approach
                weights_updated = self._update_assembly_weights(
                    emotion, 
                    input_activation, 
                    activation_error, 
                    learn_rate
                )
                
                # 2. Also update valence and arousal regions to match expected values
                va_weights_updated = self._train_valence_arousal_regions(
                    input_activation,
                    var_valence,  # Target is the original valence
                    var_arousal,  # Target is the original arousal
                    learn_rate=0.03
                )
                
                transfer_metrics['weights_updated'] += weights_updated + va_weights_updated
                transfer_metrics['samples_processed'] += 1
            
            transfer_metrics['emotions_processed'].append(emotion)
        
        return {
            "success": True,
            "metrics": transfer_metrics
        }
    
    def process_text_input(self, text_input, timesteps=20):
        """
        Process text input using the standardized bidirectional processor.
        
        Args:
            text_input: Text input for emotional processing
            timesteps: Number of timesteps for spike patterns
            
        Returns:
            Processed spike patterns for emotional processing
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
                
                # Apply emotion-specific processing if needed
                # For now, we'll just use the standard spike patterns
                self.spike_patterns = spike_patterns
                return spike_patterns
            except Exception as e:
                print(f"[AffectiveSNN] Error processing text with bidirectional processor: {e}")
                # Fall back to legacy encoding
        
        # Legacy encoding if bidirectional processor is not available or error occurs
        # Convert text to emotional features
        affective_features = self._extract_emotional_features(text_input)
        # Encode emotional features to activation pattern
        activation = self.encode_affective_input(affective_features)
        # Simulate spiking to get patterns
        spike_patterns = self.simulate_spiking(activation, timesteps=timesteps)
        
        return spike_patterns
    
    def generate_text_output(self, spike_patterns, max_length=100):
        """
        Generate text output from emotional spike patterns using the standardized bidirectional processor.
        
        Args:
            spike_patterns: Spike patterns from emotional processing
            max_length: Maximum length of generated text
            
        Returns:
            Generated text with emotional content
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
                print(f"[AffectiveSNN] Error generating text with bidirectional processor: {e}")
                # Fall back to legacy approach
        
        # Legacy approach for text generation
        # Decode emotional state from spike patterns
        affective_state = self._hybrid_decode_affective_state(
            spike_patterns, 
            torch.zeros(len(self.valence_arousal_mapping)),  # Default probs
            0.0,  # Default valence
            0.0   # Default arousal
        )
        
        # Generate simple text based on emotional state
        if affective_state:
            dominant_emotion = affective_state.get('dominant_emotion', '')
            valence = affective_state.get('valence', 0.0)
            arousal = affective_state.get('arousal', 0.0)
            
            # Format response based on emotional state
            if dominant_emotion:
                return f"Detected emotional content: {dominant_emotion} (valence: {valence:.2f}, arousal: {arousal:.2f})"
            else:
                if valence > 0.3:
                    return "Detected positive emotional content"
                elif valence < -0.3:
                    return "Detected negative emotional content"
                else:
                    return "Detected neutral emotional content"
        
        return "No clear emotional content detected"
    
    def _extract_emotional_features(self, text_input):
        """
        Legacy method to extract emotional features from text (fallback when bidirectional processor not available).
        
        Args:
            text_input: Input text to analyze
            
        Returns:
            Dictionary of affective features
        """
        # Very simple feature extraction (just for fallback)
        features = {
            'sentiment': 0.0,
            'intensity': 0.5,
            'valence': 0.0,
            'arousal': 0.5
        }
        
        # Basic word-based sentiment analysis
        positive_words = {'happy', 'joy', 'excited', 'good', 'excellent', 'wonderful', 'love', 'like'}
        negative_words = {'sad', 'angry', 'fear', 'worried', 'bad', 'terrible', 'hate', 'dislike'}
        intensity_words = {'very', 'extremely', 'incredibly', 'totally', 'absolutely', 'completely'}
        
        words = text_input.lower().split()
        
        # Count emotional words
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        intensity_count = sum(1 for word in words if word in intensity_words)
        
        # Calculate sentiment score (-1 to 1)
        if pos_count > 0 or neg_count > 0:
            features['sentiment'] = (pos_count - neg_count) / max(pos_count + neg_count, 1)
            features['valence'] = features['sentiment']
        
        # Calculate intensity (0 to 1)
        if intensity_count > 0:
            features['intensity'] = min(0.5 + (0.1 * intensity_count), 1.0)
            features['arousal'] = features['intensity']
        
        return features