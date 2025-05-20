# reasoning_snn.py
import numpy as np
import re
import random
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

class ReasoningSNN(EnhancedSpikingCore):
    """SNN specialized for reasoning and inference with hybrid learning"""
    
    def __init__(self, neuron_count=800, topology_type="scale_free", vector_dim=300, bidirectional_processor=None):
        """
        Initialize the reasoning SNN with specialized parameters.
        
        Args:
            neuron_count: Total number of neurons in the network
            topology_type: Type of network topology
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
            model_type="reasoning",
            vector_dim=vector_dim,
            bidirectional_processor=bidirectional_processor
        )
        
        # Initialize neuron biases
        self.neuron_biases = np.zeros(neuron_count)
        
        # Override with reasoning-optimized parameters
        self.connection_density = 0.18    # Higher density for complex reasoning
        self.spike_threshold = 0.45       # Lower threshold for easier activation propagation
        
        # Reasoning-specific attributes
        self.reasoning_modes = ["deductive", "inductive", "abductive"]
        self.current_mode = "abductive"   # Default reasoning mode
        self.confidence_thresholds = {
            "deductive": 0.8,             # High confidence for logical deduction
            "inductive": 0.6,             # Medium confidence for generalizations
            "abductive": 0.5              # Lower confidence for explanatory hypotheses
        }
        self.domain_knowledge = self._init_domain_knowledge()
        
        # Initialize reasoning circuits
        self.reasoning_circuits = {}
        self._init_reasoning_circuits()
        
        # Set up SNNTorch components for supervised learning
        self._setup_snntorch_components()
        
        # Register specialized reasoning synapses for hybrid learning
        self._register_reasoning_synapses()
    
    def _setup_snntorch_components(self):
        """Set up SNNTorch components for supervised learning"""
        try:
            self.torch_available = True if torch else False
            
            if not self.torch_available:
                print("[ReasoningSNN] Warning: PyTorch not available - supervised learning disabled")
                return
            
            # Beta is the decay rate for the neuron's spiking trace
            beta = 0.95
            
            # Use surrogate gradient for backpropagation through spikes
            spike_grad = surrogate.fast_sigmoid(slope=25)
            
            # Get size of reasoning features
            reasoning_features_size = 150  # Larger for complex reasoning
            
            # 1. Deductive validity predictor
            self.deductive_network = nn.Sequential(
                nn.Linear(reasoning_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output validity probability [0,1]
            )
            
            # 2. Inductive pattern predictor
            self.inductive_network = nn.Sequential(
                nn.Linear(reasoning_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, 32)  # Output pattern encoding
            )
            
            # 3. Abductive hypothesis scorer
            self.abductive_network = nn.Sequential(
                nn.Linear(reasoning_features_size, 128),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(128, 64),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output hypothesis quality score [0,1]
            )
            
            # Define loss functions
            self.deductive_loss_fn = nn.BCELoss()      # Binary validity
            self.inductive_loss_fn = nn.MSELoss()      # Pattern regression
            self.abductive_loss_fn = nn.MSELoss()      # Hypothesis scoring
            
            # Define optimizers
            self.deductive_optimizer = optim.Adam(self.deductive_network.parameters(), lr=0.01)
            self.inductive_optimizer = optim.Adam(self.inductive_network.parameters(), lr=0.01)
            self.abductive_optimizer = optim.Adam(self.abductive_network.parameters(), lr=0.01)
            
            print("[ReasoningSNN] SNNTorch components initialized successfully")
            
        except Exception as e:
            print(f"[ReasoningSNN] Warning: Error initializing SNNTorch components - {e}")
            self.torch_available = False
    
    def _register_reasoning_synapses(self):
        """
        Register specialized synapses for reasoning operations.
        This ensures the parent's plasticity won't interfere with our custom learning.
        """
        synapse_count = 0
        
        # 1. Register synapses within reasoning regions
        reasoning_regions = ['higher_cognition', 'metacognition', 'decision']
        for region_name in reasoning_regions:
            if region_name in self.regions:
                neurons = self.regions[region_name]['neurons']
                
                # Register within-region connections
                num_synapses = min(len(neurons) * len(neurons) // 3, 300)
                for _ in range(num_synapses):
                    if len(neurons) >= 2:
                        pre, post = random.sample(neurons, 2)
                        self.register_specialized_synapse(pre, post)
                        synapse_count += 1
        
        # 2. Register synapses within reasoning circuits
        for circuit_name, circuit_neurons in self.reasoning_circuits.items():
            if len(circuit_neurons) >= 2:
                num_synapses = min(len(circuit_neurons) * len(circuit_neurons) // 2, 100)
                for _ in range(num_synapses):
                    pre, post = random.sample(circuit_neurons, 2)
                    self.register_specialized_synapse(pre, post)
                    synapse_count += 1
        
        # 3. Register connections between reasoning and memory
        if 'memory' in self.regions and 'higher_cognition' in self.regions:
            memory_neurons = self.regions['memory']['neurons']
            cognition_neurons = self.regions['higher_cognition']['neurons']
            
            num_synapses = min(len(memory_neurons) * len(cognition_neurons) // 20, 200)
            for _ in range(num_synapses):
                pre = random.choice(memory_neurons)
                post = random.choice(cognition_neurons)
                self.register_specialized_synapse(pre, post)
                self.register_specialized_synapse(post, pre)  # Bidirectional
                synapse_count += 2
        
        # 4. Register symbolic layer connections if available
        if hasattr(self, 'symbolic_layer') and 'higher_cognition' in self.regions:
            symbolic_neurons = self.symbolic_layer.get('neurons', [])
            cognition_neurons = self.regions['higher_cognition']['neurons']
            
            if symbolic_neurons:
                num_synapses = min(len(symbolic_neurons) * len(cognition_neurons) // 10, 100)
                for _ in range(num_synapses):
                    pre = random.choice(symbolic_neurons)
                    post = random.choice(cognition_neurons)
                    self.register_specialized_synapse(pre, post)
                    synapse_count += 1
        
        print(f"[ReasoningSNN] Registered {synapse_count} synapses for specialized reasoning learning")
        return synapse_count
    
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
    
    def _extract_reasoning_features_for_snntorch(self, network_state, reasoning_mode):
        """
        Extract reasoning-relevant features from the SNN state for supervised learning.
        
        Args:
            network_state: Current network state after processing input
            reasoning_mode: Current reasoning mode (deductive/inductive/abductive)
            
        Returns:
            Feature vector for SNNTorch networks (150-dimensional)
        """
        features = []
        
        # 1. Region activations relevant to reasoning
        reasoning_regions = ['higher_cognition', 'metacognition', 'decision', 'memory']
        for region in reasoning_regions:
            if region in self.regions:
                activation = self.regions[region]['activation']
                features.append(activation)
            else:
                features.append(0.0)
        
        # 2. Circuit-specific activations
        for mode in self.reasoning_modes:
            if mode in self.reasoning_circuits:
                neurons = self.reasoning_circuits[mode]
                active_neurons = network_state.get('active_neurons', set())
                
                # Calculate activation ratio for this circuit
                active_count = len(set(neurons).intersection(active_neurons))
                activation_ratio = active_count / len(neurons) if neurons else 0
                features.append(activation_ratio)
                
                # Mode-specific activation strength
                if mode == reasoning_mode:
                    features.append(activation_ratio * 1.5)  # Boost current mode
                else:
                    features.append(activation_ratio)
            else:
                features.extend([0.0, 0.0])
        
        # 3. Integration metrics (crucial for complex reasoning)
        features.append(network_state.get('phi', 0.0))
        features.append(network_state.get('integration', 0.0))
        features.append(network_state.get('differentiation', 0.0))
        
        # 4. Cross-region coherence (consistency of reasoning)
        # Check correlation between related regions
        if 'higher_cognition' in self.regions and 'metacognition' in self.regions:
            cog_activation = self.regions['higher_cognition']['activation']
            meta_activation = self.regions['metacognition']['activation']
            coherence = cog_activation * meta_activation
            features.append(coherence)
        else:
            features.append(0.0)
        
        # 5. Pattern stability (for logical consistency)
        stability = self._calculate_pattern_stability(network_state)
        features.append(stability)
        
        # 6. Concept activation patterns
        active_concepts = network_state.get('active_concepts', {})
        concept_features = []
        key_concepts = ['cause', 'effect', 'if', 'then', 'all', 'some', 'none']
        for concept in key_concepts:
            activation = active_concepts.get(concept, 0.0)
            concept_features.append(activation)
        features.extend(concept_features)
        
        # 7. Mode-specific features
        if reasoning_mode == "deductive":
            # Logical structure detection
            premise_activation = self._detect_premise_activation(network_state)
            conclusion_activation = self._detect_conclusion_activation(network_state)
            logical_flow = premise_activation * conclusion_activation
            features.extend([premise_activation, conclusion_activation, logical_flow])
        elif reasoning_mode == "inductive":
            # Pattern generalization metrics
            pattern_diversity = self._calculate_pattern_diversity(network_state)
            generalization_strength = self._calculate_generalization_strength(network_state)
            features.extend([pattern_diversity, generalization_strength, pattern_diversity * generalization_strength])
        elif reasoning_mode == "abductive":
            # Hypothesis quality metrics
            hypothesis_coherence = self._calculate_hypothesis_coherence(network_state)
            explanatory_power = self._calculate_explanatory_power(network_state)
            features.extend([hypothesis_coherence, explanatory_power, hypothesis_coherence * explanatory_power])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to reach target dimension
        current_dim = len(features)
        target_dim = 150
        
        if current_dim < target_dim:
            features.extend([0.0] * (target_dim - current_dim))
        elif current_dim > target_dim:
            features = features[:target_dim]
        
        return np.array(features)
    
    def train_reasoning(self, input_activation, target_output, reasoning_mode, learn_rate=0.02):
        """
        Train reasoning using hybrid learning approach.
        
        Args:
            input_activation: Input activation pattern (reasoning problem)
            target_output: Target output (depends on reasoning mode)
            reasoning_mode: Type of reasoning (deductive/inductive/abductive)
            learn_rate: Learning rate for weight updates
            
        Returns:
            Training results
        """
        # Set training mode and control parent plasticity
        self.training_mode = True
        self.disable_parent_plasticity = True
        
        # Set current reasoning mode
        self.set_reasoning_mode(reasoning_mode)
        
        # Process input through bio-inspired network
        network_result = self.process_input(input_activation)
        
        # Extract features for supervised networks
        features = self._extract_reasoning_features_for_snntorch(network_result, reasoning_mode)
        
        # PHASE 1: Supervised learning with SNNTorch
        if self.torch_available:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            if reasoning_mode == "deductive":
                # Deductive: predict validity (binary)
                target_tensor = torch.tensor([[float(target_output)]], dtype=torch.float32)
                
                self.deductive_optimizer.zero_grad()
                output = self.deductive_network(features_tensor)
                loss = self.deductive_loss_fn(output, target_tensor)
                loss.backward()
                self.deductive_optimizer.step()
                
                predicted = output.item()
                error = target_output - predicted
                
            elif reasoning_mode == "inductive":
                # Inductive: predict pattern (vector)
                target_tensor = torch.tensor([target_output], dtype=torch.float32)
                
                self.inductive_optimizer.zero_grad()
                output = self.inductive_network(features_tensor)
                loss = self.inductive_loss_fn(output, target_tensor)
                loss.backward()
                self.inductive_optimizer.step()
                
                predicted = output.detach().numpy()[0]
                error = np.mean(target_output - predicted)
                
            elif reasoning_mode == "abductive":
                # Abductive: predict hypothesis quality (scalar)
                target_tensor = torch.tensor([[float(target_output)]], dtype=torch.float32)
                
                self.abductive_optimizer.zero_grad()
                output = self.abductive_network(features_tensor)
                loss = self.abductive_loss_fn(output, target_tensor)
                loss.backward()
                self.abductive_optimizer.step()
                
                predicted = output.item()
                error = target_output - predicted
        else:
            # No supervised learning available
            loss = 0.0
            predicted = 0.5
            error = 0.0
        
        # PHASE 2: Create supervised biases to guide bio-inspired activity
        self.neuron_biases = self._create_supervised_biases(features_tensor, reasoning_mode)
        
        # PHASE 3: Re-process with biases to get guided network state
        guided_result = self.process_input(input_activation)
        
        # PHASE 4: Apply custom error-driven plasticity
        weights_updated = self._update_reasoning_weights(
            input_activation,
            error,
            reasoning_mode,
            guided_result,
            learn_rate
        )
        
        # Reset flags
        self.disable_parent_plasticity = False
        self.training_mode = False
        
        return {
            'loss': float(loss) if isinstance(loss, (int, float)) else loss.item(),
            'predicted': predicted,
            'target': target_output,
            'error': error,
            'weights_updated': weights_updated,
            'reasoning_mode': reasoning_mode
        }
    
    def set_reasoning_mode(self, mode):
        """Set the current reasoning mode"""
        if mode in self.reasoning_modes:
            self.current_mode = mode
            print(f"[ReasoningSNN] Reasoning mode set to: {mode}")
        else:
            print(f"[ReasoningSNN] Warning: Unknown reasoning mode '{mode}'")

    def _create_supervised_biases(self, features_tensor, reasoning_mode):
        """
        Create neuron biases based on supervised network predictions.
        
        Args:
            features_tensor: Features for supervised networks
            reasoning_mode: Current reasoning mode
            
        Returns:
            Bias values for each neuron
        """
        if not self.torch_available:
            return np.zeros(self.neuron_count)
        
        biases = np.zeros(self.neuron_count)
        
        # Get predictions from appropriate network
        with torch.no_grad():
            if reasoning_mode == "deductive":
                validity = self.deductive_network(features_tensor).item()
                # Bias deductive circuit neurons
                if 'deductive' in self.reasoning_circuits:
                    for neuron in self.reasoning_circuits['deductive']:
                        biases[neuron] = validity * 0.3
            
            elif reasoning_mode == "inductive":
                pattern = self.inductive_network(features_tensor).detach().numpy()[0]
                # Bias inductive circuit neurons based on pattern
                if 'inductive' in self.reasoning_circuits:
                    for i, neuron in enumerate(self.reasoning_circuits['inductive']):
                        if i < len(pattern):
                            biases[neuron] = pattern[i] * 0.3
            
            elif reasoning_mode == "abductive":
                hypothesis_score = self.abductive_network(features_tensor).item()
                # Bias abductive circuit neurons
                if 'abductive' in self.reasoning_circuits:
                    for neuron in self.reasoning_circuits['abductive']:
                        biases[neuron] = hypothesis_score * 0.3
        
        return biases
    
    def _update_reasoning_weights(self, input_activation, error, reasoning_mode, network_state, learn_rate):
        """
        Update synaptic weights based on reasoning error using custom plasticity rules.
        
        Args:
            input_activation: Input pattern
            error: Error signal from supervised learning
            reasoning_mode: Current reasoning mode
            network_state: Network state after guided processing
            learn_rate: Learning rate
            
        Returns:
            Number of weights updated
        """
        weights_updated = 0
        
        # Get active inputs
        active_inputs = np.where(input_activation > 0.3)[0]
        
        # Get target neurons based on reasoning mode
        target_neurons = []
        if reasoning_mode in self.reasoning_circuits:
            target_neurons = self.reasoning_circuits[reasoning_mode]
        
        if not target_neurons:
            return 0
        
        # 1. Update input → reasoning circuit connections
        for input_neuron in active_inputs:
            for target_neuron in random.sample(target_neurons, min(15, len(target_neurons))):
                if (input_neuron, target_neuron) in self.specialized_learning_synapses:
                    current_weight = self.synaptic_weights[input_neuron, target_neuron]
                    
                    # Weight change based on error and input activation
                    weight_change = learn_rate * error * input_activation[input_neuron]
                    
                    new_weight = max(-1.0, min(1.0, current_weight + weight_change))
                    
                    if abs(new_weight - current_weight) > 0.001:
                        self.synaptic_weights[input_neuron, target_neuron] = new_weight
                        weights_updated += 1
        
        # 2. Update within-circuit connections for coherence
        if abs(error) < 0.3:  # Reasonably correct reasoning
            # Strengthen internal connections
            for i, pre in enumerate(target_neurons):
                for post in random.sample(target_neurons, min(10, len(target_neurons))):
                    if pre != post and (pre, post) in self.specialized_learning_synapses:
                        current_weight = self.synaptic_weights[pre, post]
                        weight_change = learn_rate * (1 - abs(error)) * 0.3
                        new_weight = min(1.0, current_weight + weight_change)
                        
                        if abs(new_weight - current_weight) > 0.001:
                            self.synaptic_weights[pre, post] = new_weight
                            weights_updated += 1
        
        # 3. Mode-specific updates
        if reasoning_mode == "deductive":
            weights_updated += self._update_deductive_pathways(error, network_state, learn_rate)
        elif reasoning_mode == "inductive":
            weights_updated += self._update_inductive_pathways(error, network_state, learn_rate)
        elif reasoning_mode == "abductive":
            weights_updated += self._update_abductive_pathways(error, network_state, learn_rate)
        
        return weights_updated
    
    def _reallocate_neurons_to_reasoning(self, target_region, count):
        """Reallocate neurons from other regions to reasoning regions"""
        target_size = int(self.neuron_count * 0.25)
        current_size = len(self.regions[target_region]['neurons'])
        
        if current_size >= target_size:
            return
        
        # Find donor regions
        donor_candidates = []
        excluded_regions = {'higher_cognition', 'metacognition', 'decision', 'memory'}
        
        for region_name, region in self.regions.items():
            if region_name not in excluded_regions:
                max_donate = len(region['neurons']) // 3
                if max_donate > 0:
                    donor_candidates.append((region_name, max_donate))
        
        donor_candidates.sort(key=lambda x: x[1], reverse=True)
        
        neurons_needed = target_size - current_size
        
        for donor_name, max_donate in donor_candidates:
            if neurons_needed <= 0:
                break
            
            to_take = min(max_donate, neurons_needed)
            
            donor_neurons = self.regions[donor_name]['neurons']
            taken_neurons = donor_neurons[:to_take]
            self.regions[donor_name]['neurons'] = donor_neurons[to_take:]
            
            self.regions[target_region]['neurons'].extend(taken_neurons)
            neurons_needed -= to_take
            
            print(f"[ReasoningSNN] Reallocated {to_take} neurons from {donor_name} to {target_region}")
    
    # Deductive reasoning implementation
    def _deductive_reasoning(self, input_text):
        """Perform deductive reasoning with neural processing"""
        # Process input through network
        activation = self.vectorize_input(input_text)
        result = self.process_input(activation)
        
        # Extract syllogism components
        premises, conclusion = self._extract_syllogism(input_text)
        
        if not premises:
            return {
                "valid": False,
                "error": "No valid premises found in input",
                "conclusion": None,
                "confidence": 0.0
            }
        
        # Neural validation using deductive circuit
        valid, confidence = self._validate_syllogism_neural(premises, conclusion, result)
        
        if valid:
            return {
                "valid": True,
                "premises": premises,
                "conclusion": conclusion,
                "confidence": confidence,
                "circuit_activation": self._get_circuit_activation('deductive', result)
            }
        else:
            # Generate valid conclusion
            valid_conclusion = self._generate_conclusion_neural(premises, result)
            
            return {
                "valid": False,
                "premises": premises,
                "attempted_conclusion": conclusion,
                "valid_conclusion": valid_conclusion,
                "confidence": confidence,
                "circuit_activation": self._get_circuit_activation('deductive', result)
            }
    
    def _extract_syllogism(self, input_text):
        """Extract premises and conclusion from input text"""
        premises = []
        conclusion = None
        
        # Enhanced patterns for syllogism extraction
        premise_patterns = [
            r"(?:all|every)\s+(\w+)s?\s+(?:is|are)\s+(\w+)s?",
            r"no\s+(\w+)s?\s+(?:is|are)\s+(\w+)s?",
            r"some\s+(\w+)s?\s+(?:is|are)\s+(\w+)s?",
            r"(\w+)\s+(?:is|are)\s+(\w+)s?"
        ]
        
        text_lower = input_text.lower()
        
        # Extract premises
        for pattern in premise_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1)
                    predicate = match.group(2)
                    quantifier = "all"
                    negative = False
                    
                    if "no" in match.group(0):
                        quantifier = "no"
                        negative = True
                    elif "some" in match.group(0):
                        quantifier = "some"
                    
                    # Skip if this is the conclusion
                    if any(marker in match.group(0) for marker in ["therefore", "thus", "so"]):
                        continue
                        
                    premises.append({
                        "subject": subject,
                        "predicate": predicate,
                        "quantifier": quantifier,
                        "negative": negative
                    })
        
        # Extract conclusion
        conclusion_pattern = r"(?:therefore|thus|so|hence),?\s+(?:all|every|no|some)?\s*(\w+)s?\s+(?:is|are)\s+(\w+)s?"
        match = re.search(conclusion_pattern, text_lower)
        
        if match and len(match.groups()) >= 2:
            subject = match.group(1)
            predicate = match.group(2)
            quantifier = "all"
            negative = False
            
            if "no" in match.group(0):
                quantifier = "no"
                negative = True
            elif "some" in match.group(0):
                quantifier = "some"
                
            conclusion = {
                "subject": subject,
                "predicate": predicate,
                "quantifier": quantifier,
                "negative": negative
            }
        
        return premises, conclusion
    
    def _validate_syllogism_neural(self, premises, conclusion, network_state):
        """Validate syllogism using neural processing"""
        # Create composite activation pattern representing the syllogism
        syllogism_activation = self._encode_syllogism_structure(premises, conclusion)
        
        # Process through deductive reasoning circuit
        if 'deductive' in self.reasoning_circuits:
            circuit_neurons = self.reasoning_circuits['deductive']
            
            # Simulate resonance in deductive circuit
            circuit_response = self._simulate_circuit_resonance(
                syllogism_activation, 
                circuit_neurons,
                iterations=10
            )
            
            # Analyze pattern coherence for logical validity
            coherence_metrics = self._analyze_logical_coherence(circuit_response)
            
            # Check for specific neural signatures of valid syllogisms
            validity_signatures = self._detect_validity_signatures(circuit_response)
            
            # Combine neural measures for validity assessment
            neural_validity = (
                coherence_metrics['temporal_consistency'] * 0.3 +
                coherence_metrics['spatial_coherence'] * 0.3 +
                validity_signatures['transitive_closure'] * 0.2 +
                validity_signatures['logical_binding'] * 0.2
            )
            
            # Apply confidence scaling based on circuit activation strength
            circuit_activation = self._get_circuit_activation('deductive', network_state)
            confidence = neural_validity * (0.5 + 0.5 * circuit_activation)
            
            # Determine validity based on threshold
            valid = confidence > self.confidence_thresholds['deductive']
            
            return valid, confidence
        
        return False, 0.0
    
    def _encode_syllogism_structure(self, premises, conclusion):
        """Encode syllogism components into neural activation pattern"""
        activation = np.zeros(self.neuron_count)
        
        # Encode premises with structured representation
        for i, premise in enumerate(premises):
            # Create distributed representation for each premise component
            subject_pattern = self._get_concept_pattern(premise['subject'])
            predicate_pattern = self._get_concept_pattern(premise['predicate'])
            
            # Encode quantifier using dedicated neural populations
            quantifier_pattern = self._encode_quantifier(premise['quantifier'])
            
            # Bind components using tensor product
            premise_pattern = self._bind_triple(
                subject_pattern, 
                predicate_pattern, 
                quantifier_pattern
            )
            
            # Add temporal ordering information
            temporal_modulation = np.exp(-i * 0.2)  # Earlier premises have stronger activation
            
            # Overlay premise pattern onto activation
            activation += premise_pattern * temporal_modulation * 0.6
        
        # Encode conclusion with expectation modulation
        if conclusion:
            conclusion_pattern = self._bind_triple(
                self._get_concept_pattern(conclusion['subject']),
                self._get_concept_pattern(conclusion['predicate']),
                self._encode_quantifier(conclusion['quantifier'])
            )
            
            # Conclusion gets weaker initial activation (to be validated)
            activation += conclusion_pattern * 0.3
        
        return activation
    
    def _check_logical_validity(self, premises, conclusion):
        """Check logical validity of syllogism"""
        if not premises or not conclusion:
            return False
        
        # Simple validation for basic syllogistic forms
        # Barbara: All A are B, All B are C, therefore All A are C
        if len(premises) == 2:
            p1, p2 = premises
            
            # Check for Barbara pattern
            if (p1['quantifier'] == 'all' and p2['quantifier'] == 'all' and
                conclusion['quantifier'] == 'all'):
                
                # Check if middle term connects properly
                if (p1['predicate'] == p2['subject'] and
                    p1['subject'] == conclusion['subject'] and
                    p2['predicate'] == conclusion['predicate']):
                    return True
        
        return False
    
    def _generate_conclusion_neural(self, premises, network_state):
        """Generate valid conclusion using neural pattern completion"""
        if not premises:
            return None
        
        # Encode premises into neural representation
        premise_activation = self._encode_premises_only(premises)
        
        # Activate deductive reasoning circuit with premises
        if 'deductive' in self.reasoning_circuits:
            circuit_neurons = self.reasoning_circuits['deductive']
            
            # Run pattern completion in circuit
            completed_pattern = self._run_pattern_completion(
                premise_activation,
                circuit_neurons,
                iterations=15
            )
            
            # Extract conclusion from completed pattern
            conclusion_pattern = self._extract_conclusion_pattern(completed_pattern)
            
            # Decode neural pattern back to symbolic form
            conclusion = self._decode_conclusion_pattern(conclusion_pattern)
            
            return conclusion
        
        return None
    
    def _run_pattern_completion(self, partial_pattern, circuit_neurons, iterations=15):
        """Run pattern completion in a neural circuit"""
        # Initialize with partial pattern
        full_pattern = partial_pattern.copy()
        
        # Create attractor dynamics for pattern completion
        for t in range(iterations):
            # Extract circuit state
            circuit_state = full_pattern[circuit_neurons]
            
            # Get circuit connectivity
            circuit_weights = self._extract_circuit_connectivity(circuit_neurons)
            
            # Compute recurrent activation
            new_circuit_state = np.tanh(np.dot(circuit_weights, circuit_state))
            
            # Blend with input (gradual influence decrease)
            alpha = 0.8 * np.exp(-t/10)  # Exponential decay
            new_circuit_state = alpha * circuit_state + (1-alpha) * new_circuit_state
            
            # Update full pattern
            full_pattern[circuit_neurons] = new_circuit_state
            
            # Spread activation to connected regions
            self._spread_activation(full_pattern, circuit_neurons)
        
        return full_pattern
    
    def _extract_circuit_connectivity(self, circuit_neurons):
        """Extract connectivity matrix for a circuit"""
        n = len(circuit_neurons)
        circuit_weights = np.zeros((n, n))
        
        for i, pre in enumerate(circuit_neurons):
            for j, post in enumerate(circuit_neurons):
                if pre != post:
                    circuit_weights[i, j] = self.synaptic_weights[pre, post]
        
        return circuit_weights
    
    # Abductive reasoning implementation
    def _abductive_reasoning(self, observation):
        """Perform abductive reasoning with neural processing"""
        # Process observation through network
        activation = self.vectorize_input(observation)
        result = self.process_input(activation)
        
        # Extract entities from observation
        entities = self._extract_entities(observation)
        
        # Get active concepts from network activation
        active_concepts = result.get('active_concepts', {})
        
        # Generate hypotheses using neural ensemble
        hypotheses = self.generate_ensemble_hypotheses(observation)
        
        # Enhance with domain knowledge
        for hypothesis in hypotheses:
            if 'components' in hypothesis:
                self._apply_relation_constraints(hypothesis)
                self._apply_relation_implications(hypothesis)
        
        # Score hypotheses using abductive circuit
        scored_hypotheses = self._score_hypotheses_neural(hypotheses, result)
        
        # Sort by score
        scored_hypotheses.sort(key=lambda x: x['neural_score'], reverse=True)
        
        return {
            "observation": observation,
            "entities": entities,
            "hypotheses": scored_hypotheses,
            "active_concepts": active_concepts,
            "circuit_activation": self._get_circuit_activation('abductive', result)
        }
    
    def _score_hypotheses_neural(self, hypotheses, network_state):
        """Score hypotheses using neural processing with abductive circuit"""
        scored = []
        
        if 'abductive' in self.reasoning_circuits:
            circuit_neurons = self.reasoning_circuits['abductive']
            
            for hypothesis in hypotheses:
                # Encode hypothesis into neural pattern
                hyp_activation = self._encode_hypothesis(hypothesis)
                
                # Test hypothesis coherence in abductive circuit
                coherence_response = self._test_hypothesis_coherence(
                    hyp_activation,
                    circuit_neurons,
                    network_state
                )
                
                # Measure explanatory power
                explanatory_metrics = self._measure_explanatory_power(
                    hypothesis,
                    coherence_response,
                    network_state
                )
                
                # Calculate neural score
                neural_score = (
                    coherence_response['resonance'] * 0.3 +
                    coherence_response['stability'] * 0.2 +
                    explanatory_metrics['coverage'] * 0.2 +
                    explanatory_metrics['simplicity'] * 0.15 +
                    explanatory_metrics['consistency'] * 0.15
                )
                
                hypothesis['neural_score'] = neural_score
                hypothesis['coherence_metrics'] = coherence_response
                hypothesis['explanatory_metrics'] = explanatory_metrics
                scored.append(hypothesis)
        
        return scored
   
    def _extract_entities(self, text):
       """Extract entities from text"""
       entities = {
           'persons': [],
           'relations': [],
           'concepts': []
       }
       
       text_lower = text.lower()
       
       # Extract persons
       if 'entity_types' in self.domain_knowledge:
           for person in self.domain_knowledge['entity_types'].get('person', []):
               if person in text_lower:
                   entities['persons'].append(person)
       
       # Extract relations
       relation_words = ['trusts', 'likes', 'knows', 'fears', 'loves', 'hates']
       for relation in relation_words:
           if relation in text_lower:
               entities['relations'].append(relation)
       
       # Extract concepts
       concept_words = ['trust', 'fear', 'knowledge', 'belief', 'doubt']
       for concept in concept_words:
           if concept in text_lower:
               entities['concepts'].append(concept)
       
       return entities
   
    def _apply_relation_constraints(self, hypothesis):
       """Apply relation constraints from domain knowledge"""
       if 'components' not in hypothesis:
           return
       
       components = hypothesis['components']
       subject = components.get('subject')
       predicate = components.get('predicate')
       obj = components.get('object')
       
       # Check if relation has constraints
       if predicate in self.domain_knowledge.get('relation_properties', {}):
           properties = self.domain_knowledge['relation_properties'][predicate]
           
           # Apply reflexivity
           if subject == obj and 'reflexivity' in properties:
               hypothesis['confidence'] *= properties['reflexivity']
           
           # Check entity type constraints if available
           if 'entity_types' in self.domain_knowledge:
               # Verify subject and object are appropriate types
               person_list = self.domain_knowledge['entity_types'].get('person', [])
               if subject in person_list and obj in person_list:
                   hypothesis['confidence'] *= 1.1  # Boost for valid entity types
   
    def _apply_relation_implications(self, hypothesis):
       """Apply relation implications from domain knowledge"""
       if 'components' not in hypothesis:
           return
       
       predicate = hypothesis['components'].get('predicate')
       
       # Check if relation has implications
       if predicate in self.domain_knowledge.get('relation_implications', {}):
           implications = self.domain_knowledge['relation_implications'][predicate]
           hypothesis['implications'] = implications
           
           # Boost confidence if implications are satisfied
           hypothesis['confidence'] *= 1.05
   
    # Inductive reasoning implementation
    def _inductive_reasoning(self, input_text):
       """Perform inductive reasoning with neural processing"""
       # Process input through network
       activation = self.vectorize_input(input_text)
       result = self.process_input(activation)
       
       # Extract examples from input
       examples = self._extract_examples(input_text)
       
       if not examples:
           return {
               "valid": False,
               "error": "No valid examples found in input",
               "generalization": None,
               "confidence": 0.0
           }
       
       # Generate generalization using neural processing
       generalization, confidence = self._generate_generalization_neural(examples, result)
       
       return {
           "valid": True,
           "examples": examples,
           "generalization": generalization,
           "confidence": confidence,
           "circuit_activation": self._get_circuit_activation('inductive', result)
       }
   
    def _extract_examples(self, input_text):
       """Extract examples from input text"""
       examples = []
       
       # Pattern for examples
       example_patterns = [
           r"(\w+)\s+(?:is|are)\s+(\w+)",
           r"(\w+)\s+(\w+)\s+(\w+)"  # Subject verb object
       ]
       
       text_lower = input_text.lower()
       
       # Look for example markers
       example_markers = ["example", "instance", "case", "such as", "like"]
       
       for pattern in example_patterns:
           matches = re.finditer(pattern, text_lower)
           for match in matches:
               # Skip if it's a conclusion
               if any(marker in match.group(0) for marker in ["therefore", "thus", "so"]):
                   continue
               
               if len(match.groups()) >= 2:
                   example = {
                       "subject": match.group(1),
                       "predicate": match.group(2) if len(match.groups()) > 2 else "is",
                       "object": match.group(3) if len(match.groups()) > 2 else match.group(2)
                   }
                   examples.append(example)
       
       return examples
   
    def _generate_generalization_neural(self, examples, network_state):
        """Generate generalization using neural pattern extraction"""
        if not examples:
            return None, 0.0
        
        # Encode examples into neural patterns
        example_patterns = []
        for example in examples:
            pattern = self._encode_example(example)
            example_patterns.append(pattern)
        
        # Use inductive circuit for pattern extraction
        if 'inductive' in self.reasoning_circuits:
            circuit_neurons = self.reasoning_circuits['inductive']
            
            # Overlay example patterns
            combined_pattern = np.mean(example_patterns, axis=0)
            
            # Extract invariant features through circuit processing
            invariant_features = self._extract_invariant_features(
                combined_pattern,
                circuit_neurons,
                example_patterns
            )
            
            # Generate generalization from invariant features
            generalization = self._construct_generalization(invariant_features)
            
            # Calculate confidence based on feature consistency
            confidence = self._calculate_inductive_confidence(
                invariant_features,
                example_patterns,
                network_state
            )
            
            return generalization, confidence
        
        return "No clear pattern found", 0.3

    def _extract_invariant_features(self, combined_pattern, circuit_neurons, example_patterns):
        """Extract features that remain consistent across examples"""
        # Run combined pattern through inductive circuit
        circuit_response = self._simulate_circuit_resonance(
            combined_pattern,
            circuit_neurons,
            iterations=12
        )
        
        # Identify stable features across examples
        feature_consistency = np.zeros(len(circuit_neurons))
        
        for pattern in example_patterns:
            # Test each example against circuit response
            example_response = self._simulate_circuit_resonance(
                pattern,
                circuit_neurons,
                iterations=5
            )
            
            # Measure similarity to combined response
            similarity = np.corrcoef(circuit_response[-1], example_response[-1])[0, 1]
            feature_consistency += (example_response[-1] > 0.5) * similarity
        
        # Normalize by number of examples
        feature_consistency /= len(example_patterns)
        
        # Extract most consistent features
        invariant_neurons = np.where(feature_consistency > 0.7)[0]
        invariant_features = {
            'neurons': circuit_neurons[invariant_neurons],
            'consistency': feature_consistency[invariant_neurons],
            'pattern': circuit_response[-1, invariant_neurons]
        }
        
        return invariant_features

    def _construct_generalization(self, invariant_features):
        """Construct symbolic generalization from neural invariant features"""
        # Decode pattern back to concepts
        pattern = np.zeros(self.neuron_count)
        for i, neuron in enumerate(invariant_features['neurons']):
            pattern[neuron] = invariant_features['pattern'][i]
        
        # Find closest matching concepts
        activated_concepts = []
        for concept, concept_neurons in self.concept_mappings.items():
            overlap = np.mean([pattern[n] for n in concept_neurons if n < len(pattern)])
            if overlap > 0.5:
                activated_concepts.append((concept, overlap))
        
        # Sort by activation strength
        activated_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Construct generalization statement
        if len(activated_concepts) >= 2:
            main_concept = activated_concepts[0][0]
            property_concept = activated_concepts[1][0]
            return f"Things that are {main_concept} tend to be {property_concept}"
        elif activated_concepts:
            return f"Pattern involves {activated_concepts[0][0]}"
        else:
            return "General pattern detected"

    def _calculate_inductive_confidence(self, invariant_features, example_patterns, network_state):
        """Calculate confidence in inductive generalization"""
        # Base confidence on feature consistency
        avg_consistency = np.mean(invariant_features['consistency']) if len(invariant_features['consistency']) > 0 else 0
        
        # Factor in number of examples
        example_factor = min(1.0, len(example_patterns) / 5.0)
        
        # Factor in circuit activation strength
        circuit_activation = self._get_circuit_activation('inductive', network_state)
        
        # Combine factors
        confidence = (
            avg_consistency * 0.4 +
            example_factor * 0.3 +
            circuit_activation * 0.3
        )
        
        return confidence
   
    # Helper methods
    def _get_circuit_activation(self, circuit_name, network_state):
       """Get activation level of a specific reasoning circuit"""
       if circuit_name in self.reasoning_circuits:
           circuit_neurons = self.reasoning_circuits[circuit_name]
           active_neurons = network_state.get('active_neurons', set())
           
           active_count = len(set(circuit_neurons).intersection(active_neurons))
           return active_count / len(circuit_neurons) if circuit_neurons else 0
       
       return 0.0
   
    def _calculate_pattern_stability(self, network_state):
       """Calculate stability of activation patterns"""
       # Simple stability metric based on membrane potentials
       membrane_potentials = network_state.get('membrane_potentials', np.zeros(self.neuron_count))
       variance = np.var(membrane_potentials)
       stability = 1.0 / (1.0 + variance)
       return stability
   
    def _detect_premise_activation(self, network_state):
       """Detect activation patterns associated with premises"""
       # Check for activation in areas associated with conditional reasoning
       if 'higher_cognition' in self.regions:
           return self.regions['higher_cognition']['activation']
       return 0.0
   
    def _detect_conclusion_activation(self, network_state):
       """Detect activation patterns associated with conclusions"""
       # Check for activation in decision regions
       if 'decision' in self.regions:
           return self.regions['decision']['activation']
       return 0.0
   
    def _calculate_pattern_diversity(self, network_state):
       """Calculate diversity of activation patterns for inductive reasoning"""
       active_neurons = network_state.get('active_neurons', set())
       total_neurons = self.neuron_count
       
       # Diversity as ratio of active neurons
       diversity = len(active_neurons) / total_neurons if total_neurons > 0 else 0
       return diversity
   
    def _calculate_generalization_strength(self, network_state):
       """Calculate strength of generalization patterns"""
       # Use integration metric as proxy for generalization
       return network_state.get('integration', 0.5)
   
    def _calculate_hypothesis_coherence(self, network_state):
       """Calculate coherence of hypothesis-related activations"""
       # Use differentiation metric as proxy for hypothesis clarity
       return network_state.get('differentiation', 0.5)
   
    def _calculate_explanatory_power(self, network_state):
       """Calculate explanatory power of hypotheses"""
       # Use phi metric as proxy for explanatory power
       return network_state.get('phi', 0.5)
   
    def _update_deductive_pathways(self, error, network_state, learn_rate):
       """Update pathways specific to deductive reasoning"""
       weights_updated = 0
       
       # Strengthen logical flow pathways
       if 'higher_cognition' in self.regions and 'decision' in self.regions:
           cognition_neurons = self.regions['higher_cognition']['neurons']
           decision_neurons = self.regions['decision']['neurons']
           
           # Sample connections to update
           for _ in range(50):
               if cognition_neurons and decision_neurons:
                   pre = random.choice(cognition_neurons)
                   post = random.choice(decision_neurons)
                   
                   if (pre, post) in self.specialized_learning_synapses:
                       current_weight = self.synaptic_weights[pre, post]
                       weight_change = learn_rate * (1 - abs(error)) * 0.2
                       new_weight = max(-1.0, min(1.0, current_weight + weight_change))
                       
                       if abs(new_weight - current_weight) > 0.001:
                           self.synaptic_weights[pre, post] = new_weight
                           weights_updated += 1
       
       return weights_updated
   
    def _update_inductive_pathways(self, error, network_state, learn_rate):
       """Update pathways specific to inductive reasoning"""
       weights_updated = 0
       
       # Strengthen pattern detection pathways
       if 'memory' in self.regions and 'inductive' in self.reasoning_circuits:
           memory_neurons = self.regions['memory']['neurons']
           inductive_neurons = self.reasoning_circuits['inductive']
           
           # Strengthen memory to inductive circuit connections
           for _ in range(50):
               if memory_neurons and inductive_neurons:
                   pre = random.choice(memory_neurons)
                   post = random.choice(inductive_neurons)
                   
                   if (pre, post) in self.specialized_learning_synapses:
                       current_weight = self.synaptic_weights[pre, post]
                       weight_change = learn_rate * (1 - abs(error)) * 0.2
                       new_weight = max(-1.0, min(1.0, current_weight + weight_change))
                       
                       if abs(new_weight - current_weight) > 0.001:
                           self.synaptic_weights[pre, post] = new_weight
                           weights_updated += 1
       
       return weights_updated
   
    def _update_abductive_pathways(self, error, network_state, learn_rate):
        """Update pathways specific to abductive reasoning"""
        weights_updated = 0
       
        # Strengthen hypothesis generation pathways
        if 'metacognition' in self.regions and 'abductive' in self.reasoning_circuits:
           meta_neurons = self.regions['metacognition']['neurons']
           abductive_neurons = self.reasoning_circuits['abductive']
           
           # Strengthen metacognition to abductive circuit connections
           for _ in range(50):
               if meta_neurons and abductive_neurons:
                   pre = random.choice(meta_neurons)
                   post = random.choice(abductive_neurons)
                   
                   if (pre, post) in self.specialized_learning_synapses:
                       current_weight = self.synaptic_weights[pre, post]
                       weight_change = learn_rate * (1 - abs(error)) * 0.2
                       new_weight = max(-1.0, min(1.0, current_weight + weight_change))
                       
                       if abs(new_weight - current_weight) > 0.001:
                           self.synaptic_weights[pre, post] = new_weight
                           weights_updated += 1
       
        return weights_updated

    def _simulate_circuit_resonance(self, input_activation, circuit_neurons, iterations=10):
        """Simulate resonance dynamics in a reasoning circuit"""
        # Initialize circuit state
        circuit_state = np.zeros((iterations, len(circuit_neurons)))
        
        # Create circuit connectivity matrix
        circuit_weights = self._extract_circuit_connectivity(circuit_neurons)
        
        # Initial activation from input
        initial_activation = np.zeros(len(circuit_neurons))
        for i, neuron in enumerate(circuit_neurons):
            initial_activation[i] = input_activation[neuron]
        
        circuit_state[0] = initial_activation
        
        # Simulate circuit dynamics
        for t in range(1, iterations):
            # Recurrent dynamics within circuit
            recurrent_input = np.dot(circuit_weights, circuit_state[t-1])
            
            # Apply nonlinear activation function
            circuit_state[t] = np.tanh(recurrent_input + initial_activation * 0.5)
            
            # Add noise for robustness
            circuit_state[t] += np.random.normal(0, 0.01, size=len(circuit_neurons))
        
        return circuit_state

    def _analyze_logical_coherence(self, circuit_response):
        """Analyze coherence patterns indicating logical validity"""
        metrics = {}
        
        # Temporal consistency: stable patterns indicate valid reasoning
        temporal_diffs = np.diff(circuit_response, axis=0)
        temporal_variance = np.mean(np.var(temporal_diffs, axis=0))
        metrics['temporal_consistency'] = 1.0 / (1.0 + temporal_variance)
        
        # Spatial coherence: synchronized activity indicates unified conclusion
        correlations = np.corrcoef(circuit_response.T)
        np.fill_diagonal(correlations, 0)  # Remove self-correlations
        metrics['spatial_coherence'] = np.mean(np.abs(correlations))
        
        # Pattern completion: convergence to attractor state
        final_states = circuit_response[-3:]
        convergence = 1.0 - np.mean(np.std(final_states, axis=0))
        metrics['pattern_completion'] = max(0, convergence)
        
        return metrics

    def _detect_validity_signatures(self, circuit_response):
        """Detect neural signatures specific to valid logical inferences"""
        signatures = {}
        
        # Transitive closure: activation flows from premises to conclusion
        activation_flow = np.mean(circuit_response, axis=1)
        flow_gradient = np.gradient(activation_flow)
        signatures['transitive_closure'] = np.mean(flow_gradient[:-2]) > 0
        
        # Logical binding: stable binding between related concepts
        binding_stability = self._measure_binding_stability(circuit_response)
        signatures['logical_binding'] = binding_stability
        
        # Contradiction detection: absence of oscillatory patterns
        oscillation_score = self._detect_oscillations(circuit_response)
        signatures['no_contradiction'] = 1.0 - oscillation_score
        
        return signatures
    
    def _detect_oscillations(self, circuit_response):
        """Detect oscillatory patterns indicating contradiction"""
        # Calculate frequency components using FFT
        fft_magnitudes = []
        
        for neuron_response in circuit_response.T:
            fft = np.fft.fft(neuron_response)
            magnitudes = np.abs(fft[1:len(fft)//2])  # Ignore DC component
            fft_magnitudes.append(magnitudes)
        
        # High frequency power indicates oscillations
        avg_magnitudes = np.mean(fft_magnitudes, axis=0)
        high_freq_power = np.mean(avg_magnitudes[len(avg_magnitudes)//2:])
        total_power = np.mean(avg_magnitudes)
        
        oscillation_score = high_freq_power / (total_power + 1e-8)
        
        return oscillation_score
    
    def _measure_binding_stability(self, circuit_response):
        """Measure stability of conceptual binding in circuit"""
        # Calculate temporal autocorrelation
        autocorrelations = []
        
        for lag in range(1, min(5, len(circuit_response))):
            corr = np.corrcoef(circuit_response[:-lag].flatten(), 
                              circuit_response[lag:].flatten())[0, 1]
            autocorrelations.append(corr)
        
        # High autocorrelation indicates stable binding
        return np.mean(autocorrelations) if autocorrelations else 0.5

    def _extract_conclusion_pattern(self, completed_pattern):
        """Extract conclusion component from completed reasoning pattern"""
        # Identify neurons with late-emerging activation (conclusion emerges last)
        temporal_profile = np.mean(completed_pattern, axis=1)
        late_activation = temporal_profile[-5:] - temporal_profile[:5]
        
        # Find neurons that became active late in processing
        conclusion_neurons = np.where(late_activation[-1] > 0.2)[0]
        
        # Extract their final activation pattern
        conclusion_pattern = completed_pattern[-1, conclusion_neurons]
        
        return conclusion_pattern, conclusion_neurons

    def _test_hypothesis_coherence(self, hyp_activation, circuit_neurons, network_state):
        """Test hypothesis coherence in abductive reasoning circuit"""
        # Combine hypothesis with current network state
        combined_activation = hyp_activation * 0.7 + network_state.get('membrane_potentials', np.zeros(self.neuron_count)) * 0.3
        
        # Run hypothesis through circuit
        circuit_response = self._simulate_circuit_resonance(
            combined_activation,
            circuit_neurons,
            iterations=8
        )
        
        # Measure coherence metrics
        coherence_metrics = {
            'resonance': self._measure_resonance_strength(circuit_response),
            'stability': self._measure_pattern_stability_temporal(circuit_response),
            'binding': self._measure_binding_coherence(circuit_response),
            'convergence': self._measure_convergence_speed(circuit_response)
        }
        
        return coherence_metrics
    
    def _measure_resonance_strength(self, circuit_response):
        """Measure resonance strength in circuit dynamics"""
        # Calculate energy increase over time
        energies = np.sum(circuit_response**2, axis=1)
        
        # Resonance shows energy buildup
        energy_increase = (energies[-1] - energies[0]) / (energies[0] + 1e-8)
        
        # Normalize to [0, 1]
        resonance = 1 / (1 + np.exp(-energy_increase))
        
        return resonance
    
    def _measure_pattern_stability_temporal(self, circuit_response):
        """Measure temporal stability of activation pattern"""
        # Calculate variance over time
        temporal_variance = np.var(circuit_response, axis=0)
        
        # Low variance indicates stability
        stability = 1 / (1 + np.mean(temporal_variance))
        
        return stability
    
    def _measure_binding_coherence(self, circuit_response):
        """Measure coherence of bound representations"""
        # Calculate mutual information between neurons
        correlations = np.corrcoef(circuit_response.T)
        
        # Remove diagonal
        np.fill_diagonal(correlations, 0)
        
        # Average absolute correlation as coherence measure
        coherence = np.mean(np.abs(correlations))
        
        return coherence

    def _measure_convergence_speed(self, circuit_response):
        """Measure how quickly the circuit converges to a stable state"""
        # Calculate differences between consecutive states
        diffs = np.diff(circuit_response, axis=0)
        diff_magnitudes = np.sum(diffs**2, axis=1)
        
        # Find convergence point (where changes become small)
        threshold = np.mean(diff_magnitudes) * 0.1
        convergence_point = np.argmax(diff_magnitudes < threshold)
        
        # Convert to speed metric (earlier convergence = higher speed)
        speed = 1 - (convergence_point / len(circuit_response))
        
        return speed

    def _measure_explanatory_power(self, hypothesis, coherence_response, network_state):
        """Measure explanatory power of hypothesis"""
        metrics = {}
        
        # Coverage: how much of the observation does hypothesis explain
        if 'components' in hypothesis:
            components = hypothesis['components']
            activated_concepts = network_state.get('active_concepts', {})
            
            covered_concepts = 0
            for concept in [components.get('subject'), components.get('predicate'), components.get('object')]:
                if concept and concept in activated_concepts:
                    covered_concepts += activated_concepts[concept]
            
            total_activation = sum(activated_concepts.values()) if activated_concepts else 1
            metrics['coverage'] = covered_concepts / total_activation if total_activation > 0 else 0
        else:
            metrics['coverage'] = 0.5
        
        # Simplicity: prefer simpler explanations (Occam's razor)
        complexity = self._measure_hypothesis_complexity(hypothesis)
        metrics['simplicity'] = 1.0 / (1.0 + complexity)
        
        # Consistency: alignment with existing knowledge
        consistency = self._check_knowledge_consistency(hypothesis)
        metrics['consistency'] = consistency
        
        return metrics
    
    def _check_knowledge_consistency(self, hypothesis):
        """Check consistency with domain knowledge"""
        consistency = 1.0
        
        if 'components' in hypothesis:
            predicate = hypothesis['components'].get('predicate')
            
            # Check against known relation properties
            if predicate in self.domain_knowledge.get('relation_properties', {}):
                properties = self.domain_knowledge['relation_properties'][predicate]
                
                # Apply consistency penalties for violations
                subject = hypothesis['components'].get('subject')
                obj = hypothesis['components'].get('object')
                
                # Check reflexivity
                if subject == obj:
                    consistency *= properties.get('reflexivity', 1.0)
                
                # Check for contradictions with known implications
                if predicate in self.domain_knowledge.get('relation_implications', {}):
                    implications = self.domain_knowledge['relation_implications'][predicate]
                    # Simple consistency check
                    consistency *= 0.9  # Slight penalty for complexity
        
        return consistency
    
    def _measure_hypothesis_complexity(self, hypothesis):
        """Measure complexity of hypothesis"""
        complexity = 0
        
        if 'components' in hypothesis:
            components = hypothesis['components']
            
            # Count non-empty components
            for key, value in components.items():
                if value:
                    complexity += 1
            
            # Additional complexity for compound concepts
            for value in components.values():
                if isinstance(value, str) and len(value.split()) > 1:
                    complexity += 0.5
        
        return complexity

    def _get_concept_pattern(self, concept):
        """Get neural pattern for a concept"""
        pattern = np.zeros(self.neuron_count)
        
        # Use existing concept mappings if available
        if concept in self.concept_mappings:
            neurons = self.concept_mappings[concept]
            for neuron in neurons:
                pattern[neuron] = 0.8 + 0.2 * random.random()
        else:
            # Create a distributed pattern for unknown concepts
            # Use hashing for consistency
            hash_val = hash(concept)
            neurons_to_activate = int(self.neuron_count * 0.05)  # 5% of neurons
            
            for i in range(neurons_to_activate):
                neuron_idx = (hash_val + i * 137) % self.neuron_count  # Deterministic spread
                pattern[neuron_idx] = 0.7 + 0.3 * random.random()
        
        return pattern

    def _encode_quantifier(self, quantifier):
        """Encode logical quantifier into neural pattern"""
        pattern = np.zeros(self.neuron_count)
        
        # Define quantifier regions (different neural populations for different quantifiers)
        quantifier_regions = {
            'all': range(0, int(self.neuron_count * 0.1)),
            'some': range(int(self.neuron_count * 0.1), int(self.neuron_count * 0.2)),
            'no': range(int(self.neuron_count * 0.2), int(self.neuron_count * 0.3)),
            'most': range(int(self.neuron_count * 0.3), int(self.neuron_count * 0.4))
        }
        
        if quantifier in quantifier_regions:
            for neuron in quantifier_regions[quantifier]:
                pattern[neuron] = 0.9 + 0.1 * random.random()
        
        return pattern

    def _bind_triple(self, subject_pattern, predicate_pattern, quantifier_pattern):
        """Bind three patterns using tensor product operation"""
        # Simplified binding using circular convolution
        # This creates a unique pattern that can be decomposed
        bound_pattern = np.zeros(self.neuron_count)
        
        # Circular convolution for binding
        for i in range(self.neuron_count):
            for j in range(self.neuron_count):
                k = (i + j) % self.neuron_count
                bound_pattern[k] += (subject_pattern[i] * predicate_pattern[j] * 
                                   quantifier_pattern[(i-j) % self.neuron_count])
        
        # Normalize
        bound_pattern = bound_pattern / (np.max(np.abs(bound_pattern)) + 1e-8)
        
        # Apply nonlinearity
        bound_pattern = np.tanh(bound_pattern)
        
        return bound_pattern

    def _spread_activation(self, pattern, source_neurons):
        """Spread activation from source neurons to connected regions"""
        # Get average activation of source neurons
        source_activation = np.mean(pattern[source_neurons])
        
        # Find strongly connected neurons
        for source in source_neurons:
            # Get outgoing connections
            connections = np.nonzero(self.synaptic_weights[source, :])[0]
            
            for target in connections:
                if target not in source_neurons:  # Only spread outside circuit
                    weight = self.synaptic_weights[source, target]
                    # Weighted propagation
                    pattern[target] += weight * source_activation * 0.3

    def _decode_conclusion_pattern(self, conclusion_pattern):
        """Decode neural pattern back to symbolic conclusion"""
        if len(conclusion_pattern) == 0:
            return None
        
        # Find closest concept matches
        best_matches = []
        
        for concept, neurons in self.concept_mappings.items():
            # Calculate overlap with conclusion pattern
            overlap = 0
            for i, neuron in enumerate(neurons):
                if neuron < len(conclusion_pattern):
                    overlap += conclusion_pattern[neuron]
            
            avg_overlap = overlap / len(neurons) if neurons else 0
            best_matches.append((concept, avg_overlap))
        
        # Sort by match strength
        best_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Construct conclusion from top matches
        if len(best_matches) >= 2:
            subject = best_matches[0][0]
            predicate_or_object = best_matches[1][0]
            
            # Simple heuristic: if second match is a relation word, use it as predicate
            relation_words = ['is', 'are', 'trusts', 'likes', 'knows', 'fears']
            if predicate_or_object in relation_words:
                return {
                    'subject': subject,
                    'predicate': predicate_or_object,
                    'object': best_matches[2][0] if len(best_matches) > 2 else subject,
                    'quantifier': 'all'  # Default quantifier
                }
            else:
                return {
                    'subject': subject,
                    'predicate': 'is',
                    'object': predicate_or_object,
                    'quantifier': 'all'
                }
        
        return None

    def _encode_hypothesis(self, hypothesis):
        """Encode hypothesis into neural activation pattern"""
        pattern = np.zeros(self.neuron_count)
        
        if 'components' in hypothesis:
            components = hypothesis['components']
            
            # Get patterns for each component
            subject_pattern = self._get_concept_pattern(components.get('subject', ''))
            predicate_pattern = self._get_concept_pattern(components.get('predicate', ''))
            object_pattern = self._get_concept_pattern(components.get('object', ''))
            
            # Combine with weighted sum
            pattern = (subject_pattern * 0.3 + 
                      predicate_pattern * 0.4 + 
                      object_pattern * 0.3)
            
            # Add confidence modulation
            confidence = hypothesis.get('confidence', 0.5)
            pattern *= confidence
        
        return pattern

    def _encode_premises_only(self, premises):
        """Encode premises without conclusion for pattern completion"""
        activation = np.zeros(self.neuron_count)
        
        for i, premise in enumerate(premises):
            # Create structured representation for each premise
            subject_pattern = self._get_concept_pattern(premise['subject'])
            predicate_pattern = self._get_concept_pattern(premise['predicate'])
            quantifier_pattern = self._encode_quantifier(premise['quantifier'])
            
            # Bind components
            premise_pattern = self._bind_triple(
                subject_pattern, 
                predicate_pattern, 
                quantifier_pattern
            )
            
            # Stronger activation for premises
            temporal_modulation = np.exp(-i * 0.1)  # Slight decay for order
            activation += premise_pattern * temporal_modulation * 0.8
        
        return activation

    def _encode_example(self, example):
        """Encode a single example for inductive reasoning"""
        pattern = np.zeros(self.neuron_count)
        
        # Get patterns for example components
        subject_pattern = self._get_concept_pattern(example['subject'])
        predicate_pattern = self._get_concept_pattern(example.get('predicate', 'is'))
        object_pattern = self._get_concept_pattern(example['object'])
        
        # Combine with equal weighting
        pattern = (subject_pattern + predicate_pattern + object_pattern) / 3
        
        return pattern

    def _init_reasoning_circuits(self):
        """Initialize specialized neural circuits for different reasoning types"""
        # Allocate neurons for each reasoning mode
        circuit_size = int(self.neuron_count * 0.1)  # 10% of neurons per circuit
        
        # Get available neurons from reasoning regions
        available_neurons = []
        reasoning_regions = ['higher_cognition', 'metacognition', 'decision']
        
        for region_name in reasoning_regions:
            if region_name in self.regions:
                available_neurons.extend(self.regions[region_name]['neurons'])
        
        # Create circuits
        self.reasoning_circuits = {}
        
        if len(available_neurons) >= circuit_size * 3:
            # Deductive circuit
            self.reasoning_circuits['deductive'] = available_neurons[:circuit_size]
            
            # Inductive circuit
            self.reasoning_circuits['inductive'] = available_neurons[circuit_size:2*circuit_size]
            
            # Abductive circuit
            self.reasoning_circuits['abductive'] = available_neurons[2*circuit_size:3*circuit_size]
            
            # Create specialized connectivity for each circuit
            self._create_circuit_connectivity()
        else:
            print("[ReasoningSNN] Warning: Not enough neurons for all reasoning circuits")
            # Create smaller circuits
            third = len(available_neurons) // 3
            self.reasoning_circuits['deductive'] = available_neurons[:third]
            self.reasoning_circuits['inductive'] = available_neurons[third:2*third]
            self.reasoning_circuits['abductive'] = available_neurons[2*third:]

    def _create_circuit_connectivity(self):
        """Create specialized connectivity patterns for reasoning circuits"""
        # Deductive circuit: feedforward for logical flow
        if 'deductive' in self.reasoning_circuits:
            neurons = self.reasoning_circuits['deductive']
            self._create_feedforward_connectivity(neurons, layers=3, density=0.4)
        
        # Inductive circuit: recurrent for pattern extraction
        if 'inductive' in self.reasoning_circuits:
            neurons = self.reasoning_circuits['inductive']
            self._create_recurrent_connectivity(neurons, density=0.5)
        
        # Abductive circuit: mixed for hypothesis generation
        if 'abductive' in self.reasoning_circuits:
            neurons = self.reasoning_circuits['abductive']
            self._create_mixed_connectivity(neurons, feedforward_density=0.3, recurrent_density=0.3)

    def _create_feedforward_connectivity(self, neurons, layers=3, density=0.4):
        """Create feedforward connectivity pattern"""
        layer_size = len(neurons) // layers
        
        for layer in range(layers - 1):
            start_idx = layer * layer_size
            end_idx = (layer + 1) * layer_size
            next_start = end_idx
            next_end = min((layer + 2) * layer_size, len(neurons))
            
            # Connect current layer to next layer
            for pre in range(start_idx, end_idx):
                for post in range(next_start, next_end):
                    if random.random() < density:
                        weight = 0.5 + 0.5 * random.random()
                        self.synaptic_weights[neurons[pre], neurons[post]] = weight

    def _create_recurrent_connectivity(self, neurons, density=0.5):
        """Create recurrent connectivity pattern"""
        for i, pre in enumerate(neurons):
            for j, post in enumerate(neurons):
                if i != j and random.random() < density:
                    weight = 0.3 + 0.4 * random.random()
                    self.synaptic_weights[pre, post] = weight

    def _create_mixed_connectivity(self, neurons, feedforward_density=0.3, recurrent_density=0.3):
        """Create mixed feedforward and recurrent connectivity"""
        # First create some feedforward structure
        self._create_feedforward_connectivity(neurons, layers=2, density=feedforward_density)
        
        # Then add recurrent connections
        self._create_recurrent_connectivity(neurons, density=recurrent_density)

    def reason(self, input_text, mode=None):
        """Main reasoning interface"""
        if mode is None:
            mode = self.current_mode
        
        self.set_reasoning_mode(mode)
        
        if mode == "deductive":
            return self._deductive_reasoning(input_text)
        elif mode == "inductive":
            return self._inductive_reasoning(input_text)
        elif mode == "abductive":
            return self._abductive_reasoning(input_text)
        else:
            return {"error": f"Unknown reasoning mode: {mode}"}

    def evaluate_reasoning(self, test_data):
        """Evaluate reasoning performance on test data"""
        results = {
            "deductive": {"accuracy": 0, "total": 0},
            "inductive": {"accuracy": 0, "total": 0},
            "abductive": {"accuracy": 0, "total": 0}
        }
        
        for item in test_data:
            mode = item.get("mode", "abductive")
            input_text = item.get("input", "")
            expected = item.get("expected", None)
            
            # Perform reasoning
            result = self.reason(input_text, mode)
            
            # Evaluate result
            if mode == "deductive":
                if expected and "valid" in expected and "valid" in result:
                    if result["valid"] == expected["valid"]:
                        results[mode]["accuracy"] += 1
                    results[mode]["total"] += 1
            
            elif mode == "inductive":
                if expected and "generalization" in expected and "generalization" in result:
                    # Simple string similarity for now
                    if expected["generalization"].lower() in result["generalization"].lower():
                        results[mode]["accuracy"] += 1
                    results[mode]["total"] += 1
            
            elif mode == "abductive":
                if expected and "hypotheses" in expected and "hypotheses" in result:
                    # Check if expected hypothesis is in top results
                    expected_hyp = expected["hypotheses"][0] if expected["hypotheses"] else None
                    if expected_hyp:
                        result_hyps = [h.get("relation", "") for h in result["hypotheses"][:3]]
                        if expected_hyp in result_hyps:
                            results[mode]["accuracy"] += 1
                    results[mode]["total"] += 1
        
        # Calculate accuracy percentages
        for mode in results:
            if results[mode]["total"] > 0:
                results[mode]["accuracy"] = results[mode]["accuracy"] / results[mode]["total"]
        
        return results

    def process_text_input(self, text_input, timesteps=20):
        """
        Process text input using the standardized bidirectional processor.
        
        Args:
            text_input: Text input for reasoning
            timesteps: Number of timesteps for spike patterns
            
        Returns:
            Processed spike patterns for reasoning
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
                
                # Apply reasoning-specific processing if needed
                if self.current_mode == "deductive":
                    # Enhanced processing for logical deduction
                    # This will use the spike patterns as input for specialized processing
                    return spike_patterns
                elif self.current_mode == "inductive":
                    # Enhanced processing for inductive generalization
                    return spike_patterns
                elif self.current_mode == "abductive":
                    # Enhanced processing for abductive hypothesis generation
                    return spike_patterns
                else:
                    # Default processing
                    return spike_patterns
            except Exception as e:
                print(f"[ReasoningSNN] Error processing text with bidirectional processor: {e}")
                # Fall back to legacy encoding
        
        # Legacy encoding if bidirectional processor is not available or error occurs
        # Convert text to activation pattern using older methods
        activation = self._encode_reasoning_text(text_input)
        # Simulate spiking to get patterns
        spike_patterns = self.simulate_spiking(activation, timesteps=timesteps)
        
        return spike_patterns
    
    def generate_text_output(self, spike_patterns, max_length=100):
        """
        Generate text output from reasoning spike patterns using the standardized bidirectional processor.
        
        Args:
            spike_patterns: Spike patterns from reasoning process
            max_length: Maximum length of generated text
            
        Returns:
            Generated text from reasoning process
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
                print(f"[ReasoningSNN] Error generating text with bidirectional processor: {e}")
                # Fall back to legacy approach
        
        # Legacy approach for text generation if no bidirectional processor
        # For simplicity, generate text based on reasoning mode
        if self.current_mode == "deductive":
            return "Deductive reasoning conclusion: Analysis indicates logical validity."
        elif self.current_mode == "inductive":
            return "Inductive generalization: Pattern suggests general principle."
        elif self.current_mode == "abductive":
            return "Abductive hypothesis: Most plausible explanation for the observation."
        else:
            return "Reasoning result: Analysis complete."
    
    def _encode_reasoning_text(self, text_input):
        """
        Legacy method to encode text for reasoning (only used as fallback).
        
        Args:
            text_input: Text to encode
            
        Returns:
            Neural activation pattern
        """
        # Create blank activation
        activation = np.zeros(self.neuron_count)
        
        # Simple encoding based on words
        words = text_input.lower().split()
        
        # Get input layer neurons
        input_neurons = []
        if 'sensory' in self.regions:
            input_neurons.extend(self.regions['sensory']['neurons'])
        elif 'higher_cognition' in self.regions:
            input_neurons.extend(self.regions['higher_cognition']['neurons'])
        
        # Encode each word
        if input_neurons:
            neurons_per_word = max(1, len(input_neurons) // (len(words) + 1))
            for i, word in enumerate(words):
                start_idx = (i * neurons_per_word) % len(input_neurons)
                end_idx = min(start_idx + neurons_per_word, len(input_neurons))
                
                # Set activation for these neurons
                for j in range(start_idx, end_idx):
                    neuron_idx = input_neurons[j]
                    # Add slight position-based decay for temporal effect
                    temporal_factor = 1.0 - (0.02 * i)
                    activation[neuron_idx] = 0.8 * temporal_factor
        
        return activation