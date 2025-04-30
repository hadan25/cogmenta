# cogmenta_core/models/snn/spiking_core.py
import random
import math
import time
from collections import defaultdict, deque

class SpikingCore:
    """
    Enhanced Spiking Neural Network core that incorporates principles from
    Integrated Information Theory (IIT) and Recurrent Processing Theory.
    
    This implementation focuses on emergence of consciousness-like properties
    through interconnection of neural assemblies.
    """
    
    def __init__(self):
        # Neural regions (assemblies of neurons)
        self.regions = {
            'perception': {
                'neurons': 100,
                'activation': 0.0,
                'connections': ['working_memory', 'conceptual'],
                'recurrent': True
            },
            'working_memory': {
                'neurons': 50,
                'activation': 0.0,
                'connections': ['perception', 'conceptual', 'prediction'],
                'recurrent': True
            },
            'conceptual': {
                'neurons': 80,
                'activation': 0.0,
                'connections': ['working_memory', 'prediction', 'metacognition'],
                'recurrent': True
            },
            'prediction': {
                'neurons': 60,
                'activation': 0.0,
                'connections': ['working_memory', 'conceptual', 'action'],
                'recurrent': True
            },
            'metacognition': {
                'neurons': 40,
                'activation': 0.0,
                'connections': ['conceptual', 'prediction', 'action'],
                'recurrent': True
            },
            'action': {
                'neurons': 30,
                'activation': 0.0,
                'connections': ['prediction', 'metacognition'],
                'recurrent': False
            }
        }
        
        # Track spikes over time
        self.spike_history = defaultdict(lambda: deque(maxlen=100))
        
        # Recurrent activation tracking
        self.recurrent_activations = {}
        
        # Integration metrics (IIT-inspired)
        self.phi = 0.0              # Overall integration measure
        self.differentiation = 0.0  # Information differentiation
        self.integration = 0.0      # Information integration
        
        # Seed concepts (for abductive reasoning)
        self.seed_concepts = [
            'likes', 'trusts', 'knows', 'fears', 'helps',
            'hates', 'avoids', 'requires', 'causes', 'enables'
        ]
        
        # Store emergent patterns
        self.emergent_patterns = []
        
        print("[SNN] Enhanced Spiking Neural Network initialized with IIT-inspired architecture")
    
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
        
        # Activation propagation
        for step in range(max_steps):
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
        
    def process_input(self, text):
        """
        Process input text through the spiking neural network
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results including activations and IIT metrics
        """
        # Initial text processing (simplistic)
        words = text.lower().split()
        
        # Check for emotion/concept words to determine initial activation regions
        emotion_words = ['like', 'hate', 'fear', 'trust', 'happy', 'sad', 'angry']
        concept_words = ['knows', 'believes', 'thinks', 'understands', 'remembers']
        
        emotion_count = sum(1 for w in words if w in emotion_words)
        concept_count = sum(1 for w in words if w in concept_words)
        
        # Determine primary activation region based on input content
        if emotion_count > concept_count:
            initial_region = 'conceptual'
            activation_strength = 0.7 + (min(emotion_count, 3) * 0.1)
        else:
            initial_region = 'perception'
            activation_strength = 0.6 + (min(concept_count, 4) * 0.1)
            
        # Propagate activation through the network
        activations = self._propagate_activation(initial_region, activation_strength)
        
        # Identify emergent patterns
        emergent_pattern = self._identify_emergent_pattern()
        if emergent_pattern:
            self.emergent_patterns.append(emergent_pattern)
            print(f"[SNN] Identified emergent pattern: {emergent_pattern['name']}")
        
        # Return processing results
        return {
            'activations': activations,
            'phi': self.phi,
            'initial_region': initial_region,
            'activation_strength': activation_strength,
            'emergent_pattern': emergent_pattern
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
    
    def abductive_reasoning(self, text):
        """
        Generate hypotheses through abductive reasoning
        
        Args:
            text: Input text to reason about
            
        Returns:
            List of hypotheses
        """
        # Process input to get network into appropriate state
        result = self.process_input(text)
        
        # Extract potential subjects and objects from text
        words = text.lower().split()
        potential_entities = [w for w in words if len(w) > 3 and w.isalpha()]
        
        # Generate hypotheses based on network activations
        hypotheses = []
        
        # Number of hypotheses based on activation and phi
        if result['phi'] < 0.3:
            # Low integration - fewer, more random hypotheses
            num_hypotheses = random.randint(1, 2)
        elif result['phi'] < 0.6:
            # Medium integration - moderate number of hypotheses
            num_hypotheses = random.randint(2, 4)
        else:
            # High integration - more hypotheses with better quality
            num_hypotheses = random.randint(3, 5)
        
        # Generate hypotheses
        for _ in range(num_hypotheses):
            if len(potential_entities) >= 2:
                # Pick random subject and object
                subj, obj = random.sample(potential_entities, 2)
                
                # Pick predicate based on activation patterns
                if self.regions['conceptual']['activation'] > 0.7:
                    # Higher conceptual activation - use more abstract predicates
                    possible_predicates = ['understands', 'causes', 'enables', 'implies']
                elif self.regions['perception']['activation'] > 0.7:
                    # Higher perception activation - use more concrete predicates
                    possible_predicates = ['sees', 'touches', 'visits', 'meets']
                else:
                    # Default - use seed concepts
                    possible_predicates = self.seed_concepts
                
                # Select predicate with some bias toward network state
                pred = random.choice(possible_predicates)
                
                # Form hypothesis
                hypothesis = f"{pred}({subj}, {obj})"
                hypotheses.append(hypothesis)
        
        # Add metacognitive hypothesis if high metacognition activation
        if self.regions['metacognition']['activation'] > 0.7 and potential_entities:
            subj = random.choice(potential_entities)
            hypothesis = f"self_aware({subj})"
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def get_network_state(self):
        """Return current network state for visualization/debugging"""
        return {
            'regions': self.regions,
            'phi': self.phi,
            'differentiation': self.differentiation,
            'integration': self.integration,
            'recurrent_steps': len(self.recurrent_activations),
            'emergent_patterns': self.emergent_patterns[-5:] if self.emergent_patterns else []
        }