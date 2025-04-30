# cogmenta_core/models/snn/network_topology.py
import numpy as np
import random
import time
from collections import defaultdict

class NetworkTopology:
    """
    Advanced neural network topology that organizes neurons into 
    functional assemblies and regions with biologically inspired connectivity.
    
    This implements more sophisticated neural organization principles to support
    emergence of consciousness-like properties.
    """
    
    def __init__(self, neuron_count=256):
        self.neuron_count = neuron_count
        
        # Neural assemblies (groups of neurons that function together)
        self.assemblies = {}
        
        # Connection patterns between assemblies
        self.assembly_connections = {}
        
        # Map concepts to neural assemblies
        self.concept_mappings = {}
        
        # Neural regions (larger functional areas)
        # Updated neuron count to ensure adequate distribution
        neurons_per_region = self._calculate_neurons_per_region(neuron_count)
        
        self.regions = {
            'sensory': {'neurons': [], 'activation': 0.0},
            'memory': {'neurons': [], 'activation': 0.0},
            'language': {'neurons': [], 'activation': 0.0},
            'executive': {'neurons': [], 'activation': 0.0},
            'emotional': {'neurons': [], 'activation': 0.0},
            'motor': {'neurons': [], 'activation': 0.0},
            'metacognitive': {'neurons': [], 'activation': 0.0}
        }
        
        # Region connectivity (roughly based on brain connectivity)
        self.region_connectivity = {
            'sensory': ['memory', 'language', 'executive'],
            'memory': ['sensory', 'language', 'executive', 'metacognitive'],
            'language': ['sensory', 'memory', 'executive', 'emotional'],
            'executive': ['sensory', 'memory', 'language', 'emotional', 'motor', 'metacognitive'],
            'emotional': ['language', 'executive', 'motor'],
            'motor': ['executive', 'emotional'],
            'metacognitive': ['memory', 'executive']
        }
        
        # Initialize the topology
        self._initialize_topology()
        
        print(f"[Network] Initialized neural topology with {neuron_count} neurons")
    
    def _calculate_neurons_per_region(self, total_neurons):
        """Calculate the number of neurons per region based on total count"""
        # Ensure we have enough neurons (at least 210 for all assemblies)
        min_required = 210  # Based on assembly sizes defined in _create_default_assemblies
        
        if total_neurons < min_required:
            print(f"[Network] Warning: Increasing neuron count from {total_neurons} to {min_required}")
            total_neurons = min_required
        
        # Distribution ratios
        ratios = {
            'sensory': 0.15,
            'memory': 0.15,
            'language': 0.15,
            'executive': 0.15,
            'emotional': 0.10,
            'motor': 0.10,
            'metacognitive': 0.20
        }
        
        # Calculate neurons per region
        neurons_per_region = {}
        allocated = 0
        
        for region, ratio in ratios.items():
            count = max(30, int(total_neurons * ratio))  # At least 30 neurons per region
            neurons_per_region[region] = count
            allocated += count
            
        # Adjust if we've allocated too many
        if allocated > total_neurons:
            print(f"[Network] Warning: Adjusted neuron count from {total_neurons} to {allocated}")
            self.neuron_count = allocated
            
        return neurons_per_region
    
    def _initialize_topology(self):
        """Initialize the neural network topology with assemblies and regions"""
        # Allocate neurons to regions
        neurons_per_region = self._calculate_neurons_per_region(self.neuron_count)
        
        # Assign neurons to regions
        start_idx = 0
        for region_name, count in neurons_per_region.items():
            self.regions[region_name]['neurons'] = list(range(start_idx, start_idx + count))
            start_idx += count
        
        # Create basic assemblies in each region
        self._create_default_assemblies()
        
        # Create default concept mappings
        self._initialize_concept_mappings()
    
    def _create_default_assemblies(self):
        """Create default neural assemblies in each region"""
        # Sensory assemblies - allocate proper sizes ensuring we don't exceed region capacity
        sensory_neurons = len(self.regions['sensory']['neurons'])
        visual_size = min(20, sensory_neurons * 3 // 4)
        auditory_size = min(20, sensory_neurons - visual_size)
        
        self.create_neural_assembly('visual', visual_size, region='sensory', concepts=['see', 'look', 'watch'])
        self.create_neural_assembly('auditory', auditory_size, region='sensory', concepts=['hear', 'listen'])
        
        # Memory assemblies
        memory_neurons = len(self.regions['memory']['neurons'])
        episodic_size = min(15, memory_neurons // 2)
        semantic_size = min(15, memory_neurons - episodic_size)
        
        self.create_neural_assembly('episodic', episodic_size, region='memory', concepts=['remember', 'recall'])
        self.create_neural_assembly('semantic', semantic_size, region='memory', concepts=['know', 'understand', 'mean'])
        
        # Language assemblies
        language_neurons = len(self.regions['language']['neurons'])
        grammar_size = min(12, language_neurons // 3)
        vocab_size = min(12, language_neurons // 3)
        concept_size = min(15, language_neurons - grammar_size - vocab_size)
        
        self.create_neural_assembly('grammar', grammar_size, region='language')
        self.create_neural_assembly('vocabulary', vocab_size, region='language')
        self.create_neural_assembly('concepts', concept_size, region='language', concepts=['concept', 'idea'])
        
        # Executive assemblies
        exec_neurons = len(self.regions['executive']['neurons'])
        planning_size = min(15, exec_neurons // 2)
        decision_size = min(15, exec_neurons - planning_size)
        
        self.create_neural_assembly('planning', planning_size, region='executive', concepts=['plan', 'intend'])
        self.create_neural_assembly('decision', decision_size, region='executive', concepts=['decide', 'choose'])
        
        # Emotional assemblies
        emot_neurons = len(self.regions['emotional']['neurons'])
        trust_size = min(10, emot_neurons // 3)
        fear_size = min(10, emot_neurons // 3) 
        like_size = min(10, emot_neurons - trust_size - fear_size)
        
        self.create_neural_assembly('trust', trust_size, region='emotional', concepts=['trust', 'believe'])
        self.create_neural_assembly('fear', fear_size, region='emotional', concepts=['fear', 'afraid'])
        self.create_neural_assembly('like', like_size, region='emotional', concepts=['like', 'love'])
        
        # Motor assemblies
        motor_neurons = len(self.regions['motor']['neurons'])
        action_size = min(15, motor_neurons // 2)
        speech_size = min(15, motor_neurons - action_size)
        
        self.create_neural_assembly('action', action_size, region='motor', concepts=['do', 'act', 'move'])
        self.create_neural_assembly('speech', speech_size, region='motor', concepts=['say', 'tell', 'speak'])
        
        # Metacognitive assemblies
        meta_neurons = len(self.regions['metacognitive']['neurons'])
        self_model_size = min(20, meta_neurons // 3)
        theory_size = min(15, meta_neurons // 3)
        reflection_size = min(15, meta_neurons - self_model_size - theory_size)
        
        self.create_neural_assembly('self_model', self_model_size, region='metacognitive', concepts=['self', 'me', 'i'])
        self.create_neural_assembly('theory_of_mind', theory_size, region='metacognitive', concepts=['think', 'believe'])
        self.create_neural_assembly('reflection', reflection_size, region='metacognitive', concepts=['reflect', 'aware'])
        
        # Create connections between assemblies based on cognitive relationships
        connected_pairs = [
            ('visual', 'episodic'),
            ('auditory', 'episodic'),
            ('episodic', 'semantic'),
            ('semantic', 'vocabulary'),
            ('vocabulary', 'grammar'),
            ('concepts', 'semantic'),
            ('planning', 'decision'),
            ('decision', 'action'),
            ('trust', 'decision'),
            ('fear', 'decision'),
            ('like', 'decision'),
            ('speech', 'grammar'),
            ('self_model', 'reflection'),
            ('theory_of_mind', 'reflection'),
            ('reflection', 'decision')
        ]
        
        # Add connections with strengths
        for source, target in connected_pairs:
            if source in self.assemblies and target in self.assemblies:
                self.connect_assemblies(source, target, strength=0.6)
            else:
                print(f"[Network] Warning: Cannot connect {source} -> {target}, one or both missing")
    
    def _initialize_concept_mappings(self):
        """Initialize mappings from concepts to neural assemblies"""
        relation_concepts = {
            'trust': 'trust',
            'fear': 'fear',
            'like': 'like',
            'know': 'semantic',
            'understand': 'semantic',
            'remember': 'episodic',
            'see': 'visual',
            'hear': 'auditory',
            'think': 'theory_of_mind',
            'believe': 'theory_of_mind',
            'self': 'self_model',
            'aware': 'reflection',
            'reflect': 'reflection'
        }
        
        for concept, assembly_name in relation_concepts.items():
            if assembly_name in self.assemblies:
                self.concept_mappings[concept] = self.assemblies[assembly_name]['neurons']
    
    def create_neural_assembly(self, name, size, region=None, neuron_ids=None, concepts=None):
        """
        Create a neural assembly (group of neurons that function together)
        
        Args:
            name: Name of the assembly
            size: Number of neurons in the assembly
            region: Region to create the assembly in
            neuron_ids: Specific neuron IDs to use (if None, will be assigned)
            concepts: Concepts associated with this assembly
            
        Returns:
            The created assembly
        """
        # Handle zero size assemblies
        if size <= 0:
            print(f"[Network] Warning: Cannot create assembly '{name}' with size {size}")
            return None
            
        if name in self.assemblies:
            print(f"[Network] Warning: Assembly {name} already exists")
            return self.assemblies[name]
        
        # Determine neurons for this assembly
        if neuron_ids is not None:
            # Use provided neuron IDs
            neurons = neuron_ids
        elif region is not None and region in self.regions:
            # Allocate from region's neurons
            available_neurons = self.regions[region]['neurons']
            
            # Check if we have already allocated neurons from this region
            allocated = []
            for a in self.assemblies.values():
                if a.get('region') == region:
                    allocated.extend(a['neurons'])
            
            # Find available neurons in this region
            available = [n for n in available_neurons if n not in allocated]
            
            if len(available) < size:
                print(f"[Network] Warning: Not enough neurons in region {region}, using available {len(available)}")
                size = len(available)
                
            if size == 0:
                print(f"[Network] Warning: Cannot create assembly '{name}' - no neurons available in region {region}")
                return None
                
            neurons = available[:size]  # Take what we can
        else:
            # Random allocation
            neurons = random.sample(range(self.neuron_count), min(size, self.neuron_count))
        
        # Create the assembly
        assembly = {
            'name': name,
            'neurons': neurons,
            'size': len(neurons),
            'region': region,
            'activation': 0.0,
            'concepts': concepts or [],
            'created': time.time()
        }
        
        self.assemblies[name] = assembly
        print(f"[Network] Created neural assembly: {name} with {len(neurons)} neurons in region {region}")
        
        # Add concept mappings if provided
        if concepts:
            for concept in concepts:
                self.concept_mappings[concept] = neurons
        
        return assembly
    
    def connect_assemblies(self, source_name, target_name, strength=0.5, bidirectional=True):
        """
        Create a connection between two neural assemblies
        
        Args:
            source_name: Source assembly name
            target_name: Target assembly name
            strength: Connection strength (0-1)
            bidirectional: Whether the connection is bidirectional
            
        Returns:
            Success status
        """
        if source_name not in self.assemblies or target_name not in self.assemblies:
            print(f"[Network] Cannot connect: assembly not found")
            return False
        
        # Initialize connection dictionary if needed
        if source_name not in self.assembly_connections:
            self.assembly_connections[source_name] = {}
        
        # Add the connection
        self.assembly_connections[source_name][target_name] = strength
        
        # Add reciprocal connection if bidirectional
        if bidirectional:
            if target_name not in self.assembly_connections:
                self.assembly_connections[target_name] = {}
            self.assembly_connections[target_name][source_name] = strength
        
        print(f"[Network] Connected {source_name} -> {target_name} (strength={strength})")
        return True
    
    def activate_assembly(self, name, activation=1.0):
        """
        Activate a neural assembly
        
        Args:
            name: Assembly name
            activation: Activation level (0-1)
            
        Returns:
            Success status
        """
        if name not in self.assemblies:
            return False
        
        # Set activation level
        self.assemblies[name]['activation'] = activation
        
        # Update region activation based on active assemblies
        if self.assemblies[name]['region']:
            region_name = self.assemblies[name]['region']
            region_assemblies = [a for a in self.assemblies.values() if a['region'] == region_name]
            if region_assemblies:
                region_activation = sum(a['activation'] for a in region_assemblies) / len(region_assemblies)
                self.regions[region_name]['activation'] = region_activation
        
        return True
    
    def propagate_activation(self, steps=3):
        """
        Propagate activation through connected assemblies
        
        Args:
            steps: Number of propagation steps
            
        Returns:
            Dictionary of assembly activations
        """
        for _ in range(steps):
            # Store current activations
            current_activations = {name: assembly['activation'] for name, assembly in self.assemblies.items()}
            
            # Calculate new activations based on connections
            new_activations = current_activations.copy()
            
            for source, connections in self.assembly_connections.items():
                for target, strength in connections.items():
                    # Activation flows from source to target based on connection strength
                    if source in current_activations and target in new_activations and current_activations[source] > 0.1:
                        # Only propagate significant activation
                        incoming_activation = current_activations[source] * strength
                        new_activations[target] = min(1.0, new_activations[target] + (incoming_activation * 0.3))
            
            # Apply decay
            for name in new_activations:
                new_activations[name] *= 0.9  # 10% decay per step
            
            # Update activations
            for name, activation in new_activations.items():
                self.assemblies[name]['activation'] = activation
                
            # Update region activations
            for region_name in self.regions:
                region_assemblies = [a for a in self.assemblies.values() if a['region'] == region_name]
                if region_assemblies:
                    region_activation = sum(a['activation'] for a in region_assemblies) / len(region_assemblies)
                    self.regions[region_name]['activation'] = region_activation
        
        # Return current activations
        return {name: assembly['activation'] for name, assembly in self.assemblies.items()}
    
    def activate_concept(self, concept, strength=0.8):
        """
        Activate neurons associated with a concept
        
        Args:
            concept: Concept to activate
            strength: Activation strength
            
        Returns:
            Whether the concept was found and activated
        """
        # Check if we have a mapping for this concept
        if concept in self.concept_mappings:
            # Find assemblies containing these neurons
            concept_neurons = set(self.concept_mappings[concept])
            
            activated = False
            for name, assembly in self.assemblies.items():
                # Check if this assembly contains any of the concept neurons
                if any(n in concept_neurons for n in assembly['neurons']):
                    # Activate this assembly
                    overlap_ratio = sum(1 for n in assembly['neurons'] if n in concept_neurons) / len(assembly['neurons'])
                    activation_strength = strength * overlap_ratio
                    self.activate_assembly(name, activation_strength)
                    activated = True
            
            return activated
        
        return False
    
    def get_active_assemblies(self, threshold=0.3):
        """
        Get currently active neural assemblies
        
        Args:
            threshold: Minimum activation level
            
        Returns:
            List of (assembly_name, activation) tuples
        """
        active = [(name, assembly['activation']) 
                for name, assembly in self.assemblies.items()
                if assembly['activation'] >= threshold]
        
        return sorted(active, key=lambda x: x[1], reverse=True)
    
    def get_active_regions(self, threshold=0.3):
        """
        Get currently active brain regions
        
        Args:
            threshold: Minimum activation level
            
        Returns:
            List of (region_name, activation) tuples
        """
        active = [(name, region['activation']) 
                for name, region in self.regions.items()
                if region['activation'] >= threshold]
        
        return sorted(active, key=lambda x: x[1], reverse=True)
    
    def get_topology_state(self):
        """
        Get the current state of the neural topology
        
        Returns:
            Dictionary with the current topology state
        """
        return {
            'assemblies': {name: {'activation': a['activation'], 'region': a['region'], 'size': a['size']}
                        for name, a in self.assemblies.items()},
            'regions': {name: {'activation': r['activation'], 'neurons': len(r['neurons'])}
                      for name, r in self.regions.items()},
            'active_assemblies': self.get_active_assemblies(),
            'active_regions': self.get_active_regions()
        }