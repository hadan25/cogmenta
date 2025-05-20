# neural_orchestrator.py
import time
import numpy as np
from collections import defaultdict

class NeuralOrchestrator:
    """Coordinates information flow between specialized neural components"""
    
    def __init__(self, components=None):
        """
        Initialize with dictionary of components
        Example: {'perceptual': PerceptualSNN(), 'memory': MemorySNN(), ...}
        """
        self.components = components or {}
        self.pathways = self._setup_default_pathways()
        self.active_pathways = set()
        self.component_states = {}
        self.processing_history = []
        self.global_workspace = {
            'current_focus': None,
            'broadcast_strength': 0.0,
            'last_update': time.time(),
            'active_elements': []
        }
        
    def _setup_default_pathways(self):
        """Define biologically inspired communication pathways"""
        return {
            "perception_to_memory": {
                "sender": "perceptual", 
                "receiver": "memory",
                "threshold": 0.6,
                "transform_fn": self._default_transform
            },
            "memory_to_reasoning": {
                "sender": "memory", 
                "receiver": "reasoning",
                "threshold": 0.5,
                "transform_fn": self._default_transform
            },
            "reasoning_to_decision": {
                "sender": "reasoning", 
                "receiver": "decision",
                "threshold": 0.7,
                "transform_fn": self._default_transform
            },
            "perception_to_affective": {
                "sender": "perceptual", 
                "receiver": "affective",
                "threshold": 0.4,  # Lower threshold for emotional processing
                "transform_fn": self._default_transform
            },
            "affective_to_reasoning": {
                "sender": "affective", 
                "receiver": "reasoning",
                "threshold": 0.5,
                "transform_fn": self._affective_influence_transform
            },
            "memory_to_affective": {
                "sender": "memory", 
                "receiver": "affective",
                "threshold": 0.5,
                "transform_fn": self._default_transform
            },
            "metacognitive_monitor": {
                "sender": None,  # Special pathway for metacognition to monitor all components
                "receiver": "metacognitive",
                "threshold": 0.0,  # Always active
                "transform_fn": self._metacognitive_transform
            },
            "metacognitive_control": {
                "sender": "metacognitive",
                "receiver": None,  # Affects all components
                "threshold": 0.7,
                "transform_fn": self._metacognitive_control_transform
            }
        }
    
    def _default_transform(self, sender_output, pathway):
        """Default transformation between components"""
        # Simple passthrough with metadata
        return {
            "data": sender_output,
            "source": pathway["sender"],
            "timestamp": time.time()
        }
    
    def _affective_influence_transform(self, sender_output, pathway):
        """Transform affective output to influence reasoning"""
        # Extract emotional valence and arousal
        if isinstance(sender_output, dict) and "valence" in sender_output and "arousal" in sender_output:
            valence = sender_output.get("valence", 0)
            arousal = sender_output.get("arousal", 0)
            dominant_emotion = sender_output.get("dominant_emotion", None)
            
            # Create influence parameters
            return {
                "data": sender_output,
                "source": pathway["sender"],
                "timestamp": time.time(),
                "influence_params": {
                    "valence_bias": valence * 0.5,  # Scale valence influence
                    "confidence_modifier": arousal * 0.3,  # Arousal affects confidence
                    "emotion_context": dominant_emotion
                }
            }
        
        # Fallback to default transform
        return self._default_transform(sender_output, pathway)
    
    def _metacognitive_transform(self, sender_output, pathway):
        """Transform component states for metacognitive monitoring"""
        # Collect states from all components
        all_states = {name: comp.get_state() if hasattr(comp, 'get_state') else None 
                     for name, comp in self.components.items()}
        
        # Add integration metrics
        integration_metrics = {
            "global_workspace": self.global_workspace,
            "active_pathways": list(self.active_pathways),
            "timestamp": time.time()
        }
        
        return {
            "data": all_states,
            "integration": integration_metrics,
            "source": "orchestrator",
            "timestamp": time.time()
        }
    
    def _metacognitive_control_transform(self, sender_output, pathway):
        """Transform metacognitive output to control signals for other components"""
        if not isinstance(sender_output, dict) or "control_signals" not in sender_output:
            return None
            
        control_signals = sender_output["control_signals"]
        return {
            "data": control_signals,
            "source": pathway["sender"],
            "timestamp": time.time()
        }
    
    def add_component(self, name, component):
        """Add a component to the orchestrator"""
        self.components[name] = component
        return True
    
    def add_pathway(self, name, sender, receiver, threshold=0.5, transform_fn=None):
        """Add a new communication pathway"""
        if transform_fn is None:
            transform_fn = self._default_transform
            
        self.pathways[name] = {
            "sender": sender,
            "receiver": receiver,
            "threshold": threshold,
            "transform_fn": transform_fn
        }
        return True
    
    def _update_active_pathways(self):
        """Determine which pathways should be active based on component activations"""
        self.active_pathways = set()
        
        for name, pathway in self.pathways.items():
            sender = pathway["sender"]
            
            # Skip pathways without a sender (metacognitive monitoring)
            if sender is None:
                self.active_pathways.add(name)
                continue
                
            # Check if sender component is sufficiently activated
            if sender in self.component_states:
                # Extract activation level from component state
                activation = self._extract_activation(self.component_states[sender])
                
                if activation >= pathway["threshold"]:
                    self.active_pathways.add(name)
    
    def _extract_activation(self, component_state):
        """Extract activation level from component state"""
        if isinstance(component_state, dict):
            # Try common keys for activation
            for key in ["activation", "phi", "overall_activation", "average_activation"]:
                if key in component_state:
                    return component_state[key]
            
            # Check for region activations
            if "region_activations" in component_state:
                region_activations = component_state["region_activations"]
                if isinstance(region_activations, dict) and region_activations:
                    return sum(region_activations.values()) / len(region_activations)
        
        # Default activation level
        return 0.5
    
    def _process_pathway(self, pathway_name, input_data):
        """Process data through a specific pathway"""
        pathway = self.pathways[pathway_name]
        sender_name = pathway["sender"]
        receiver_name = pathway["receiver"]
        
        # Skip if sender or receiver not available
        if sender_name and sender_name not in self.components:
            return False
        if receiver_name and receiver_name not in self.components:
            return False
            
        # Get sender output (or use input data for initial pathways)
        sender_output = None
        if sender_name:
            sender_output = self.component_states.get(sender_name, None)
        else:
            # For pathways without a sender (e.g., metacognitive monitoring)
            sender_output = input_data
            
        # Apply transformation function
        transformed_data = pathway["transform_fn"](sender_output, pathway)
        
        if transformed_data is None:
            return False
            
        # Process through receiver component
        if receiver_name:
            receiver = self.components[receiver_name]
            
            # Call appropriate method based on receiver type
            if hasattr(receiver, 'process_input'):
                result = receiver.process_input(transformed_data)
            elif hasattr(receiver, 'process'):
                result = receiver.process(transformed_data)
            elif hasattr(receiver, 'monitor_system_state') and receiver_name == "metacognitive":
                result = receiver.monitor_system_state(transformed_data)
            else:
                result = None
                
            # Store result in component states
            if result is not None:
                self.component_states[receiver_name] = result
                return True
        elif pathway_name == "metacognitive_control":
            # Special case for metacognitive control (affects multiple components)
            self._apply_metacognitive_control(transformed_data)
            return True
            
        return False
    
    def _apply_metacognitive_control(self, control_data):
        """Apply metacognitive control signals to components"""
        if not isinstance(control_data, dict) or "data" not in control_data:
            return
            
        control_signals = control_data["data"]
        
        # Apply control signals to specific components
        for component_name, signals in control_signals.items():
            if component_name in self.components:
                component = self.components[component_name]
                
                # Apply signals based on component type
                if hasattr(component, 'receive_control_signals'):
                    component.receive_control_signals(signals)
                elif hasattr(component, 'set_parameters') and isinstance(signals, dict):
                    component.set_parameters(**signals)
    
    def _update_global_workspace(self):
        """Update the global workspace based on component activations"""
        # Find component with highest activation
        max_activation = 0.0
        max_component = None
        
        for name, state in self.component_states.items():
            activation = self._extract_activation(state)
            if activation > max_activation:
                max_activation = activation
                max_component = name
        
        # Update global workspace if sufficiently activated
        if max_activation > 0.7 and max_component:
            self.global_workspace['current_focus'] = max_component
            self.global_workspace['broadcast_strength'] = max_activation
            self.global_workspace['last_update'] = time.time()
            
            # Add relevant information to active elements
            if max_component in self.component_states:
                self._update_active_elements(max_component, self.component_states[max_component])
    
    def _update_active_elements(self, component_name, state):
        """Extract and store relevant information from component state"""
        elements = []
        
        # Extract from different component types
        if component_name == "perceptual" and isinstance(state, dict):
            # Extract detected features
            features = state.get("detected_features", {})
            for feature_name, activation in features.items():
                if activation > 0.7:
                    elements.append({"type": "feature", "name": feature_name, "activation": activation})
        
        elif component_name == "memory" and isinstance(state, dict):
            # Extract retrieved memories
            memories = state.get("retrieved_memories", [])
            for memory in memories[:3]:  # Limit to top 3
                elements.append({"type": "memory", "content": memory.get("content", ""), 
                                "relevance": memory.get("relevance", 0)})
        
        elif component_name == "reasoning" and isinstance(state, dict):
            # Extract hypotheses or conclusions
            hypotheses = state.get("hypotheses", [])
            for hypothesis in hypotheses[:3]:  # Limit to top 3
                elements.append({"type": "hypothesis", "relation": hypothesis.get("relation", ""),
                                "confidence": hypothesis.get("confidence", 0)})
        
        elif component_name == "affective" and isinstance(state, dict):
            # Extract emotional state
            emotions = state.get("emotions", {})
            for emotion, activation in emotions.items():
                if activation > 0.6:
                    elements.append({"type": "emotion", "name": emotion, "activation": activation})
        
        # Add new elements to global workspace
        self.global_workspace['active_elements'] = elements
    
    def process_input(self, input_data):
        """Process input through the complete neural assembly"""
        # Reset component states
        self.component_states = {}
        
        # Record processing start
        processing_record = {
            'input': input_data,
            'timestamp': time.time(),
            'component_states': {},
            'active_pathways': [],
            'global_workspace': {}
        }
        
        # Step 1: Process through perceptual component first
        if 'perceptual' in self.components:
            perceptual_result = self.components['perceptual'].process_input(input_data)
            self.component_states['perceptual'] = perceptual_result
        
        # Step 2: Determine active pathways based on activation levels
        self._update_active_pathways()
        
        # Step 3: Sequential processing through active pathways
        # Sort pathways to ensure proper processing order
        ordered_pathways = self._get_ordered_pathways()
        
        for pathway_name in ordered_pathways:
            if pathway_name in self.active_pathways:
                self._process_pathway(pathway_name, input_data)
                
        # Step 4: Update global workspace
        self._update_global_workspace()
        
        # Step 5: Metacognitive monitoring
        if "metacognitive" in self.components and "metacognitive_monitor" in self.active_pathways:
            self._process_pathway("metacognitive_monitor", input_data)
            
            # Apply metacognitive control if needed
            if "metacognitive_control" in self.active_pathways:
                self._process_pathway("metacognitive_control", input_data)
        
        # Record final component states
        for name, state in self.component_states.items():
            processing_record['component_states'][name] = state
            
        processing_record['active_pathways'] = list(self.active_pathways)
        processing_record['global_workspace'] = dict(self.global_workspace)
        
        # Add to processing history
        self.processing_history.append(processing_record)
        
        # Generate final output (typically from decision component or global workspace)
        return self._generate_output()
    
    def _get_ordered_pathways(self):
        """Get pathways in proper processing order"""
        # Define processing stages
        stages = [
            # Stage 1: Initial perception and affective processing
            ["perception_to_memory", "perception_to_affective"],
            
            # Stage 2: Memory processing and emotional influence
            ["memory_to_reasoning", "memory_to_affective", "affective_to_reasoning"],
            
            # Stage 3: Reasoning and decision making
            ["reasoning_to_decision"],
            
            # Stage 4: Metacognitive processes
            ["metacognitive_monitor", "metacognitive_control"]
        ]
        
        # Flatten ordered pathways
        ordered_pathways = []
        for stage in stages:
            for pathway in stage:
                if pathway in self.pathways:
                    ordered_pathways.append(pathway)
        
        # Add any remaining pathways not explicitly ordered
        for pathway in self.pathways:
            if pathway not in ordered_pathways:
                ordered_pathways.append(pathway)
                
        return ordered_pathways
    
    def _generate_output(self):
        """Generate final output from component states"""
        # Prioritize decision component if available
        if 'decision' in self.component_states:
            return self.component_states['decision']
            
        # Use reasoning component as fallback
        if 'reasoning' in self.component_states:
            return self.component_states['reasoning']
            
        # Use global workspace as last resort
        if self.global_workspace['active_elements']:
            return {
                'output_type': 'global_workspace',
                'focus': self.global_workspace['current_focus'],
                'elements': self.global_workspace['active_elements'],
                'integration_level': self.global_workspace['broadcast_strength']
            }
            
        # Default minimal output
        return {
            'output_type': 'minimal',
            'component_states': {name: self._extract_activation(state) 
                               for name, state in self.component_states.items()},
            'active_pathways': list(self.active_pathways)
        }
    
    def get_component_state(self, component_name):
        """Get the current state of a specific component"""
        return self.component_states.get(component_name, None)
    
    def get_system_state(self):
        """Get the current state of the entire system"""
        return {
            'component_states': self.component_states,
            'active_pathways': list(self.active_pathways),
            'global_workspace': dict(self.global_workspace)
        }