# snn_interfaces.py
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import numpy as np
import json
import time

class DataType(Enum):
    """Enum defining the types of data that can be communicated between components"""
    ACTIVATION_PATTERN = "activation_pattern"  # Raw neuron activation
    SPIKE_PATTERN = "spike_pattern"            # Sequence of spikes
    CONCEPT_ACTIVATIONS = "concept_activations"  # Activated concepts
    SYMBOLIC_FACTS = "symbolic_facts"          # Symbolic knowledge representation
    HYPOTHESIS = "hypothesis"                  # Generated hypothesis
    EMOTIONAL_STATE = "emotional_state"        # Affective evaluation
    MEMORY_RETRIEVAL = "memory_retrieval"      # Retrieved memory
    DECISION_OUTPUT = "decision_output"        # Decision result
    CONTROL_SIGNAL = "control_signal"          # Metacognitive control
    SYSTEM_STATE = "system_state"              # Overall system state
    RAW_TEXT = "raw_text"                      # Plain text input/output

class CommunicationPacket:
    """Standard packet format for inter-component communication"""
    
    def __init__(self, 
                 data_type: DataType,
                 content: Any,
                 source: str,
                 destination: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize a communication packet
        
        Args:
            data_type: Type of data being communicated
            content: The actual data payload
            source: Component that created this packet
            destination: Target component (None for broadcast)
            metadata: Additional information about the data
        """
        self.data_type = data_type
        self.content = content
        self.source = source
        self.destination = destination
        self.metadata = metadata or {}
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict:
        """Convert packet to dictionary representation"""
        result = {
            "data_type": self.data_type.value,
            "source": self.source,
            "destination": self.destination,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
        
        # Convert content based on data type
        if self.data_type == DataType.ACTIVATION_PATTERN:
            # Convert numpy array to list
            result["content"] = self.content.tolist() if isinstance(self.content, np.ndarray) else self.content
        elif self.data_type == DataType.SPIKE_PATTERN:
            # Convert spike pattern to compact representation
            result["content"] = [[int(n), float(s)] for n, s in self.content]
        else:
            # For other types, just include directly
            result["content"] = self.content
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CommunicationPacket':
        """Create packet from dictionary representation"""
        # Convert data type string to enum
        data_type = DataType(data["data_type"])
        
        # Handle content based on data type
        content = data["content"]
        if data_type == DataType.ACTIVATION_PATTERN and isinstance(content, list):
            # Convert list back to numpy array
            content = np.array(content)
        
        return cls(
            data_type=data_type,
            content=content,
            source=data["source"],
            destination=data.get("destination"),
            metadata=data.get("metadata", {})
        )

class SNNComponent:
    """Interface defining standard methods for SNN components"""
    
    def __init__(self, name: str):
        """Initialize component with a name"""
        self.name = name
        
    def process(self, packet: CommunicationPacket) -> Optional[CommunicationPacket]:
        """
        Process an incoming communication packet
        
        Args:
            packet: The incoming packet to process
            
        Returns:
            Response packet or None if no response
        """
        raise NotImplementedError("SNNComponent subclasses must implement process()")
    
    def get_state(self) -> Dict:
        """
        Get the current state of this component
        
        Returns:
            Dictionary representing component state
        """
        raise NotImplementedError("SNNComponent subclasses must implement get_state()")
    
    def receive_control_signals(self, control_signals: Dict) -> bool:
        """
        Receive control signals from metacognitive component
        
        Args:
            control_signals: Dictionary of control parameters
            
        Returns:
            Success status
        """
        raise NotImplementedError("SNNComponent subclasses must implement receive_control_signals()")

class SNNAdapter:
    """
    Adapter class to make existing SNN classes compatible with the standard interface
    
    This follows the Adapter pattern to allow existing SNN implementations to work with
    the new communication protocol without modifying their code.
    """
    
    def __init__(self, snn_instance, name: str):
        """
        Initialize with an existing SNN instance
        
        Args:
            snn_instance: Instance of an SNN class
            name: Name for this component
        """
        self.snn = snn_instance
        self.name = name
    
    def process(self, packet: CommunicationPacket) -> Optional[CommunicationPacket]:
        """Process an incoming packet by adapting to the underlying SNN API"""
        # Convert packet to appropriate input format
        if packet.data_type == DataType.RAW_TEXT:
            # Text input
            if hasattr(self.snn, 'process_input'):
                result = self.snn.process_input(packet.content)
            else:
                result = None
                
        elif packet.data_type == DataType.ACTIVATION_PATTERN:
            # Activation pattern
            if hasattr(self.snn, 'simulate_spiking'):
                spike_patterns = self.snn.simulate_spiking(packet.content)
                result = {
                    'spike_patterns': spike_patterns,
                    'phi': getattr(self.snn, 'phi', 0.0)
                }
            else:
                result = None
                
        elif packet.data_type == DataType.CONCEPT_ACTIVATIONS:
            # Concept activations - find most appropriate method
            if hasattr(self.snn, 'process_concepts'):
                result = self.snn.process_concepts(packet.content)
            elif hasattr(self.snn, 'process_input'):
                result = self.snn.process_input(packet.content)
            else:
                result = None
                
        elif packet.data_type == DataType.CONTROL_SIGNAL:
            # Control signals
            if hasattr(self.snn, 'receive_control_signals'):
                success = self.snn.receive_control_signals(packet.content)
                result = {'success': success}
            else:
                result = {'success': False}
                
        elif packet.data_type == DataType.SYSTEM_STATE:
            # System state for metacognitive monitoring
            if hasattr(self.snn, 'monitor_system_state'):
                result = self.snn.monitor_system_state(packet.content)
            else:
                result = None
        else:
            # Default handling for other data types
            if hasattr(self.snn, 'process'):
                result = self.snn.process(packet.content)
            elif hasattr(self.snn, 'process_input'):
                result = self.snn.process_input(packet.content)
            else:
                result = None
                
        # Create response packet if there was a result
        if result is not None:
            # Determine response data type
            response_type = self._determine_response_type(result)
            
            return CommunicationPacket(
                data_type=response_type,
                content=result,
                source=self.name,
                destination=packet.source if packet.source != self.name else None,
                metadata={
                    'in_response_to': packet.data_type.value,
                    'processing_time': time.time() - packet.timestamp
                }
            )
        
        return None
    
    def _determine_response_type(self, result):
        """Determine the appropriate data type for the response based on content"""
        if isinstance(result, dict):
            # Check for common keys in different response types
            if 'spike_patterns' in result:
                return DataType.SPIKE_PATTERN
            elif 'active_concepts' in result:
                return DataType.CONCEPT_ACTIVATIONS
            elif 'hypotheses' in result:
                return DataType.HYPOTHESIS
            elif 'emotions' in result or 'valence' in result:
                return DataType.EMOTIONAL_STATE
            elif 'retrieved_memories' in result:
                return DataType.MEMORY_RETRIEVAL
            elif 'decision_type' in result:
                return DataType.DECISION_OUTPUT
            elif 'success' in result and len(result) == 1:
                return DataType.CONTROL_SIGNAL
        
        # Default to system state for other dict results
        if isinstance(result, dict):
            return DataType.SYSTEM_STATE
            
        # Default for numpy arrays
        if isinstance(result, np.ndarray):
            return DataType.ACTIVATION_PATTERN
            
        # Default for text
        if isinstance(result, str):
            return DataType.RAW_TEXT
            
        # Generic fallback
        return DataType.SYSTEM_STATE
    
    def get_state(self) -> Dict:
        """Get the state of the underlying SNN"""
        if hasattr(self.snn, 'get_state'):
            return self.snn.get_state()
        
        # Fallback - construct basic state from attributes
        state = {
            'name': self.name,
            'type': self.snn.__class__.__name__
        }
        
        # Add commonly used attributes if available
        for attr in ['phi', 'integration', 'differentiation', 'membrane_potentials',
                    'active_neurons_cache', 'regions']:
            if hasattr(self.snn, attr):
                value = getattr(self.snn, attr)
                # Convert numpy arrays to lists for serialization
                if isinstance(value, np.ndarray):
                    state[attr] = value.tolist()
                else:
                    state[attr] = value
        
        return state
    
    def receive_control_signals(self, control_signals: Dict) -> bool:
        """Pass control signals to the underlying SNN"""
        if hasattr(self.snn, 'receive_control_signals'):
            return self.snn.receive_control_signals(control_signals)
        
        # Fallback - try to set attributes directly
        success = False
        for param, value in control_signals.items():
            if hasattr(self.snn, param):
                try:
                    setattr(self.snn, param, value)
                    success = True
                except:
                    pass
        
        return success