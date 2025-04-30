from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import time

@dataclass
class CognitiveState:
    """Standardized format for cognitive state interchange"""
    timestamp: float
    component_id: str
    activation_level: float
    confidence: float
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ProcessingStep:
    """Standardized format for processing steps"""
    step_id: str
    timestamp: float
    component: str
    operation: str
    input_state: Optional[CognitiveState]
    output_state: Optional[CognitiveState]
    metadata: Dict[str, Any]

class InterchangeManager:
    """Manages standardized data interchange between components"""
    
    def __init__(self):
        self.current_states: Dict[str, CognitiveState] = {}
        self.processing_history: List[ProcessingStep] = []
    
    def update_component_state(self, component_id: str, state: CognitiveState) -> None:
        """Update the state of a component"""
        self.current_states[component_id] = state
        
    def record_processing_step(self, step: ProcessingStep) -> None:
        """Record a processing step"""
        self.processing_history.append(step)
        
    def get_component_state(self, component_id: str) -> Optional[CognitiveState]:
        """Get the current state of a component"""
        return self.current_states.get(component_id)
        
    def get_recent_history(self, limit: int = 10) -> List[ProcessingStep]:
        """Get recent processing history"""
        return self.processing_history[-limit:]
