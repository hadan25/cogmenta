import numpy as np
import time
from collections import deque

class ConsciousnessField:
    """Implements a field model of consciousness based on GWT and IIT"""
    
    def __init__(self, field_size=100, decay_rate=0.95):
        self.field_size = field_size
        self.decay_rate = decay_rate
        self.field = np.zeros(field_size)
        self.activation_history = deque(maxlen=1000)
        self.oscillation_phase = 0.0
        self.base_frequency = 40  # 40Hz gamma oscillations
        
        # Track information flow
        self.information_streams = {
            'perceptual': np.zeros(field_size),
            'conceptual': np.zeros(field_size),
            'memory': np.zeros(field_size)
        }
        
    def update(self, dt):
        """Update consciousness field state"""
        # Update oscillation phase
        self.oscillation_phase += 2 * np.pi * self.base_frequency * dt
        self.oscillation_phase %= 2 * np.pi
        
        # Apply oscillatory modulation
        oscillation = 0.5 * (1 + np.sin(self.oscillation_phase))
        
        # Update field with decay
        self.field *= self.decay_rate * oscillation
        
        # Record current state
        self.activation_history.append({
            'timestamp': time.time(),
            'mean_activation': np.mean(self.field),
            'max_activation': np.max(self.field),
            'oscillation_phase': self.oscillation_phase
        })
        
    def broadcast(self, content, location, strength):
        """Broadcast content into consciousness field"""
        # Create gaussian activation around location
        x = np.linspace(0, self.field_size-1, self.field_size)
        activation = strength * np.exp(-(x - location)**2 / (2 * 10**2))
        
        # Add to field
        self.field = np.maximum(self.field, activation)
        
    def get_conscious_content(self, threshold=0.5):
        """Get currently conscious content"""
        active_regions = np.where(self.field > threshold)[0]
        return {
            'active_regions': active_regions,
            'mean_activation': np.mean(self.field[active_regions]) if len(active_regions) > 0 else 0,
            'coverage': len(active_regions) / self.field_size
        }
