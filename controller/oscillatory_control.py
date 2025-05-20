import numpy as np
import time
from collections import deque

class OscillatoryController:
    """Controls neural oscillations for workspace broadcasting"""
    
    def __init__(self):
        self.oscillations = {
            'gamma': {'freq': 40, 'phase': 0, 'amp': 1.0},  # 40Hz - consciousness
            'theta': {'freq': 6, 'phase': 0, 'amp': 0.7},   # 6Hz - memory
            'alpha': {'freq': 10, 'phase': 0, 'amp': 0.5}   # 10Hz - inhibition
        }
        self.last_update = time.time()
        self.state_history = deque(maxlen=1000)
        
    def update(self, dt):
        """Update oscillatory state"""
        # Update phases
        for osc in self.oscillations.values():
            osc['phase'] += 2 * np.pi * osc['freq'] * dt
            osc['phase'] %= 2 * np.pi
            
        # Calculate combined amplitude
        total = sum(osc['amp'] * np.sin(osc['phase']) 
                   for osc in self.oscillations.values())
        normalized = (total / len(self.oscillations) + 1) / 2
        
        # Record state
        self.state_history.append({
            'timestamp': time.time(),
            'amplitude': normalized,
            'phases': {k: v['phase'] for k, v in self.oscillations.items()}
        })
        
        return normalized
        
    def modulate_frequencies(self, attention_load):
        """Modulate frequencies based on attention load"""
        # Increase gamma frequency with attention
        self.oscillations['gamma']['freq'] = 40 + (attention_load * 10)
        # Decrease alpha with attention
        self.oscillations['alpha']['amp'] = max(0.2, 0.5 - (attention_load * 0.3))
