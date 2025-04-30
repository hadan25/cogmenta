from cognitive.global_workspace import GlobalWorkspace
from cognitive.metacognitive_monitor import MetacognitiveMonitor
from controller.oscillatory_control import OscillatoryController
from controller.attention import AttentionMechanism
import time

class ConsciousnessIntegration:
    def __init__(self, thought_trace=None):
        self.global_workspace = GlobalWorkspace()
        self.metacognitive = MetacognitiveMonitor(thought_trace=thought_trace)
        self.oscillator = OscillatoryController()
        self.attention = AttentionMechanism()
        
        # Integration state
        self.last_update = time.time() - 0.1  # Ensure first update allows broadcast
        self.integration_active = True
        
    def process_cognitive_state(self, state):
        """Process current cognitive state through consciousness pipeline"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Ensure minimum dt to allow oscillation
        dt = max(dt, 0.01)
        
        # Update oscillatory state with higher initial amplitude
        oscillation = self.oscillator.update(dt)
        broadcast_strength = min(1.0, oscillation * 1.5)  # Boost broadcast strength
        
        # Calculate attention scores with boosted initial values
        if 'salient_items' in state:
            attention_focus = self.attention.update_focus(
                state['salient_items'],
                item_properties=state.get('item_properties', {'default': {'intensity': 0.9}})
            )
        else:
            attention_focus = None
            
        # Broadcast with lower threshold
        if oscillation > 0.3 and attention_focus:  # Lower threshold further
            self.global_workspace.broadcast(attention_focus, broadcast_strength)
            
        # Update metacognitive monitoring
        meta_state = self.metacognitive.update_state({
            'workspace': self.global_workspace.get_workspace_state(),
            'attention': self.attention.get_attention_state(),
            'oscillation': oscillation,
            **state
        })
        
        # Store timing
        self.last_update = current_time
        
        return {
            'workspace_state': self.global_workspace.get_workspace_state(),
            'meta_state': meta_state,
            'attention_focus': attention_focus,
            'oscillation': oscillation
        }
