# cogmenta_core/visualization/activation_trace.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64

class ActivationTraceVisualizer:
    """
    Visualizes neural activations over time to show how activation patterns evolve.
    """
    
    def __init__(self):
        self.activation_history = []
        
    def record_activation(self, activation_data, step, metadata=None):
        """Record an activation state."""
        self.activation_history.append({
            'step': step,
            'activations': activation_data,
            'metadata': metadata or {}
        })
    
    def generate_heatmap(self, max_steps=None, region_boundaries=None):
        """Generate a heatmap of activations over time."""
        if not self.activation_history:
            return None
            
        # Determine how many steps to show
        steps_to_show = min(max_steps or len(self.activation_history), len(self.activation_history))
        history_subset = self.activation_history[-steps_to_show:]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract activations
        activations = np.array([h['activations'] for h in history_subset])
        
        # Create heatmap
        im = ax.imshow(
            activations, 
            aspect='auto', 
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Level')
        
        # Add region boundaries if provided
        if region_boundaries:
            for region, (start, end) in region_boundaries.items():
                ax.axhline(y=start, color='r', linestyle='-', alpha=0.3)
                ax.axhline(y=end, color='r', linestyle='-', alpha=0.3)
                ax.text(activations.shape[1] + 1, (start + end) / 2, region, 
                       verticalalignment='center')
        
        # Set labels
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Time Step')
        ax.set_title('Neural Activation Patterns Over Time')
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 for web display
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
    
    def generate_animation(self, region_mapping=None):
        """Generate an animation of activation patterns evolving over time."""
        if not self.activation_history:
            return None
            
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine dimensions
        num_neurons = len(self.activation_history[0]['activations'])
        
        # Create initial empty plot
        bars = ax.bar(range(num_neurons), np.zeros(num_neurons))
        
        # Set axis limits
        ax.set_ylim(0, 1)
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Activation Level')
        ax.set_title('Neural Activation Animation')
        
        # Color code by region if mapping provided
        if region_mapping:
            colors = plt.cm.tab10(np.linspace(0, 1, len(region_mapping)))
            color_map = {}
            for i, (region, indices) in enumerate(region_mapping.items()):
                for idx in indices:
                    if idx < num_neurons:
                        color_map[idx] = colors[i]
                        
            # Apply colors
            for i, bar in enumerate(bars):
                bar.set_color(color_map.get(i, 'blue'))
        
        # Animation update function
        def update(frame):
            data = self.activation_history[frame]['activations']
            for i, val in enumerate(data):
                bars[i].set_height(val)
            ax.set_title(f'Neural Activation - Step {frame+1}')
            return bars
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(self.activation_history),
            blit=True, interval=200
        )
        
        # Save to bytes buffer
        buf = io.BytesIO()
        ani.save(buf, writer='pillow', fps=5)
        buf.seek(0)
        
        # Convert to base64 for web display
        vid_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return vid_str