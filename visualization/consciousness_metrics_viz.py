# cogmenta_core/visualization/consciousness_metrics_viz.py
"""
Visualization utilities for consciousness-related metrics.
Helps visualize IIT metrics, recurrent processing, and global workspace dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import defaultdict

class ConsciousnessMetricsVisualizer:
    def __init__(self, output_dir="visualization/output"):
        """Initialize the visualizer with output directory."""
        self.output_dir = output_dir
        self.metrics_history = defaultdict(list)
        self.timestamps = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default styling
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def add_metrics(self, metrics):
        """
        Add a set of metrics to the history.
        
        Args:
            metrics: Dictionary of metrics to track
        """
        timestamp = time.time()
        self.timestamps.append(timestamp)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
    
    def plot_integration_level(self, show=True, save=True):
        """Plot the integration level (Phi) over time."""
        if 'phi' not in self.metrics_history:
            print("No integration level data available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.metrics_history['phi'])), self.metrics_history['phi'], 
                marker='o', linestyle='-', linewidth=2)
        
        plt.title('Integration Level (Φ) Over Time', fontsize=16)
        plt.xlabel('Processing Steps', fontsize=12)
        plt.ylabel('Phi (Φ) Value', fontsize=12)
        plt.grid(True)
        
        # Add horizontal lines for consciousness thresholds
        plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, 
                   label='Minimal Consciousness Threshold')
        plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.7,
                   label='Strong Consciousness Threshold')
        
        plt.legend()
        
        if save:
            filename = os.path.join(self.output_dir, f'integration_level_{int(time.time())}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved integration level plot to {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_subsystem_activities(self, show=True, save=True):
        """Plot the activities of different subsystems over time."""
        subsystem_keys = [k for k in self.metrics_history.keys() 
                         if k.startswith('subsystem_')]
        
        if not subsystem_keys:
            print("No subsystem activity data available")
            return
        
        plt.figure(figsize=(12, 7))
        
        for key in subsystem_keys:
            subsystem_name = key.replace('subsystem_', '')
            plt.plot(range(len(self.metrics_history[key])), 
                    self.metrics_history[key], 
                    label=subsystem_name, linewidth=2)
        
        plt.title('Subsystem Activities Over Time', fontsize=16)
        plt.xlabel('Processing Steps', fontsize=12)
        plt.ylabel('Activity Level', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        if save:
            filename = os.path.join(self.output_dir, f'subsystem_activities_{int(time.time())}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved subsystem activities plot to {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_recurrent_processing(self, show=True, save=True):
        """Plot metrics related to recurrent processing."""
        if 'recurrent_loops' not in self.metrics_history:
            print("No recurrent processing data available")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot number of recurrent loops
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.metrics_history['recurrent_loops'])), 
                self.metrics_history['recurrent_loops'], 
                marker='s', linestyle='-', color='purple')
        plt.title('Recurrent Processing Depth', fontsize=14)
        plt.ylabel('Number of Loops', fontsize=12)
        plt.grid(True)
        
        # Plot phi vs recurrent loops if available
        if 'phi' in self.metrics_history and len(self.metrics_history['phi']) == len(self.metrics_history['recurrent_loops']):
            plt.subplot(2, 1, 2)
            plt.scatter(self.metrics_history['recurrent_loops'], self.metrics_history['phi'], 
                      s=80, c=range(len(self.metrics_history['phi'])), cmap='viridis', alpha=0.7)
            plt.colorbar(label='Processing Step')
            plt.title('Integration Level vs. Recurrent Depth', fontsize=14)
            plt.xlabel('Recurrent Loops', fontsize=12)
            plt.ylabel('Phi (Φ) Value', fontsize=12)
            plt.grid(True)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f'recurrent_processing_{int(time.time())}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved recurrent processing plot to {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_consciousness_dashboard(self, title="Consciousness Metrics Dashboard"):
        """Generate a comprehensive dashboard of all consciousness metrics."""
        # Only proceed if we have data
        if not self.metrics_history:
            print("No metrics data available for dashboard")
            return
        
        plt.figure(figsize=(15, 10))
        plt.suptitle(title, fontsize=18)
        
        # Plot integration level (Phi)
        if 'phi' in self.metrics_history:
            plt.subplot(2, 2, 1)
            plt.plot(range(len(self.metrics_history['phi'])), self.metrics_history['phi'], 
                   marker='o', linestyle='-', linewidth=2, color='blue')
            plt.title('Integration Level (Φ)', fontsize=14)
            plt.xlabel('Steps', fontsize=10)
            plt.ylabel('Phi Value', fontsize=10)
            plt.grid(True)
            
            # Add consciousness thresholds
            plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Min')
            plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.5, label='Strong')
            plt.legend(fontsize=8)
        
        # Plot recurrent loops
        if 'recurrent_loops' in self.metrics_history:
            plt.subplot(2, 2, 2)
            plt.plot(range(len(self.metrics_history['recurrent_loops'])), 
                   self.metrics_history['recurrent_loops'], 
                   marker='s', linestyle='-', color='purple')
            plt.title('Recurrent Processing Depth', fontsize=14)
            plt.xlabel('Steps', fontsize=10)
            plt.ylabel('Loops', fontsize=10)
            plt.grid(True)
        
        # Plot subsystem activities
        subsystem_keys = [k for k in self.metrics_history.keys() 
                         if k.startswith('subsystem_')]
        if subsystem_keys:
            plt.subplot(2, 2, 3)
            
            for key in subsystem_keys:
                subsystem_name = key.replace('subsystem_', '')
                plt.plot(range(len(self.metrics_history[key])), 
                       self.metrics_history[key], 
                       label=subsystem_name, linewidth=1.5)
            
            plt.title('Subsystem Activities', fontsize=14)
            plt.xlabel('Steps', fontsize=10)
            plt.ylabel('Activity', fontsize=10)
            plt.legend(fontsize=8)
            plt.grid(True)
        
        # Plot phi vs recurrent loops
        if 'phi' in self.metrics_history and 'recurrent_loops' in self.metrics_history:
            plt.subplot(2, 2, 4)
            plt.scatter(self.metrics_history['recurrent_loops'], self.metrics_history['phi'], 
                      s=80, c=range(len(self.metrics_history['phi'])), cmap='viridis', alpha=0.7)
            plt.colorbar(label='Step')
            plt.title('Integration vs. Recurrence', fontsize=14)
            plt.xlabel('Recurrent Loops', fontsize=10)
            plt.ylabel('Phi (Φ) Value', fontsize=10)
            plt.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the dashboard
        filename = os.path.join(self.output_dir, f'consciousness_dashboard_{int(time.time())}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved consciousness dashboard to {filename}")
        
        plt.show()

    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history = defaultdict(list)
        self.timestamps = []

# Example usage
if __name__ == "__main__":
    # Create a visualizer
    viz = ConsciousnessMetricsVisualizer()
    
    # Simulate adding metrics
    for i in range(20):
        metrics = {
            'phi': np.random.normal(0.4, 0.15),
            'recurrent_loops': np.random.randint(1, 6),
            'subsystem_symbolic': np.random.normal(0.6, 0.1),
            'subsystem_neural': np.random.normal(0.5, 0.15),
            'subsystem_memory': np.random.normal(0.4, 0.1)
        }
        viz.add_metrics(metrics)
    
    # Generate visualizations
    viz.plot_integration_level()
    viz.plot_subsystem_activities()
    viz.plot_recurrent_processing()
    viz.generate_consciousness_dashboard()