# cogmenta_core/visualization/reasoning_path_viz.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import LinearSegmentedColormap

class ReasoningPathVisualizer:
    """
    Visualizes reasoning paths for exploring thought traces.
    """
    
    def __init__(self, thought_trace):
        self.thought_trace = thought_trace
        self.layout_options = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout
        }
        
    def generate_reasoning_graph(self, trace_id, layout='spring', highlight_critical=True):
        """Generate a graph visualization of the reasoning process."""
        if trace_id not in self.thought_trace.trace_index:
            return None
            
        trace = self.thought_trace.get_trace(trace_id)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each step
        nodes = {}
        for i, step in enumerate(trace['steps']):
            node_id = f"step_{i}"
            nodes[step['id']] = node_id
            
            # Create node label
            label = f"{step['component']}\n{step['operation']}"
            
            # Add node with attributes
            G.add_node(
                node_id, 
                label=label,
                component=step['component'],
                operation=step['operation'],
                step_num=i
            )
            
            # Add edge from previous step if it exists
            if i > 0:
                prev_step = trace['steps'][i-1]
                G.add_edge(nodes[prev_step['id']], node_id)
        
        # Add branch connections
        for branch in trace['branches']:
            branch_trace = self.thought_trace.get_trace(branch['branch_id'])
            if branch_trace and branch_trace['steps']:
                branch_start = f"branch_{branch['branch_id'][:8]}"
                G.add_node(
                    branch_start, 
                    label=f"Branch: {branch['reason']}",
                    component="Branch",
                    operation=branch['reason'],
                    step_num=-1
                )
                
                # Connect to parent trace at appropriate step
                for i, step in enumerate(trace['steps']):
                    if step['timestamp'] <= branch['timestamp'] and (i == len(trace['steps']) - 1 or 
                                                                   trace['steps'][i+1]['timestamp'] > branch['timestamp']):
                        G.add_edge(nodes[step['id']], branch_start)
                        break
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create layout using selected algorithm
        pos = self.layout_options.get(layout, nx.spring_layout)(G)
        
        # Color nodes by component
        components = set(nx.get_node_attributes(G, 'component').values())
        colormap = plt.cm.tab10
        color_mapping = {comp: colormap(i/len(components)) for i, comp in enumerate(components)}
        
        # Highlight critical nodes if requested
        if highlight_critical:
            critical_nodes = [n for n in G.nodes() if G.nodes[n].get('step_num', -1) in 
                            [s['step_num'] for s in trace.get('critical_steps', [])]]
            node_colors = ['red' if n in critical_nodes else color_mapping[G.nodes[n]['component']] 
                         for n in G.nodes()]
        else:
            node_colors = [color_mapping[G.nodes[n]['component']] for n in G.nodes()]
        
        # Draw the graph
        nx.draw(
            G, pos, 
            with_labels=True,
            node_color=node_colors,
            node_size=1500,
            font_size=8,
            labels={n: G.nodes[n]['label'] for n in G.nodes()}
        )
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color, markersize=10,
                             label=comp) for comp, color in color_mapping.items()]
        plt.legend(handles=legend_elements, title="Components")
        
        plt.title(f"Reasoning Path for: {trace['trigger']}")
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 for web display
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
        
    def generate_confidence_timeline(self, trace_id):
        """Generate timeline visualization of confidence over reasoning steps"""
        trace = self.thought_trace.get_trace(trace_id)
        if not trace or not trace.get('steps'):
            return None

        plt.figure(figsize=(12, 6))
        
        # Get all relevant metrics
        timestamps = []
        confidences = []
        phi_values = []
        critical_points = []
        
        start_time = trace['start_time']
        
        # Process steps
        for i, step in enumerate(trace['steps']):
            rel_time = step['timestamp'] - start_time
            timestamps.append(rel_time)
            
            # Get confidence from step data or phi values
            conf = step.get('data', {}).get('confidence', None)
            if conf is None:
                phi_values = trace['metrics'].get('phi_values', [])
                conf = phi_values[min(len(phi_values)-1, len(confidences))] if phi_values else 0.5
            confidences.append(conf)
            
            # Mark critical points
            if i in trace.get('critical_steps', []):
                critical_points.append((rel_time, conf))

        # Plot main confidence line
        plt.plot(timestamps, confidences, 'b-', label='Confidence', linewidth=2)
        
        # Highlight critical points
        if critical_points:
            crit_times, crit_confs = zip(*critical_points)
            plt.plot(crit_times, crit_confs, 'ro', label='Critical Points', markersize=10)

        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence / Integration Level')
        plt.title('Reasoning Timeline with Critical Points')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add confidence bands
        if len(confidences) > 1:
            plt.fill_between(timestamps, 
                           [max(0, c-0.1) for c in confidences],
                           [min(1, c+0.1) for c in confidences],
                           color='blue', alpha=0.2)

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.read()).decode('utf-8')