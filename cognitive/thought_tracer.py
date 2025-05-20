# cogmenta_core/cognitive/thought_trace.py

import time
import uuid
import numpy as np
from collections import defaultdict

class ThoughtTrace:
    """
    Centralized system for tracking reasoning processes across components.
    Implements a standardized format for capturing thought processes.
    """
    
    def __init__(self):
        self.traces = []
        self.current_trace_id = None
        self.trace_index = {}  # For efficient lookup
        self.active_branches = {}  # For tracking parallel reasoning
        self.activation_history = {}  # Track neural activations
        self.activation_patterns = {}  # Store neural activation patterns
        self.reasoning_chains = {}     # Store reasoning chains
        self.critical_points = {}      # Store important decision points
        
    def start_trace(self, trigger, source_component, operation=None, data=None):
        """Start a new thought trace."""
        if not trigger or not source_component:
            return None

        trace_id = str(uuid.uuid4())
        trace = {
            'id': trace_id,
            'trigger': trigger,
            'source': source_component,
            'start_time': time.time(),
            'end_time': None,
            'steps': [],
            'branches': [],
            'conclusion': None,
            'metrics': {
                'recurrent_loops': 0,
                'phi_values': [],
                'confidence': None
            },
            'critical_steps': []
        }
        
        self.traces.append(trace)
        self.trace_index[trace_id] = trace
        self.current_trace_id = trace_id
        
        if operation:
            self.add_step(trace_id, source_component, operation, data or {})
            
        return trace_id
        
    def add_step(self, trace_id, component, operation, data):
        """Add a reasoning step to a trace."""
        if trace_id not in self.trace_index:
            return False
            
        step = {
            'id': str(uuid.uuid4()),
            'trace_id': trace_id,
            'component': component,
            'operation': operation,
            'timestamp': time.time(),
            'data': data
        }
        
        self.trace_index[trace_id]['steps'].append(step)
        return step['id']
        
    def branch_trace(self, parent_trace_id, branch_reason):
        """Create a branch of reasoning from an existing trace."""
        if parent_trace_id not in self.trace_index:
            return None
            
        branch_id = self.start_trace(
            f"Branch from {parent_trace_id}: {branch_reason}",
            self.trace_index[parent_trace_id]['source']
        )
        
        # Record branch relationship
        self.trace_index[parent_trace_id]['branches'].append({
            'branch_id': branch_id,
            'reason': branch_reason,
            'timestamp': time.time()
        })
        
        return branch_id
        
    def end_trace(self, trace_id, conclusion, confidence=None):
        """End a trace with a conclusion."""
        if trace_id not in self.trace_index:
            return False
            
        trace = self.trace_index[trace_id]
        trace['end_time'] = time.time()
        trace['conclusion'] = conclusion
        
        if confidence is not None:
            trace['metrics']['confidence'] = confidence
            
        return True
        
    def update_metrics(self, trace_id, metric_name, value):
        """Update metrics for a trace."""
        if trace_id not in self.trace_index:
            return False
            
        if metric_name == 'phi':
            self.trace_index[trace_id]['metrics']['phi_values'].append(value)
        elif metric_name == 'recurrent_loops':
            self.trace_index[trace_id]['metrics']['recurrent_loops'] = value
        else:
            self.trace_index[trace_id]['metrics'][metric_name] = value
            
        return True
        
    def get_trace(self, trace_id):
        """Get a complete trace by ID."""
        return self.trace_index.get(trace_id)
        
    def get_recent_traces(self, limit=10):
        """Get the most recent traces."""
        return sorted(self.traces, key=lambda t: t['start_time'], reverse=True)[:limit]
        
    def search_traces(self, query):
        """Search traces based on content, components, etc."""
        results = []
        
        for trace in self.traces:
            # Search in trigger
            if query.lower() in trace['trigger'].lower():
                results.append(trace)
                continue
                
            # Search in steps
            for step in trace['steps']:
                if (query.lower() in step['component'].lower() or
                    query.lower() in step['operation'].lower()):
                    results.append(trace)
                    break
                    
            # Search in conclusion
            if trace['conclusion'] and query.lower() in str(trace['conclusion']).lower():
                if trace not in results:
                    results.append(trace)
                    
        return results
        
    def add_neural_activation(self, trace_id, layer_name, activations):
        """Record neural network layer activations."""
        if trace_id not in self.trace_index:
            return False
            
        if trace_id not in self.activation_history:
            self.activation_history[trace_id] = {}
            
        self.activation_history[trace_id][layer_name] = {
            'timestamp': time.time(),
            'mean_activation': float(np.mean(activations)),
            'max_activation': float(np.max(activations)),
            'activation_pattern': activations.tolist() if len(activations) < 100 else 
                                [float(x) for x in np.percentile(activations, range(0,101,10))]
        }
        return True
        
    def get_neural_activations(self, trace_id):
        """Get recorded neural activations for a trace."""
        return self.activation_history.get(trace_id, {})

    def record_neural_activation(self, trace_id, layer_name, activation_pattern, importance=0.5):
        """Record neural network layer activations with importance score"""
        if trace_id not in self.activation_patterns:
            self.activation_patterns[trace_id] = []
            
        self.activation_patterns[trace_id].append({
            'layer': layer_name,
            'pattern': activation_pattern,
            'importance': importance,
            'timestamp': time.time()
        })

    def mark_critical_point(self, trace_id, reason, data=None):
        """Mark an important point in reasoning"""
        if trace_id not in self.critical_points:
            self.critical_points[trace_id] = []
            
        self.critical_points[trace_id].append({
            'reason': reason,
            'data': data,
            'timestamp': time.time(),
            'step_index': len(self.trace_index[trace_id]['steps']) - 1
        })
        
        # Add to critical steps list
        if trace_id in self.trace_index:
            self.trace_index[trace_id]['critical_steps'].append(
                len(self.trace_index[trace_id]['steps']) - 1
            )