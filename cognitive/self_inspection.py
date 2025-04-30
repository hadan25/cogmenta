# cogmenta_core/cognitive/self_inspection.py

import numpy as np
from collections import Counter, defaultdict

class SelfInspection:
    """
    Provides self-inspection capabilities to analyze and improve reasoning.
    """
    
    def __init__(self, thought_trace, prolog_engine=None, vector_engine=None):
        """Initialize with access to thought traces and reasoning engines."""
        self.thought_trace = thought_trace
        self.prolog = prolog_engine
        self.vector = vector_engine
        self.knowledge_patterns = defaultdict(list)
        
    def analyze_trace(self, trace_id):
        """Analyze a reasoning trace for patterns and potential improvements."""
        trace = self.thought_trace.get_trace(trace_id)
        if not trace:
            return None
            
        analysis = {
            'trace_id': trace_id,
            'integration_level': np.mean(trace['metrics'].get('phi_values', [0])),
            'recurrent_depth': trace['metrics'].get('recurrent_loops', 0),
            'duration': trace['end_time'] - trace['start_time'] if trace['end_time'] else 0,
            'component_usage': self._analyze_component_usage(trace),
            'critical_steps': self._identify_critical_steps(trace),
            'bottlenecks': self._identify_bottlenecks(trace),
            'improvement_suggestions': []
        }
        
        # Generate improvement suggestions
        if analysis['integration_level'] < 0.4:
            analysis['improvement_suggestions'].append({
                'issue': 'Low integration level',
                'suggestion': 'Increase recurrent processing loops or enhance cross-component communication'
            })
            
        if analysis['recurrent_depth'] < 3:
            analysis['improvement_suggestions'].append({
                'issue': 'Shallow recurrent processing',
                'suggestion': 'Increase recurrent loops to allow for deeper processing'
            })
            
        for bottleneck in analysis['bottlenecks']:
            analysis['improvement_suggestions'].append({
                'issue': f"Bottleneck in {bottleneck['component']}",
                'suggestion': bottleneck['suggestion']
            })
            
        return analysis
        
    def _analyze_component_usage(self, trace):
        """Analyze which components were used in reasoning."""
        components = Counter([step['component'] for step in trace['steps']])
        
        return {
            'components': dict(components),
            'dominant_component': components.most_common(1)[0][0] if components else None,
            'component_balance': self._calculate_component_balance(components)
        }
        
    def _calculate_component_balance(self, component_counter):
        """Calculate how balanced the component usage is."""
        if not component_counter:
            return 1.0  # No components used
            
        counts = list(component_counter.values())
        total = sum(counts)
        
        if total == 0:
            return 1.0
            
        # Calculate entropy-based balance measure
        proportions = [count/total for count in counts]
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        max_entropy = np.log2(len(counts))
        
        # Normalize to 0-1 range
        balance = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return balance
        
    def _identify_critical_steps(self, trace):
        """Identify critical steps in the reasoning process."""
        critical_steps = []
        
        if not trace['steps']:
            return critical_steps
            
        # Look for steps that preceded significant changes in activation or confidence
        phi_values = trace['metrics'].get('phi_values', [])
        if len(phi_values) > 1:
            for i in range(len(phi_values) - 1):
                if abs(phi_values[i+1] - phi_values[i]) > 0.2:  # Significant change
                    if i < len(trace['steps']):
                        critical_steps.append({
                            'step_id': trace['steps'][i]['id'],
                            'component': trace['steps'][i]['component'],
                            'operation': trace['steps'][i]['operation'],
                            'significance': 'Large change in integration level',
                            'delta': phi_values[i+1] - phi_values[i]
                        })
        
        # Identify branch points
        for branch in trace['branches']:
            # Find the step that preceded this branch
            for i, step in enumerate(trace['steps']):
                if step['timestamp'] <= branch['timestamp'] and (i == len(trace['steps']) - 1 or 
                                                                trace['steps'][i+1]['timestamp'] > branch['timestamp']):
                    critical_steps.append({
                        'step_id': step['id'],
                        'component': step['component'],
                        'operation': step['operation'],
                        'significance': f"Branching point: {branch['reason']}",
                        'branch_id': branch['branch_id']
                    })
                    break
                    
        return critical_steps
        
    def _identify_bottlenecks(self, trace):
        """Identify potential bottlenecks in the reasoning process."""
        bottlenecks = []
        
        if not trace['steps']:
            return bottlenecks
            
        # Calculate time spent in each component
        component_times = defaultdict(float)
        for i in range(len(trace['steps']) - 1):
            current = trace['steps'][i]
            next_step = trace['steps'][i+1]
            duration = next_step['timestamp'] - current['timestamp']
            component_times[current['component']] += duration
            
        # Add time for final step
        if trace['steps'] and trace['end_time']:
            final = trace['steps'][-1]
            duration = trace['end_time'] - final['timestamp']
            component_times[final['component']] += duration
            
        # Identify components taking excessive time
        total_time = trace['end_time'] - trace['start_time'] if trace['end_time'] else 0
        for component, time_spent in component_times.items():
            if time_spent > 0.5 * total_time:  # Component takes >50% of total time
                bottlenecks.append({
                    'component': component,
                    'time_spent': time_spent,
                    'percentage': (time_spent / total_time) * 100 if total_time > 0 else 0,
                    'suggestion': f"Optimize processing in {component} to reduce overall reasoning time"
                })
                
        return bottlenecks
        
    def compare_traces(self, trace_id1, trace_id2):
        """Compare two reasoning traces to identify differences in approach."""
        trace1 = self.thought_trace.get_trace(trace_id1)
        trace2 = self.thought_trace.get_trace(trace_id2)
        
        if not trace1 or not trace2:
            return None
            
        comparison = {
            'trace_ids': [trace_id1, trace_id2],
            'triggers': [trace1['trigger'], trace2['trigger']],
            'integration_levels': [
                np.mean(trace1['metrics'].get('phi_values', [0])),
                np.mean(trace2['metrics'].get('phi_values', [0]))
            ],
            'recurrent_depths': [
                trace1['metrics'].get('recurrent_loops', 0),
                trace2['metrics'].get('recurrent_loops', 0)
            ],
            'durations': [
                trace1['end_time'] - trace1['start_time'] if trace1['end_time'] else 0,
                trace2['end_time'] - trace2['start_time'] if trace2['end_time'] else 0
            ],
            'step_counts': [len(trace1['steps']), len(trace2['steps'])],
            'branch_counts': [len(trace1['branches']), len(trace2['branches'])],
            'component_usage': [
                Counter([step['component'] for step in trace1['steps']]),
                Counter([step['component'] for step in trace2['steps']])
            ],
            'key_differences': []
        }
        
        # Identify key differences
        if abs(comparison['integration_levels'][0] - comparison['integration_levels'][1]) > 0.2:
            comparison['key_differences'].append({
                'aspect': 'Integration level',
                'difference': f"{comparison['integration_levels'][0]:.2f} vs {comparison['integration_levels'][1]:.2f}",
                'significance': 'Different degrees of integration indicating varying processing depth'
            })
            
        return comparison

    def analyze_knowledge_distribution(self):
        """Analyze how knowledge is distributed between symbolic and vector representations"""
        analysis = {
            'prolog_facts': 0,
            'vector_facts': 0,
            'shared_concepts': [],
            'knowledge_gaps': [],
            'suggestions': []
        }
        
        if self.prolog:
            # Count logical/mathematical facts
            query = "confident_fact(P, S, O, C)"
            prolog_facts = list(self.prolog.query(query))
            analysis['prolog_facts'] = len(prolog_facts)
            
            # Identify mathematical/logical concepts
            math_concepts = set()
            for fact in prolog_facts:
                if any(term in str(fact['P']).lower() for term in ['equals', 'greater', 'less', 'sum', 'product']):
                    math_concepts.add(str(fact['S']))
                    math_concepts.add(str(fact['O']))
        
        if self.vector:
            # Count semantic/language facts
            analysis['vector_facts'] = len(self.vector.facts)
            
            # Find concepts that exist in both systems
            vector_concepts = set(self.vector.concept_vectors.keys())
            analysis['shared_concepts'] = list(math_concepts & vector_concepts)
            
            # Identify potential knowledge gaps
            for concept in math_concepts - vector_concepts:
                analysis['knowledge_gaps'].append({
                    'concept': concept,
                    'missing_in': 'vector',
                    'suggestion': 'Add semantic representation for mathematical concept'
                })
            
            for concept in vector_concepts - math_concepts:
                if any(term in concept.lower() for term in ['number', 'equation', 'formula', 'proof']):
                    analysis['knowledge_gaps'].append({
                        'concept': concept,
                        'missing_in': 'prolog',
                        'suggestion': 'Add logical representation for mathematical term'
                    })
        
        return analysis