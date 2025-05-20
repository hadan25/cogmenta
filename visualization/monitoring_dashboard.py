import dash
from dash import html, dcc
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd

class MonitoringDashboard:
    def __init__(self, thought_trace, update_interval=1000):
        self.thought_trace = thought_trace
        self.app = dash.Dash(__name__)
        self.update_interval = update_interval
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1('Cogmenta Core Monitoring'),
            
            # Performance metrics panel
            html.Div([
                html.H2('System Performance'),
                html.Div([
                    html.Div(id='phi-metric'),
                    html.Div(id='memory-metric'),
                    html.Div(id='active-traces-metric')
                ], className='metrics-panel')
            ]),
            
            # Active traces section
            html.Div([
                html.H2('Active Traces'),
                dcc.Graph(id='active-traces-graph'),
            ]),
            
            # Integration levels
            html.Div([
                html.H2('System Integration Levels'),
                dcc.Graph(id='integration-graph'),
            ]),
            
            # Component activity
            html.Div([
                html.H2('Component Activity'),
                dcc.Graph(id='component-activity-graph'),
            ]),
            
            # Memory usage graph
            html.Div([
                html.H2('Memory Usage'),
                dcc.Graph(id='memory-graph'),
            ]),
            
            # Global workspace status
            html.Div([
                html.H2('Global Workspace Status'),
                dcc.Graph(id='workspace-graph'),
            ]),
            
            # Add Metacognition Monitoring Section
            html.Div([
                html.H2('Metacognitive Analysis'),
                dcc.Graph(id='learning-progress-graph'),
                dcc.Graph(id='strategy-adaptation-graph'),
                dcc.Graph(id='pattern-discovery-graph')
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard update callbacks"""
        @self.app.callback(
            [Output('phi-metric', 'children'),
             Output('memory-metric', 'children'),
             Output('active-traces-metric', 'children'),
             Output('active-traces-graph', 'figure'),
             Output('integration-graph', 'figure'),
             Output('component-activity-graph', 'figure'),
             Output('memory-graph', 'figure'),
             Output('workspace-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(_):
            # Get latest metrics
            metrics = self._get_current_metrics()
            
            return [
                f"Φ (Phi): {metrics['phi']:.3f}",
                f"Memory: {metrics['memory_usage']:.1f}MB",
                f"Active Traces: {metrics['active_traces']}",
                self._get_trace_figure(),
                self._get_integration_figure(),
                self._get_activity_figure(),
                self._get_memory_figure(),
                self._get_workspace_figure()
            ]

    def _get_current_metrics(self):
        """Get current system metrics"""
        recent_traces = self.thought_trace.get_recent_traces(limit=1)
        metrics = {
            'phi': 0.0,
            'memory_usage': 0.0,
            'active_traces': len(self.thought_trace.traces)
        }
        
        if recent_traces:
            trace = recent_traces[0]
            phi_values = trace['metrics'].get('phi_values', [])
            if phi_values:
                metrics['phi'] = phi_values[-1]
        
        return metrics

    def _get_trace_figure(self):
        """Generate active traces figure"""
        recent_traces = self.thought_trace.get_recent_traces(limit=5)
        
        return {
            'data': [{
                'type': 'scatter',
                'x': [t['start_time'] for t in recent_traces],
                'y': [len(t['steps']) for t in recent_traces],
                'mode': 'markers+lines',
                'name': 'Steps Count'
            }],
            'layout': {
                'title': 'Active Traces Processing Steps',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Steps'}
            }
        }

    def _get_integration_figure(self):
        """Generate integration levels figure"""
        recent_traces = self.thought_trace.get_recent_traces(limit=5)
        phi_values = []
        
        for trace in recent_traces:
            if 'metrics' in trace and 'phi_values' in trace['metrics']:
                phi_values.extend(trace['metrics']['phi_values'])

        return {
            'data': [{
                'type': 'scatter',
                'y': phi_values,
                'mode': 'lines',
                'name': 'Integration (Φ)'
            }],
            'layout': {
                'title': 'System Integration Levels',
                'yaxis': {'title': 'Phi (Φ)'}
            }
        }

    def _get_activity_figure(self):
        """Generate component activity figure"""
        recent_traces = self.thought_trace.get_recent_traces(limit=1)
        if not recent_traces:
            return {}
            
        trace = recent_traces[0]
        components = {}
        
        for step in trace['steps']:
            comp = step['component']
            if comp not in components:
                components[comp] = 0
            components[comp] += 1

        return {
            'data': [{
                'type': 'bar',
                'x': list(components.keys()),
                'y': list(components.values()),
            }],
            'layout': {
                'title': 'Component Activity Distribution',
                'xaxis': {'title': 'Component'},
                'yaxis': {'title': 'Activity Count'}
            }
        }
    
    def _get_memory_figure(self):
        """Generate memory usage figure"""
        # ... implementation
        
    def _get_workspace_figure(self):
        """Generate global workspace status figure"""
        # ... implementation
    
    def _get_learning_progress_figure(self):
        """Generate learning progress visualization"""
        return {
            'data': [{
                'type': 'scatter',
                'y': [p['success_rate'] for p in self.thought_trace.monitor.learned_patterns],
                'mode': 'lines+markers',
                'name': 'Learning Progress'
            }],
            'layout': {
                'title': 'Metacognitive Learning Progress',
                'yaxis': {'title': 'Success Rate'}
            }
        }
    
    def run(self, debug=False, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=8050)
