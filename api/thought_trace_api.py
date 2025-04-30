# cogmenta_core/api/thought_trace_api.py

from flask import Blueprint, jsonify, request
from cogmenta_core.cognitive.thought_tracer import ThoughtTrace
from cogmenta_core.visualization.reasoning_path_viz import ReasoningPathVisualizer

thought_trace_api = Blueprint('thought_trace_api', __name__)
thought_trace = ThoughtTrace()  # Singleton instance
visualizer = ReasoningPathVisualizer(thought_trace)

@thought_trace_api.route('/traces', methods=['GET'])
def get_traces():
    """Get recent thought traces."""
    limit = request.args.get('limit', 10, type=int)
    traces = thought_trace.get_recent_traces(limit=limit)
    
    # Format for API response
    response = []
    for trace in traces:
        response.append({
            'id': trace['id'],
            'trigger': trace['trigger'],
            'source': trace['source'],
            'start_time': trace['start_time'],
            'end_time': trace['end_time'],
            'step_count': len(trace['steps']),
            'branch_count': len(trace['branches']),
            'metrics': trace['metrics']
        })
        
    return jsonify(response)

@thought_trace_api.route('/traces/<trace_id>', methods=['GET'])
def get_trace(trace_id):
    """Get a specific thought trace by ID."""
    trace = thought_trace.get_trace(trace_id)
    if not trace:
        return jsonify({'error': 'Trace not found'}), 404
        
    return jsonify(trace)

@thought_trace_api.route('/traces/<trace_id>/visualization', methods=['GET'])
def get_visualization(trace_id):
    """Get visualization for a thought trace."""
    viz_type = request.args.get('type', 'graph')
    
    if viz_type == 'graph':
        img_data = visualizer.generate_reasoning_graph(trace_id)
    elif viz_type == 'confidence':
        img_data = visualizer.generate_confidence_timeline(trace_id)
    else:
        return jsonify({'error': 'Invalid visualization type'}), 400
        
    if not img_data:
        return jsonify({'error': 'Could not generate visualization'}), 500
        
    return jsonify({'image_data': img_data})

@thought_trace_api.route('/traces/search', methods=['GET'])
def search_traces():
    """Search for traces matching a query."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
        
    traces = thought_trace.search_traces(query)
    
    # Format for API response
    response = []
    for trace in traces:
        response.append({
            'id': trace['id'],
            'trigger': trace['trigger'],
            'source': trace['source'],
            'start_time': trace['start_time'],
            'end_time': trace['end_time'],
            'step_count': len(trace['steps']),
            'metrics': trace['metrics']
        })
        
    return jsonify(response)

@thought_trace_api.route('/traces/<trace_id>/activations', methods=['GET'])
def get_neural_activations(trace_id):
    """Get neural activation data for a thought trace."""
    activations = thought_trace.get_neural_activations(trace_id)
    if not activations:
        return jsonify({'error': 'No activation data found'}), 404
        
    return jsonify(activations)