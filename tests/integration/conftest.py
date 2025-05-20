import pytest
from cognitive.thought_tracer import ThoughtTrace
from visualization.reasoning_path_viz import ReasoningPathVisualizer
from cognitive.self_inspection import SelfInspection

@pytest.fixture
def thought_trace():
    return ThoughtTrace()

@pytest.fixture
def visualizer(thought_trace):
    return ReasoningPathVisualizer(thought_trace)

@pytest.fixture
def inspector(thought_trace):
    return SelfInspection(thought_trace)

@pytest.fixture
def sample_trace(thought_trace):
    """Create a sample trace with some steps for testing"""
    trace_id = thought_trace.start_trace("Test trace", "TestComponent")
    
    # Add some steps
    thought_trace.add_step(trace_id, "ComponentA", "operationA", {"data": "test"})
    thought_trace.add_step(trace_id, "ComponentB", "operationB", {"data": "test"})
    
    # Add some metrics
    thought_trace.update_metrics(trace_id, "phi", 0.5)
    thought_trace.update_metrics(trace_id, "recurrent_loops", 2)
    
    thought_trace.end_trace(trace_id, "Test conclusion")
    
    return trace_id
