import unittest
import time
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cogmenta_core.cognitive.thought_tracer import ThoughtTrace
from cogmenta_core.visualization.reasoning_path_viz import ReasoningPathVisualizer
from cogmenta_core.cognitive.self_inspection import SelfInspection
from cogmenta_core.models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge

class TestThoughtTraceIntegration(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.visualizer = ReasoningPathVisualizer(self.thought_trace)
        self.inspector = SelfInspection(self.thought_trace)
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)

    def test_complete_reasoning_flow(self):
        # Test complete flow from bridge through visualization
        result = self.bridge.process_text_and_reason("What is the relationship between Alice and Bob?")
        
        # Verify trace was created
        trace_id = result.get("trace_id")
        self.assertIsNotNone(trace_id)
        
        # Verify trace contains steps
        trace = self.thought_trace.get_trace(trace_id)
        self.assertIsNotNone(trace)
        self.assertTrue(len(trace['steps']) > 0)
        
        # Test visualization generation
        graph_img = self.visualizer.generate_reasoning_graph(trace_id)
        self.assertIsNotNone(graph_img)
        
        # Test self-inspection
        analysis = self.inspector.analyze_trace(trace_id)
        self.assertIsNotNone(analysis)
        self.assertIn('integration_level', analysis)

    def test_multi_trace_comparison(self):
        # Create two traces
        trace1_id = self.thought_trace.start_trace("Test query 1", "TestComponent")
        trace2_id = self.thought_trace.start_trace("Test query 2", "TestComponent")
        
        # Add steps to both traces
        self.thought_trace.add_step(trace1_id, "Component1", "operation1", {"data": "test1"})
        self.thought_trace.add_step(trace2_id, "Component1", "operation1", {"data": "test2"})
        
        # Update metrics
        self.thought_trace.update_metrics(trace1_id, "phi", 0.7)
        self.thought_trace.update_metrics(trace2_id, "phi", 0.4)
        
        # End traces
        self.thought_trace.end_trace(trace1_id, "Conclusion 1")
        self.thought_trace.end_trace(trace2_id, "Conclusion 2")
        
        # Compare traces
        comparison = self.inspector.compare_traces(trace1_id, trace2_id)
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison['trace_ids']), 2)

    def test_visualization_components(self):
        # Create a trace with multiple steps and branches
        trace_id = self.thought_trace.start_trace("Visual test", "TestComponent")
        
        # Add main steps
        step1 = self.thought_trace.add_step(trace_id, "ComponentA", "operationA", {"data": "A"})
        step2 = self.thought_trace.add_step(trace_id, "ComponentB", "operationB", {"data": "B"})
        
        # Add a branch
        branch_id = self.thought_trace.branch_trace(trace_id, "Alternative path")
        self.thought_trace.add_step(branch_id, "ComponentC", "operationC", {"data": "C"})
        
        # Generate visualizations
        graph = self.visualizer.generate_reasoning_graph(trace_id)
        timeline = self.visualizer.generate_confidence_timeline(trace_id)
        
        self.assertIsNotNone(graph)
        self.assertIsNotNone(timeline)

if __name__ == '__main__':
    unittest.main()
