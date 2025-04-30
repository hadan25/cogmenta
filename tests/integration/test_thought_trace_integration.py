import unittest
import time
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cognitive.thought_tracer import ThoughtTrace
from visualization.reasoning_path_viz import ReasoningPathVisualizer
from cognitive.self_inspection import SelfInspection
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge

class TestThoughtTraceIntegration(unittest.TestCase):
    def setUp(self):
        """Initialize test components"""
        self.thought_trace = ThoughtTrace()
        self.visualizer = ReasoningPathVisualizer(self.thought_trace)
        self.inspector = SelfInspection(self.thought_trace)
        self.bridge = EnhancedNeuroSymbolicBridge(
            thought_trace=self.thought_trace,
            use_enhanced_snn=True
        )

    def test_complete_reasoning_flow(self):
        """Test complete reasoning flow"""
        # Ensure bridge is properly initialized
        self.assertIsNotNone(self.bridge)
        self.assertIsNotNone(self.bridge.thought_trace)
        
        # Test processing
        result = self.bridge.process_text_and_reason("Alice trusts Bob")
        
        # Verify result structure
        self.assertIsNotNone(result, "Result should not be None")
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        
        # Verify trace_id
        self.assertIn('trace_id', result, "Result should contain trace_id")
        trace_id = result.get('trace_id')
        self.assertIsNotNone(trace_id, "Trace ID should not be None")
        
        # Verify trace contains steps
        trace = self.thought_trace.get_trace(trace_id)
        self.assertIsNotNone(trace, "Should be able to retrieve trace")
        self.assertTrue(len(trace['steps']) > 0, "Trace should contain steps")
        
        # Verify IIT metrics
        self.assertIn('system_state', result, "Result should include system state")
        system_state = result['system_state']
        self.assertIn('integration_level', system_state, "Should have integration level")
        self.assertIn('recurrent_loops', system_state, "Should have recurrent loops count")
        self.assertIn('subsystem_activities', system_state, "Should have subsystem activities")
        
        # Test visualization generation
        graph_img = self.visualizer.generate_reasoning_graph(trace_id)
        self.assertIsNotNone(graph_img, "Should generate graph visualization")
        
        # Test self-inspection with new metrics
        analysis = self.inspector.analyze_trace(trace_id)
        self.assertIsNotNone(analysis, "Should generate analysis")
        self.assertIn('integration_level', analysis, "Analysis should include integration level")
        self.assertIn('recurrent_depth', analysis, "Analysis should include recurrent depth")

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
        
        # Add main steps with confidence values
        step1 = self.thought_trace.add_step(
            trace_id, 
            "ComponentA", 
            "operationA", 
            {"data": "A", "confidence": 0.5}
        )
        
        step2 = self.thought_trace.add_step(
            trace_id, 
            "ComponentB", 
            "operationB", 
            {"data": "B", "confidence": 0.7}
        )
        
        # Add a branch
        branch_id = self.thought_trace.branch_trace(trace_id, "Alternative path")
        self.thought_trace.add_step(branch_id, "ComponentC", "operationC", {"data": "C"})
        
        # Add metrics
        self.thought_trace.update_metrics(trace_id, "phi", 0.6)
        
        # End the trace
        self.thought_trace.end_trace(trace_id, "Test conclusion")
        
        # Generate visualizations
        graph = self.visualizer.generate_reasoning_graph(trace_id)
        self.assertIsNotNone(graph, "Should generate graph visualization")
        
        timeline = self.visualizer.generate_confidence_timeline(trace_id)
        self.assertIsNotNone(timeline, "Should generate timeline visualization")

    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with invalid input
        result = self.bridge.process_text_and_reason("")
        self.assertIsNotNone(result.get('trace_id'))
        self.assertIn('error', result)
        
        # Test with malformed input
        result = self.bridge.process_text_and_reason("@#$%")
        self.assertIsNotNone(result.get('trace_id'))
        self.assertTrue(result.get('response', '').startswith("I"))

    def test_multiple_traces(self):
        """Test handling multiple traces in sequence"""
        # Process multiple inputs
        results = []
        inputs = [
            "Alice trusts Bob",
            "Bob likes Carol",
            "Carol fears Dave"
        ]
        
        for text in inputs:
            result = self.bridge.process_text_and_reason(text)
            results.append(result)
            
        # Verify each has unique trace_id
        trace_ids = [r.get('trace_id') for r in results]
        self.assertEqual(len(set(trace_ids)), len(inputs))

if __name__ == '__main__':
    unittest.main()
