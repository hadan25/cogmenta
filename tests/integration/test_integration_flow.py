import unittest
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge
from cognitive.thought_tracer import ThoughtTrace
from visualization.reasoning_path_viz import ReasoningPathVisualizer

class TestIntegrationFlow(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)
        self.visualizer = ReasoningPathVisualizer(self.thought_trace)

    def test_full_pipeline(self):
        """Test complete pipeline from input to visualization"""
        result = self.bridge.process_text_and_reason("Alice trusts Bob")
        self.assertIn('trace_id', result)
        
        # Test visualization
        graph = self.visualizer.generate_reasoning_graph(result['trace_id'])
        self.assertIsNotNone(graph)
        
        # Verify system state
        self.assertIn('system_state', result)
        self.assertIn('integration_level', result['system_state'])

if __name__ == '__main__':
    unittest.main()
