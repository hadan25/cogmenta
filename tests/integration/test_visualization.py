import unittest
from cognitive.thought_tracer import ThoughtTrace
from visualization.reasoning_path_viz import ReasoningPathVisualizer
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.visualizer = ReasoningPathVisualizer(self.thought_trace)
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)

    def test_graph_generation(self):
        """Test graph visualization generation"""
        result = self.bridge.process_text_and_reason("Alice trusts Bob")
        graph = self.visualizer.generate_reasoning_graph(result['trace_id'])
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, str)  # Should be base64 encoded image

    def test_timeline_generation(self):
        """Test timeline visualization generation"""
        result = self.bridge.process_text_and_reason("Bob fears Carol")
        timeline = self.visualizer.generate_confidence_timeline(result['trace_id'])
        self.assertIsNotNone(timeline)
