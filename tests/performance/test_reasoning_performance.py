import unittest
import time
from cognitive.thought_tracer import ThoughtTrace
from models.hybrid.enhanced_neuro_symbolic_bridge1 import EnhancedNeuroSymbolicBridge

class TestReasoningPerformance(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)

    def test_processing_time(self):
        """Test processing time stays within acceptable range"""
        start_time = time.time()
        result = self.bridge.process_text_and_reason("Alice trusts Bob")
        processing_time = time.time() - start_time
        
        self.assertLess(processing_time, 1.0)  # Should process in under 1 second
        self.assertIsNotNone(result.get('trace_id'))
