import unittest
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge
from cognitive.thought_tracer import ThoughtTrace

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.bridge.process_text_and_reason("")
        self.assertIn('trace_id', result)
        self.assertIn('error', result)
    
    def test_malformed_input(self):
        """Test handling of malformed input"""
        result = self.bridge.process_text_and_reason("@#$%^")
        self.assertIn('trace_id', result)
        self.assertIn('response', result)
    
    def test_very_long_input(self):
        """Test handling of very long input"""
        long_text = "Alice " * 1000
        result = self.bridge.process_text_and_reason(long_text)
        self.assertIn('trace_id', result)
        self.assertIn('response', result)

if __name__ == '__main__':
    unittest.main()
