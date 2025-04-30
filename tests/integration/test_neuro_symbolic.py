import unittest
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge
from cognitive.thought_tracer import ThoughtTrace

class TestNeuroSymbolicIntegration(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.bridge = EnhancedNeuroSymbolicBridge(thought_trace=self.thought_trace)

    def test_basic_fact_extraction(self):
        """Test extraction of simple facts"""
        result = self.bridge.process_text_and_reason("Alice trusts Bob")
        self.assertIsNotNone(result)
        self.assertIn('trace_id', result)
        self.assertIn('system_state', result)
        self.assertIn('response', result)
        self.assertTrue(result.get('success', False), 
                       f"Expected success=True, got {result}")
        self.assertIsNone(result.get('error'))

    def test_complex_reasoning(self):
        """Test multi-step reasoning"""
        # Add fact about Alice and Bob
        result1 = self.bridge.process_text_and_reason("Alice trusts Bob")
        self.assertIn('trace_id', result1)
        self.assertTrue(result1.get('success', False))
        
        # Add fact about Bob and Carol
        result2 = self.bridge.process_text_and_reason("Bob likes Carol")
        self.assertIn('trace_id', result2)
        self.assertTrue(result2.get('success', False))
        
        # Query indirect relationship
        result3 = self.bridge.process_text_and_reason("What is the relationship between Alice and Carol?")
        self.assertIn('trace_id', result3)
        self.assertIn('response', result3)
        self.assertIsNotNone(result3.get('response'))

    def test_error_handling(self):
        """Test error handling cases"""
        # Test empty input
        result = self.bridge.process_text_and_reason("")
        self.assertIn('trace_id', result)
        self.assertIn('error', result)
        self.assertFalse(result.get('success', True))
        
        # Test malformed input
        result = self.bridge.process_text_and_reason("@#$%")
        self.assertIn('trace_id', result)
        self.assertIn('response', result)
        self.assertIsNotNone(result.get('response'))

if __name__ == '__main__':
    unittest.main()
