import unittest
from cognitive.metacognitive_monitor import MetacognitiveMonitor
from cognitive.thought_tracer import ThoughtTrace

class TestMetacognitiveMonitor(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.monitor = MetacognitiveMonitor(thought_trace=self.thought_trace)

    def test_state_update(self):
        """Test state updates"""
        state = {
            'workspace': {'active_elements': ['elem1', 'elem2']},
            'processing': {'recurrent_loops': 3},
            'integration': {'phi': 0.7},
            'results': [{'confidence': 0.8}]
        }
        
        metrics = self.monitor.update_state(state)
        self.assertGreater(metrics['attention_load'], 0)
        self.assertGreater(metrics['processing_depth'], 0)
        self.assertEqual(metrics['coherence'], 0.7)

    def test_regulation(self):
        """Test processing regulation"""
        state = {
            'workspace': {'active_elements': ['elem1'] * 9},  # High load
            'processing': {'recurrent_loops': 1},  # Low depth
            'integration': {'phi': 0.3},  # Low coherence
            'results': [{'confidence': 0.4}]  # Low confidence
        }
        
        self.monitor.update_state(state)
        actions = self.monitor._regulate_processing()
        
        self.assertIsNotNone(actions)
        self.assertTrue(any(a['type'] == 'attention' for a in actions))
        self.assertTrue(any(a['type'] == 'processing' for a in actions))

    def test_consciousness_analysis(self):
        """Test consciousness state analysis"""
        state = {
            'workspace': {'active_elements': ['elem1', 'elem2']},
            'processing': {'recurrent_loops': 3},
            'integration': {'phi': 0.7},
            'results': [{'confidence': 0.8}]
        }
        
        self.monitor.update_state(state)
        analysis = self.monitor.analyze_consciousness_state()
        
        self.assertIn('phi_value', analysis)
        self.assertIn('attention_focus', analysis)
        self.assertIn('processing_depth', analysis)
        self.assertIn('recommendations', analysis)
