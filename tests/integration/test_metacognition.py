import unittest
from cognitive.metacognitive_monitor import MetacognitiveMonitor
from cognitive.decision_confidence import DecisionConfidence
from cognitive.thought_tracer import ThoughtTrace
from learning.experience_learner import ExperienceLearner
from optimization.performance_optimizer import PerformanceOptimizer
import time

class TestMetacognitionIntegration(unittest.TestCase):
    def setUp(self):
        self.thought_trace = ThoughtTrace()
        self.monitor = MetacognitiveMonitor(thought_trace=self.thought_trace)
        
    def test_complete_metacognitive_pipeline(self):
        """Test full metacognitive pipeline integration"""
        # Initialize with trace
        self.thought_trace.start_trace("test_pipeline", "TestMetacognition")
        
        # Setup test state with required fields
        state = {
            'workspace': {'active_elements': ['concept_a', 'concept_b']},
            'processing': {'recurrent_loops': 3},
            'integration': {'phi': 0.7},
            'results': [{'confidence': 0.8}],
            'decision_data': {
                'evidence': [{'confidence': 0.8}],
                'process_trace': [{'confidence': 0.7}],
                'inputs': {'required_fields': ['field1'], 'provided_fields': ['field1']}
            }
        }
        
        # Test pipeline
        metrics = self.monitor.update_state(state)
        self.assertIsNotNone(metrics)
        
        # Verify trace exists
        trace = self.thought_trace.get_trace(self.thought_trace.current_trace_id)
        self.assertIsNotNone(trace, "Trace should be created")
        self.assertTrue(any(step['component'] == 'MetacognitiveMonitor' for step in trace['steps']))

    def test_adaptive_learning(self):
        """Test adaptive learning capabilities"""
        # Set initial config
        if 'monitoring' not in self.monitor.config:
            self.monitor.config['monitoring'] = {}
        if 'attention' not in self.monitor.config['monitoring']:
            self.monitor.config['monitoring']['attention'] = {}
        
        self.monitor.config['monitoring']['attention']['load_threshold'] = 0.8
        initial_config = self.monitor.config['monitoring']['attention']['load_threshold']
        
        # Generate states with significantly different load to trigger adaptation
        for i in range(6):
            state = {
                'workspace': {'active_elements': ['elem' + str(j) for j in range(2)]},  # Low load
                'processing': {'recurrent_loops': 3},
                'integration': {'phi': 0.7},
                'success': True,
                'results': [{'confidence': 0.9}]
            }
            self.monitor.update_state(state)
            
        # Process enough successful low-load states to learn
        result = self.monitor.learn_from_experience()
        self.assertIsNotNone(result, "Learning should return a result")
        
        final_config = self.monitor.config['monitoring']['attention']['load_threshold']
        self.assertNotEqual(
            initial_config, 
            final_config, 
            f"Config should adapt from {initial_config} to match lower load pattern (got {final_config})"
        )
        self.assertLess(
            final_config, 
            initial_config, 
            "Load threshold should decrease after seeing successful low-load states"
        )

    def test_performance_optimization_integration(self):
        """Test performance optimization integration"""
        # Create high-load state
        state = {
            'workspace': {'active_elements': ['elem' + str(i) for i in range(10)]},
            'processing': {'recurrent_loops': 1},
            'integration': {'phi': 0.3}
        }
        
        # Update state and trigger optimization
        self.monitor.update_state(state)
        optimization_result = self.monitor.optimize_performance()
        
        # Verify optimization response
        self.assertTrue(any(b['component'] == 'workspace' for b in optimization_result))
        self.assertTrue(any(b['optimization_strategy'] == 'cleanup' for b in optimization_result))

    def test_decision_confidence_integration(self):
        """Test decision confidence integration"""
        decision_data = {
            'evidence': [{'confidence': 0.8}, {'confidence': 0.9}],
            'process_trace': [{'confidence': 0.7}],
            'inputs': {
                'required_fields': ['field1', 'field2'],
                'provided_fields': ['field1', 'field2'],
                'quality_scores': {'field1': 0.9, 'field2': 0.8}
            }
        }
        
        assessment = self.monitor.assess_decision_confidence(decision_data)
        
        # Verify assessment structure
        self.assertIn('confidence', assessment)
        self.assertIn('component_scores', assessment)
        self.assertIn('recommendations', assessment)
        
        # Verify confidence calculation
        self.assertGreater(assessment['confidence'], 0.7)
        self.assertTrue(assessment['high_confidence'])

if __name__ == '__main__':
    unittest.main()
