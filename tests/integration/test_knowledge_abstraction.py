import unittest
import numpy as np
from models.symbolic.knowledge_abstraction import KnowledgeAbstraction
from models.symbolic.abstraction_validation import AbstractionValidator
from models.symbolic.prolog_engine import PrologEngine
from models.symbolic.vector_symbolic import VectorSymbolicEngine

class TestKnowledgeAbstractionIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize components
        self.prolog = PrologEngine()
        
        # Add base rules first
        self.prolog.prolog.assertz(":-dynamic(fact/4)")
        self.prolog.prolog.assertz(":-dynamic(confident_fact/4)")
        
        # Define confident_fact rule properly
        self.prolog.prolog.assertz("confident_fact(P, S, O, C) :- fact(P, S, O, C)")
        
        # Ensure facts are properly added in setUp
        facts = [
            ("trusts", "alice", "bob", 0.8),
            ("likes", "bob", "alice", 0.7),
            ("fears", "charlie", "dave", 0.7)
        ]
        
        for pred, subj, obj, conf in facts:
            # Add both fact and confident_fact
            self.prolog.prolog.assertz(f"fact('{pred}', '{subj}', '{obj}', {conf})")
            self.prolog.prolog.assertz(f"confident_fact('{pred}', '{subj}', '{obj}', {conf})")
            
        self.vector = VectorSymbolicEngine(dimension=100)
        self.validator = AbstractionValidator(
            prolog_engine=self.prolog,
            vector_engine=self.vector
        )
        self.abstraction = KnowledgeAbstraction(
            prolog_engine=self.prolog,
            vector_engine=self.vector,
            validator=self.validator
        )
        
    def test_abstraction_validation_pipeline(self):
        """Test complete abstraction and validation pipeline"""
        # Find patterns
        abstractions = self.abstraction.find_patterns()
        self.assertTrue(len(abstractions) > 0)
        
        # Apply and validate abstractions
        validated_count = self.abstraction.apply_abstractions()
        self.assertGreater(validated_count, 0)
        
        # Check validation history
        self.assertGreater(len(self.validator.validation_history), 0)
        
        # Analyze quality
        quality = self.abstraction.analyze_abstraction_quality()
        self.assertGreater(quality['validation_rate'], 0)
        
    def test_pattern_learning(self):
        """Test automatic pattern learning capabilities"""
        # Run initial abstractions to build history
        self.abstraction.apply_abstractions()
        
        # Learn new patterns
        new_rules = self.abstraction.learn_new_abstractions(min_confidence=0.7)
        self.assertIsInstance(new_rules, list)
        
        # Verify validation patterns
        patterns = self.validator.analyze_validation_patterns()
        self.assertIn('success_patterns', patterns)
        self.assertIn('failure_patterns', patterns)
        
    def test_cross_engine_consistency(self):
        """Test consistency between Prolog and Vector representations"""
        # Add fact to both engines
        self.prolog.assert_neural_triples([
            ("bob", "likes", "carol", 0.8)
        ])
        
        self.vector.create_fact(
            subject="bob",
            predicate="likes", 
            object_value="carol",
            confidence=0.8
        )
        
        # Find and validate abstractions
        abstractions = self.abstraction.find_patterns()
        for pred, subj, obj, conf in abstractions:
            # Validate using both engines
            fact = {
                'subject': subj,
                'predicate': pred,
                'object': obj,
                'confidence': conf
            }
            validation = self.validator.validate_abstraction(fact)
            
            # Check both symbolic and vector support
            self.assertGreater(validation['symbolic_support'], 0)
            self.assertGreater(validation['vector_support'], 0)
            
    def test_constraint_learning(self):
        """Test automatic constraint learning"""
        # Add more test data
        self.prolog.assert_neural_triples([
            ("alice", "likes", "carol", 0.8),
            ("bob", "likes", "dave", 0.7),
            ("eve", "dislikes", "frank", 0.8)
        ])
        
        # Run validations to build history
        self.abstraction.apply_abstractions()
        
        # Learn new constraints
        constraints = self.validator.learn_new_constraints(min_support=0.6)
        self.assertTrue(len(constraints) > 0)
        
        # Verify learned constraints are valid
        for constraint in constraints:
            self.assertIn('type', constraint)
            self.assertIn('confidence', constraint)
            self.assertGreaterEqual(constraint['confidence'], 0.6)
            
    def test_error_handling(self):
        """Test handling of invalid abstractions"""
        invalid_pattern = {
            'type': 'relation_pattern',  # Changed to match expected type
            'predicate': 'invalid_pred',
            'subject': 'invalid_subj',
            'object': 'invalid_obj'
        }
        
        validation = self.validator.validate_fact(invalid_pattern)
        self.assertFalse(validation['is_valid'])
        self.assertTrue('issues' in validation)
        self.assertGreater(len(validation['issues']), 0)

if __name__ == '__main__':
    unittest.main()
