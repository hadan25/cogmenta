"""
Test Runner for Integration Tests

This script runs the OpenAI model integration tests with the correct Python path setup.
"""
import sys
import os
import unittest

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# Import the test modules
import test_open_source_integrations
import test_scientific_reasoning
import test_individual_components
import test_bayesian_validation

# Create a test suite with all the tests
def create_test_suite():
    suite = unittest.TestSuite()
    
    # Add test classes using loader
    loader = unittest.TestLoader()
    
    # Add all tests from each test module
    suite.addTests(loader.loadTestsFromTestCase(test_open_source_integrations.TestOpenSourceIntegrations))
    suite.addTests(loader.loadTestsFromTestCase(test_scientific_reasoning.TestScientificReasoning))
    suite.addTests(loader.loadTestsFromTestCase(test_individual_components.TestTransformerIntegrations))
    suite.addTests(loader.loadTestsFromTestCase(test_bayesian_validation.TestBayesianValidation))
    
    return suite

if __name__ == '__main__':
    # Run the tests
    print("Running integration tests for open-source models...")
    print("Testing SciBERT, PyReason, PyMC, and GPT-J integrations...")
    
    # Create and run the test suite
    test_suite = create_test_suite()
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Report the results
    print(f"\nTest Results:")
    print(f"  Run: {result.testsRun}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    
    # Print integration status
    if len(result.errors) == 0 and len(result.failures) == 0:
        print("\n✅ All integrations are working correctly!")
        print("   - SciBERT scientific text understanding ✓")
        print("   - PyReason logical inference ✓")
        print("   - PyMC Bayesian uncertainty modeling ✓")
        print("   - GPT-J constrained text generation ✓")
        print("\nThe neuro-symbolic bridge successfully connects all components.")
    else:
        print("\n❌ Some integrations failed. Please check the test results.")
    
    # Exit with appropriate status code
    sys.exit(len(result.errors) + len(result.failures)) 