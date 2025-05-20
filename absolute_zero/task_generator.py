import random
import numpy as np
from typing import Dict, Tuple, Any, List

# These imports should be properly configured based on your project structure
# The actual import paths may need to be adjusted based on where these modules live
try:
    # Try importing directly if they're in the Python path
    from models.symbolic.prolog_engine import PrologEngine as SymbolicEngine
    from models.symbolic.vector_symbolic import VectorSymbolicEngine as VSAEngine
except ImportError:
    # Fallback - these will be provided via constructor if imports fail
    pass

# Mock VSA engine for fallback
class MockVSAEngine:
    def encode(self, text):
        # Simple hash-based encoding for testing
        return np.array([hash(word) % 100 / 100.0 for word in text.split()])
    
    def similarity(self, vec1, vec2):
        # Simple cosine similarity
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Make vectors the same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

class CogmentaTaskGenerator:
    def __init__(self, symbolic_engine=None, vsa_engine=None):
        """Initialize the task generator with symbolic and VSA engines
        
        Args:
            symbolic_engine: Symbolic reasoning engine (e.g., Prolog-based)
            vsa_engine: Vector Symbolic Architecture engine for encoding
        """
        # Use provided symbolic engine or create a simple mock
        if symbolic_engine is None:
            class MockSymbolicEngine:
                def deduce(self, rules, facts):
                    return "mock_conclusion"
                    
                def verify(self, rules, facts, conclusion):
                    return random.random() > 0.3
            self.symbolic_engine = MockSymbolicEngine()
        else:
            self.symbolic_engine = symbolic_engine
        
        # Use provided VSA engine or fallback to mock implementation
        self.vector_symbolic = vsa_engine if vsa_engine is not None else MockVSAEngine()
        
        self.complexity_level = 1.0
        self.task_types = ['deduction', 'abduction', 'induction', 'pattern_recognition']
        
    def generate_task(self) -> Dict:
        """Generate a task of random type based on current complexity"""
        task_type = random.choice(self.task_types)
        
        if task_type == 'deduction':
            task = self.generate_deduction_task()
        elif task_type == 'abduction':
            task = self.generate_abduction_task()
        elif task_type == 'induction':
            task = self.generate_induction_task()
        else:  # pattern_recognition
            task = self.generate_pattern_task()
            
        # Add task metadata
        task['type'] = task_type
        task['complexity'] = self.complexity_level
        
        return task
        
    def generate_deduction_task(self) -> Dict:
        """Given rules + facts, predict conclusion"""
        # Example: parent(A,B), parent(B,C) -> ancestor(A,C)
        rules = self.sample_rules()
        facts = self.sample_facts()
        conclusion = self.deduce_from_rules(rules, facts)
        return {"input": {"rules": rules, "facts": facts}, "output": conclusion}
        
    def generate_abduction_task(self) -> Dict:
        """Given rules + conclusion, find facts"""
        rules = self.sample_rules()
        conclusion = self.sample_conclusion()
        facts = self.abduce_from_rules(rules, conclusion)
        return {"input": {"rules": rules, "conclusion": conclusion}, "output": facts}
        
    def generate_induction_task(self) -> Dict:
        """Given examples, induce pattern/rule"""
        examples = self.generate_examples()
        pattern = self.induce_pattern(examples)
        return {"input": {"examples": examples}, "output": pattern}
    
    def generate_pattern_task(self) -> Dict:
        """Generate pattern recognition task"""
        # Example: sequence prediction, pattern matching
        pattern_length = int(3 + self.complexity_level * 2)
        sequence = self.generate_sequence(pattern_length)
        next_element = self.calculate_next_element(sequence)
        return {"input": {"sequence": sequence}, "output": {"next_element": next_element}}
        
    def increase_complexity(self):
        """Increase the complexity of generated tasks"""
        self.complexity_level = min(10.0, self.complexity_level * 1.1)
        
    def decrease_complexity(self):
        """Decrease the complexity of generated tasks"""
        self.complexity_level = max(0.5, self.complexity_level * 0.9)
    
    def set_complexity(self, complexity: float):
        """Set the complexity level directly"""
        self.complexity_level = max(0.5, min(10.0, complexity))
    
    # Helper methods
    def sample_rules(self) -> List[str]:
        """Sample rules based on complexity level"""
        num_rules = max(1, int(self.complexity_level))
        # Simplified implementation
        rule_templates = [
            "parent(X,Y) :- father(X,Y)",
            "parent(X,Y) :- mother(X,Y)",
            "ancestor(X,Y) :- parent(X,Y)",
            "ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)",
            "sibling(X,Y) :- parent(Z,X), parent(Z,Y), X != Y"
        ]
        return random.sample(rule_templates, min(num_rules, len(rule_templates)))
    
    def sample_facts(self) -> List[str]:
        """Sample facts based on complexity level"""
        num_facts = max(2, int(self.complexity_level * 2))
        # Simplified implementation
        fact_templates = [
            "father(john, bob)",
            "mother(mary, bob)",
            "father(bob, alice)",
            "mother(jane, tim)",
            "father(tim, sam)"
        ]
        return random.sample(fact_templates, min(num_facts, len(fact_templates)))
        
    def sample_conclusion(self) -> str:
        """Sample a conclusion to abduce from"""
        conclusions = [
            "ancestor(john, alice)",
            "sibling(bob, tim)",
            "parent(mary, bob)"
        ]
        return random.choice(conclusions)
        
    def deduce_from_rules(self, rules: List[str], facts: List[str]) -> str:
        """Given rules and facts, deduce a conclusion"""
        # This would typically use the symbolic engine
        # Simplified implementation
        if "father(john, bob)" in facts and "father(bob, alice)" in facts:
            return "ancestor(john, alice)"
        return "no_conclusion"
    
    def abduce_from_rules(self, rules: List[str], conclusion: str) -> List[str]:
        """Given rules and conclusion, abduce facts"""
        # Simplified implementation
        if conclusion == "ancestor(john, alice)":
            return ["father(john, bob)", "father(bob, alice)"]
        return ["no_facts_found"]
    
    def generate_examples(self) -> List[Dict]:
        """Generate examples for induction task"""
        # Simplified implementation
        num_examples = max(3, int(self.complexity_level * 2))
        examples = []
        for i in range(num_examples):
            x = random.randint(1, 10)
            examples.append({"input": x, "output": x * 2})
        return examples
    
    def induce_pattern(self, examples: List[Dict]) -> Dict:
        """Induce a pattern from examples"""
        # Simplified implementation
        if all(example["output"] == example["input"] * 2 for example in examples):
            return {"type": "multiplication", "factor": 2}
        return {"type": "unknown"}
    
    def generate_sequence(self, length: int) -> List[int]:
        """Generate a sequence based on complexity"""
        if self.complexity_level < 3:
            # Arithmetic sequence
            start = random.randint(1, 10)
            diff = random.randint(1, 5)
            return [start + i * diff for i in range(length)]
        else:
            # Fibonacci-like sequence
            if random.random() < 0.5:
                a, b = random.randint(1, 5), random.randint(1, 5)
                sequence = [a, b]
                for i in range(length - 2):
                    sequence.append(sequence[-1] + sequence[-2])
                return sequence
            else:
                # Quadratic sequence
                start = random.randint(1, 5)
                return [start * (i+1)**2 for i in range(length)]
    
    def calculate_next_element(self, sequence: List[int]) -> int:
        """Calculate the next element in a sequence"""
        if len(sequence) < 2:
            return 0
            
        # Try to detect pattern
        if len(sequence) >= 3:
            # Check arithmetic sequence
            if sequence[1] - sequence[0] == sequence[2] - sequence[1]:
                diff = sequence[1] - sequence[0]
                return sequence[-1] + diff
                
            # Check geometric sequence
            if sequence[0] != 0 and sequence[1] / sequence[0] == sequence[2] / sequence[1]:
                ratio = sequence[1] / sequence[0]
                return int(sequence[-1] * ratio)
                
            # Check Fibonacci-like
            if sequence[2] == sequence[0] + sequence[1]:
                return sequence[-1] + sequence[-2]
                
        # Default: repeat the pattern (for simple sequences)
        return sequence[-1] + (sequence[-1] - sequence[-2])