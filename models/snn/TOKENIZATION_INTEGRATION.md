# Tokenization Integration Guide

## Overview

This document outlines two important improvements to the SNN architecture:

1. A hybrid vector space approach to prevent hallucinations and false generations
2. Steps to integrate the bidirectional tokenization system across all SNN model classes

## 1. Hybrid Vector Space for Hallucination Prevention

### Problem

The current system may generate hallucinations/false content because:

- The vector symbolic engine lacks statistical grounding
- There's no verification mechanism to check the plausibility of generated content
- Different facets (memory, perception, reasoning) use separate vector spaces

### Solution: Hybrid Vector Space

We'll implement a `HybridVectorSpace` class that combines:

1. **Vector Symbolic Architecture (VSA)** - For compositional reasoning and symbolic operations
2. **Statistical Vector Space** - For real-world statistical grounding

```python
class HybridVectorSpace:
    def __init__(self, symbolic_dim=300, statistical_dim=300):
        # Vector symbolic component
        self.symbolic_engine = VectorSymbolicEngine(dim=symbolic_dim)
        
        # Statistical component
        self.statistical_space = StatisticalVectorSpace(dim=statistical_dim)
        
        # Mapping between spaces
        self.grounding_matrix = None
        
        # Hallucination detection
        self.anomaly_detector = AnomalyDetector()
```

### Implementation Steps

1. Create a statistical vector space that tracks:
   - Token/concept co-occurrence frequencies
   - Contextual likelihood distributions
   - Linguistic coherence metrics

2. Implement a verification mechanism:
   ```python
   def verify_output(self, generated_content, context=None):
       """Verify if generated content is likely to be hallucinated."""
       # Calculate statistical likelihood
       likelihood = self.statistical_space.calculate_likelihood(generated_content)
       
       # Check for logical consistency with symbolic operations
       consistency = self.symbolic_engine.verify_consistency(generated_content)
       
       # Combine scores with context-sensitive weighting
       confidence = self.combine_scores(likelihood, consistency, context)
       
       # Flag as potential hallucination if below threshold
       return confidence, confidence < self.threshold
   ```

3. Create a bidirectional mapping between vector spaces:
   ```python
   def map_between_spaces(self, vector, from_space="symbolic", to_space="statistical"):
       """Map vector between symbolic and statistical spaces."""
       if from_space == "symbolic" and to_space == "statistical":
           return torch.matmul(vector, self.sym_to_stat_matrix)
       elif from_space == "statistical" and to_space == "symbolic":
           return torch.matmul(vector, self.stat_to_sym_matrix)
   ```

### File Structure

New files to implement:
- `models/snn/hybrid_vector_space.py` - Main implementation
- `models/snn/statistical_vector_space.py` - Statistical component
- `models/snn/anomaly_detector.py` - Hallucination detection

## 2. Tokenization Integration Across SNN Classes

### Current State

- The `BidirectionalProcessor` has been integrated into the `EnhancedSpikingCore` base class
- `MemorySNN` has been updated with specific text memory methods
- Other child classes still use their own custom tokenization approaches

### Required Changes

#### 1. Update Class Constructors

All SNN child classes should:
- Accept and pass through tokenization parameters to parent constructor
- Remove redundant tokenization code
- Use parent tokenization methods

```python
def __init__(self, ..., model_type="specific_type", vector_dim=300, bidirectional_processor=None):
    super().__init__(
        ...,
        model_type=model_type,
        vector_dim=vector_dim,
        bidirectional_processor=bidirectional_processor
    )
```

#### 2. Replace Custom Tokenization

The following classes need updates:

| Class             | Current Tokenization      | Required Changes                            |
|-------------------|---------------------------|---------------------------------------------|
| MemorySNN         | MemoryTokenizer           | ✅ Replaced with BidirectionalProcessor     |
| PerceptualSNN     | PerceptualTokenizer       | Replace with inherited bidirectional methods|
| ReasoningSNN      | Custom tokenization funcs | Replace with inherited bidirectional methods|
| DecisionSNN       | Simple word splitting     | Replace with inherited bidirectional methods|
| MetacognitiveSNN  | Custom encoding funcs     | Replace with inherited bidirectional methods|
| AffectiveSNN      | EmotionTokenizer          | Replace with inherited bidirectional methods|

#### 3. Update Text Processing Functions

For each class, replace functions like:
- `tokenize_input()`
- `vectorize_text()`
- `encode_text_to_spikes()`
- `decode_spikes_to_text()`

With calls to parent methods:
- `process_text_input()`
- `process_text_sequence()`
- `generate_text_output()`
- `generate_text_from_sequence()`

#### 4. Specialized Model-Type Tokenization

Each SNN type should use the appropriate `model_type` to enable specialized tokenization:

```python
# In PerceptualSNN
super().__init__(
    ...,
    model_type="perceptual",  # Enables perceptual-specific tokenization
    vector_dim=vector_dim
)

# In ReasoningSNN
super().__init__(
    ...,
    model_type="reasoning",   # Enables reasoning-specific tokenization
    vector_dim=vector_dim
)
```

### Implementation Checklist

- [x] Update MemorySNN tokenization
- [ ] Update PerceptualSNN tokenization
- [ ] Update ReasoningSNN tokenization
- [ ] Update DecisionSNN tokenization
- [ ] Update MetacognitiveSNN tokenization
- [ ] Update AffectiveSNN tokenization
- [ ] Add missing `find_similar_memory()` function to classes that need it
- [ ] Ensure all classes pass `model_type` parameter correctly
- [ ] Add tests to verify tokenization consistency across models

## Next Steps

1. Implement the `HybridVectorSpace` class
2. Update all SNN child classes to use integrated tokenization
3. Create integration tests to verify consistent tokenization across models
4. Measure hallucination rates before and after implementing hybrid vector space 