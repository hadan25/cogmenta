# SNN Development Roadmap

This document outlines the step-by-step roadmap for developing and integrating the Spiking Neural Network (SNN) models in the Cogmenta Core framework.

## Core Development Steps

1. **Fix Tokenization/Vectorization for each model**
   - [x] Ensure consistent information encoding across all SNN models
   - [x] Standardize input/output formats for cross-model compatibility
   - [x] Optimize for efficiency in spike-based representations
   - [x] Create a Unified Interface (bidirectional_encoding.py)
   - [x] Standardize Model Constructors
   - [x] Model-specific Implementations (all SNNs use unified interface)
   - [x] Update the SNNVectorEngine
   - [x] Integration with Advanced Tokenizer
   - [x] Standardize Spike Encoding/Decoding
   - [x] Vectorization Consistency
   - [x] Cross-model Testing
   - **[x] Implement Quantitative Fidelity Gate**
     - Explicit, automated, and enforced measurement of information preservation across encode–decode–re-encode cycles
     - Hard numerical targets for reconstruction error, cosine similarity, or semantic preservation (e.g., cosine similarity > 0.999)
     - Test suite for round-trip fidelity for all modalities (text, image, audio, VSA vector)
     - Strict pass/fail thresholds; pipeline must not proceed to pre-training if fidelity is below threshold

2. **Work on the Encoding/Decoding (Making it Adaptive)**
   - [x] Implement learnable encoding/decoding mechanisms (AdaptiveEncodingLayer, AdaptiveDecodingLayer)
   - [x] Develop adaptive conversion between different data modalities
   - [x] Ensure efficient information transfer between modules
   - **[x] Implement Explicit Learning Protocol for Adaptiveness**
     - Define and document the loss function for adaptive layers (e.g., MSE, negative information retention reward)
     - Ensure adaptive layers are included in the optimizer and updated during training
     - Document when and how adaptiveness is enabled (pre-training, joint training, etc.)
     - Consider making spike encoder/decoder or bidirectional processor parameters learnable if not already
     - Mechanism for freezing/unfreezing adaptiveness based on performance

3. **Define the Unified System Objective/Reward (CRITICAL)**
   - [x] Establish global optimization criteria for the entire system
   - [x] Create hierarchical reward structure that supports component-level and system-level goals
   - [x] Implement mechanisms to propagate reward signals across all models
   - [x] Support for different training approaches (pretraining, hybrid, absolute zero)
   - **[x] Implement Reward Propagation Across Paradigms**
     - Draft and document a protocol for how symbolic rewards/errors influence neural learning (and vice versa)
     - For each symbolic component, define how its success/failure is measured numerically and mapped to a reward signal
     - Implement a minimal "reward adapter" or "bridge" for at least one symbolic–neural interface as a proof of concept
     - Sketch (diagram/pseudocode) how errors/rewards are back-propagated or otherwise influence both sides
     - Consider differentiable symbolic methods, RL with symbolic feedback, or hybrid approaches

4. **Train the models (individually pre-train)**
   - Develop model-specific training procedures
   - Use appropriate datasets for each SNN type
   - Establish performance baselines for each model

5. **Integrate and Train Together**
   - Focus on Interfaces & Global Reward
   - Ensure proper communication between modules
   - Optimize for the unified objective defined in step 3

6. **Use hybrid training with absolute zero to drive integration**
   - Implement temperature-based training schedule
   - Balance exploration vs. exploitation during integration
   - Monitor convergence metrics during hybrid training

7. **Explicitly train interfaces/bridges using modular_self_play**
   - Develop specialized training for inter-module communication
   - Implement self-play mechanisms to optimize information transfer
   - Test interface performance with controlled experiments

8. **Ensure learning signals propagate across paradigms**
   - Implement gradient flow mechanisms between different SNN types
   - Verify signal integrity across architectural boundaries
   - Optimize for global objective alignment

9. **Rigorous, Targeted Integration Testing & Metrics (CRITICAL)**
   - Focus on semantic consistency, logical coherence, and inter-module communication
   - Include tests for AnomalyDetector and KnowledgeValidator effectiveness
   - Develop quantitative metrics for integration quality

10. **Prompting and General Validation Testing**
    - Test system with diverse input prompts and scenarios
    - Validate against established benchmarks
    - Identify edge cases and failure modes

11. **Retrain based on results**
    - Implement targeted retraining for underperforming components
    - Fine-tune system parameters based on validation metrics
    - Optimize problematic interfaces

12. **Iterate, Iterate, Iterate**
    - Establish continuous improvement cycle
    - Implement systematic A/B testing protocol
    - Document progress and insights at each iteration

## Implementation Priorities

The roadmap prioritizes these critical elements:
1. Adaptive Encoding/Decoding (Step 2)
2. Unified System Objective (Step 3)
3. Interface Training (Step 7)
4. Integration Testing (Step 9)

**Before pre-training, the following must be completed:**
- [x] Quantitative Fidelity Gate (Step 1)
- [x] Concrete Adaptiveness Mechanism (Step 2)
- [x] Reward Propagation Across Paradigms (Step 3)

---

## Summary Table

| Step | What's Present | What's Missing/Needs Emphasis | Action |
|------|---------------|-------------------------------|--------|
| 1. Tokenization/Vectorization | Standardization, cross-model testing | Quantitative fidelity gate, strict metrics | Implement fidelity test suite, enforce thresholds |
| 2. Adaptive Encoding/Decoding | Learnable layers, adaptive processor | Explicit learning protocol, optimizer integration | Define/update training loop, document adaptiveness |
| 3. Unified Reward System | Hierarchical rewards, propagation | Cross-paradigm reward propagation, symbolic–neural bridge | Draft protocol, implement reward adapter/bridge |

Success of the overall system depends heavily on proper implementation of these key steps, with particular emphasis on the unified objective definition and rigorous integration testing. 