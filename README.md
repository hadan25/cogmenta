# Cogmenta Core

Cogmenta Core is a comprehensive cognitive architecture that integrates multiple reasoning capabilities, including conceptual knowledge, natural language inference, logical reasoning, and commonsense reasoning.

## Training System

The Cogmenta training system is organized around specialized trainers for different cognitive capabilities, orchestrated by a central training plan.

### Running the Training System

We provide several entry points to run the training system:

1. **Running the full training pipeline**:
   ```bash
   python run_training.py
   ```

2. **Running the ConceptNet training only**:
   ```bash
   python run_conceptnet.py --max_facts 1000 --epochs 3
   ```

3. **Running individual trainers sequentially**:
   ```bash
   python run_all_trainers.py --epochs 1
   ```
   
   You can also specify which trainers to run:
   ```bash
   python run_all_trainers.py --trainers concept_net,nli
   ```

### Training Structure

The training system consists of the following components:

1. **Training Plan** (`training/training_plan.py`): Orchestrates multiple training phases with dependency management.

2. **Specialized Trainers**:
   - **ConceptNetTrainer**: Trains on conceptual knowledge and relationships using ConceptNet data
   - **NLITrainer**: Natural language inference training using SNLI dataset
   - **LogiQATrainer**: Logical reasoning over text using LogiQA dataset
   - **AtomicTrainer**: Commonsense reasoning about everyday events
   - **RuleTakerTrainer**: Rule-based reasoning with explanation

3. **Entry Points**:
   - `run_training.py`: Main entry point for the full training pipeline
   - `run_conceptnet.py`: Entry point for ConceptNet training
   - `run_all_trainers.py`: Run all trainers sequentially

### Training Output

The training system generates output in the specified output directories:
- Training metrics (accuracy, loss, etc.)
- Checkpoints of trained models
- Evaluation results
- Visualizations of training progress

For more details on the training system, see [training/README.md](training/README.md).

## Overview

Cogmenta combines multiple AI paradigms into a cohesive system:

- **Symbolic Reasoning**: Logic-based rule systems and knowledge representation
- **Vector Symbolic Architecture**: Distributed representations for concepts and relationships
- **Neural Networks**: Learning and pattern recognition
- **Spiking Neural Networks**: Event-based, biologically-inspired processing

These components work together to enable capabilities such as:

- Multi-step logical reasoning with explanation
- Common-sense reasoning about everyday situations
- Temporal reasoning over events and actions
- Conceptual abstraction and analogy-making
- Self-reflection and metacognition

## Key Components

- **Symbolic System**: Prolog-based symbolic reasoning engine
- **Vector Symbolic Engine**: Implementation of Vector Symbolic Architecture (VSA)
- **Knowledge Components**:
  - ConceptNet integration for common-sense knowledge
  - ATOMIC integration for social and event reasoning
  - RuleTaker for logical reasoning
- **Neural-Symbolic Bridge**: Bidirectional translation between symbolic and neural representations
- **Thought Tracing**: System for tracking and visualizing cognitive processes

## Getting Started

### ConceptNet Training

To set up the ConceptNet training environment:

```bash
# Windows
setup_conceptnet_training.bat

# Linux/macOS
python -m training.run_concept_training --sample_data
```

See `training/README_CONCEPTNET.md` for detailed documentation on using the ConceptNet training module.

### Full System Training

For training the complete Cogmenta system:

```bash
python -m training.quickstart_training
```

## Project Structure

- `api/`: API interfaces for external integration
- `cognitive/`: Core cognitive processes
- `conceptual/`: Conceptual knowledge and processing
- `models/`: Symbolic, neural, and hybrid models
- `processing/`: Data processing utilities
- `training/`: Training systems and datasets
- `visualization/`: Visualization tools

## Development Roadmap

See `ROADMAP.md` for the current development roadmap and status.

## License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Setup and Dependencies

1. Install Python 3.8+ and pip.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare datasets:
   - For NLI (SNLI): Place the SNLI zip file (`snli_1.0.zip`) in the project root and run the processing script:
     ```bash
     python training/datasets/nli/process_snli.py
     ```
   - Other datasets: See respective dataset preparation scripts in `training/datasets/`. 