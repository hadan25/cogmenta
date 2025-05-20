# SNN Training Datasets

This document outlines the training datasets used for each Spiking Neural Network (SNN) model in the Cogmenta Core framework.

## 1. AffectiveSNN (Emotion Recognition)

| Dataset | Status | Description |
|---------|--------|-------------|
| WESAD | Available | Wearable stress and affect dataset with physiological signals |
| DEAP | Available | Database for Emotion Analysis using Physiological signals |
| AFF-WILD | Available | In-the-wild facial expression videos labeled for valence/arousal (video_train/validation/test) |
| MAHNOB-HCI | Not Available | Multimodal dataset (EEG, ECG, facial videos) for emotion analysis |

## 2. PerceptualSNN (Sensory/Feature Detection)

| Dataset | Status | Description |
|---------|--------|-------------|
| CIFAR10-DVS | Available | Event-based version of CIFAR-10 for DVS input |
| DVS Gesture | Available | Event-based gesture recognition dataset |
| LibriSpeech | Available | Large speech dataset for audio-to-spike conversion |
| ESC-50 | Available | Environmental Sound Classification dataset |
| SHD (Spiking Heidelberg Digits) | Not Available | Spike-based digit recognition dataset |
| SSC (Spiking Speech Commands) | Not Available | Spike-based speech recognition dataset |
| TIDIGITS | Not Available | Speech dataset for audio-to-spike conversion |

## 3. MemorySNN (Encoding/Recall/Association)

| Dataset | Status | Description |
|---------|--------|-------------|
| Mini-ImageNet | Available | Common dataset for few-shot learning |
| Something-Something-V2 | Available | Large collection of labeled video snippets showing basic actions |
| Omniglot | Not Available | Few-shot learning dataset with handwritten characters |
| Meta-Dataset | Not Available | Diverse set for testing transfer learning and generalization |
| PAL Tasks | Not Available | Synthetic datasets for paired associative learning |

## 4. DecisionSNN (Action Selection/RL)

| Dataset | Status | Description |
|---------|--------|-------------|
| Something-Something-V2 | Available | Video dataset for action recognition and decision making |
| DVS Gesture | Available | Event-based gesture dataset for decision tasks |
| NeuroGym | Not Available | RL tasks designed for cognitive control with neuroscience relevance |
| SpikeGym | Not Available | Adaptation layer over OpenAI Gym using spike-based encoding |
| DVS-CAR | Not Available | Driving dataset with event-based vision |
| DDD17 | Not Available | Driving dataset for real-world decision-making |

## 5. ReasoningSNN (Symbolic + Hybrid Reasoning)

| Dataset | Status | Description |
|---------|--------|-------------|
| derender_proposals | Available | Dataset for visual reasoning and scene understanding |
| CommonsenseQA | Not Available | Dataset that can be translated into inference chains |
| bAbI Tasks | Not Available | Synthetic but rich set of QA/logic tasks |
| TACRED | Not Available | Relation extraction dataset for inference model supervision |
| ReClor | Not Available | Logical reasoning tasks for inference model supervision |

## 6. StatisticalSNN (Concept Embedding + Generalization)

| Dataset | Status | Description |
|---------|--------|-------------|
| Mini-ImageNet | Available | Can be used for concept embedding and statistical learning |
| Something-Something-V2 | Available | Video dataset for action concept learning |
| ATOMIC2020 | Not Available | Commonsense knowledge graph with cause-effect relations |
| Visual Genome | Not Available | Images with scene graphs, grounding statistical similarity |
| COIN | Not Available | Dataset to bridge symbolic and perceptual categories |
| Conceptual Captions | Not Available | Dataset to bridge symbolic and perceptual categories |

## 7. MetacognitiveSNN (Self-monitoring)

| Dataset | Status | Description |
|---------|--------|-------------|
| WESAD | Available | Can be used for physiological self-monitoring |
| DEAP | Available | Emotion analysis dataset useful for metacognitive tasks |
| Synthetic Confidence Tasks | Not Available | Custom generated tasks for confidence estimation |
| Error Awareness Tasks | Not Available | Tasks designed to test error detection capabilities |

## Dataset Availability and Organization

All datasets marked as "Available" are present in the `training/new_training_files` directory and can be extracted using the `prepare_snn_datasets.py` script. The datasets are organized as follows:

```
training/
├── datasets/                   # Main datasets directory
│   ├── wesad/                  # WESAD dataset (Affective, Metacognitive)
│   ├── deap/                   # DEAP dataset (Affective, Metacognitive)
│   ├── aff_wild/               # AFF-WILD dataset (Affective)
│   ├── cifar10_dvs/            # CIFAR10-DVS dataset (Perceptual)
│   ├── dvs_gesture/            # DVS Gesture dataset (Perceptual, Decision)
│   ├── librispeech/            # LibriSpeech dataset (Perceptual)
│   ├── esc50/                  # ESC-50 dataset (Perceptual)
│   ├── mini_imagenet/          # Mini-ImageNet dataset (Memory, Statistical)
│   ├── something_something_v2/ # Something-Something-V2 dataset (Memory, Decision, Statistical)
│   └── derender_proposals/     # derender_proposals dataset (Reasoning)
├── configs/                    # Dataset configuration files
└── new_training_files/         # Original compressed dataset files
```

## Dataset Mapping to Training Files

| SNN Model | Training File | Datasets Used |
|-----------|---------------|---------------|
| AffectiveSNN | train_affective_snn.py | WESAD, DEAP, AFF-WILD |
| PerceptualSNN | train_perceptual_snn.py | CIFAR10-DVS, DVS Gesture, LibriSpeech, ESC-50 |
| MemorySNN | train_memory_snn.py | Mini-ImageNet, Something-Something-V2 |
| DecisionSNN | train_decision_snn.py | Something-Something-V2, DVS Gesture |
| ReasoningSNN | train_reasoning_snn.py | derender_proposals |
| StatisticalSNN | train_statistical_snn.py | Mini-ImageNet, Something-Something-V2 |
| MetacognitiveSNN | train_metacognitive_snn.py | WESAD, DEAP |

## Dataset Preprocessing Requirements

Each dataset requires specific preprocessing to be compatible with the SNN models:

1. **Event-based datasets** (CIFAR10-DVS, DVS Gesture): Already in spike-compatible format
2. **Image datasets** (Mini-ImageNet): Require conversion to spike trains using rate or temporal coding
3. **Video datasets** (Something-Something-V2, AFF-WILD): Require frame extraction and conversion to spike sequences
4. **Audio datasets** (LibriSpeech, ESC-50): Require spectrogram conversion and spike encoding
5. **Physiological datasets** (WESAD, DEAP): Require temporal alignment and spike conversion

The `utils_snn_datasets.py` file contains utilities for these preprocessing steps.

## Dataset Extraction and Preparation

To extract and prepare all datasets, run:

```bash
python training/prepare_snn_datasets.py
```

This will extract all available datasets from `training/new_training_files`, organize them into the appropriate directory structure, and create configuration files for each SNN type.

To verify the dataset organization, run:

```bash
python training/test_snn_datasets.py
```

This will check if all datasets are properly extracted and organized, and print a summary of available datasets for each SNN type.

## Dataset References

- **WESAD**: Wearable Stress and Affect Detection dataset
  - Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection.

- **DEAP**: Database for Emotion Analysis using Physiological Signals
  - Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., ... & Patras, I. (2011). DEAP: A database for emotion analysis using physiological signals.

- **CIFAR10-DVS**: Event-based version of CIFAR-10 for DVS input
  - Li, H., Liu, H., Ji, X., Li, G., & Shi, L. (2017). CIFAR10-DVS: An event-stream dataset for object classification.

- **DVS Gesture**: Event-based gesture recognition dataset
  - Amir, A., Taba, B., Berg, D., Melano, T., McKinstry, J., Di Nolfo, C., ... & Flickner, M. (2017). A low power, fully event-based gesture recognition system.

- **LibriSpeech**: Large speech dataset for audio-to-spike conversion
  - Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). Librispeech: an ASR corpus based on public domain audio books.

- **ESC-50**: Environmental Sound Classification dataset
  - Piczak, K. J. (2015). ESC: Dataset for environmental sound classification.

- **Mini-ImageNet**: Common dataset for few-shot learning
  - Vinyals, O., Blundell, C., Lillicrap, T., & Wierstra, D. (2016). Matching networks for one shot learning.

- **Something-Something-V2**: Large collection of labeled video snippets showing basic actions
  - Goyal, R., Kahou, S. E., Michalski, V., Materzynska, J., Westphal, S., Kim, H., ... & Bengio, Y. (2017). The "something something" video database for learning and evaluating visual common sense.

- **AFF-WILD**: In-the-wild facial expression videos labeled for valence/arousal
  - Zafeiriou, S., Kollias, D., Nicolaou, M. A., Papaioannou, A., Zhao, G., & Kotsia, I. (2017). Aff-wild: Valence and arousal 'in-the-wild' challenge.

- **derender_proposals**: Dataset for visual reasoning and scene understanding
  - Proposed by the Cogmenta Core team for reasoning tasks. 