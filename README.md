# Action-Level Reinforcement Learning Reward Models from Human Feedback

This repository contains the official code, environment configurations, and reproduction instructions for the paper:  
**"Action-Level Reinforcement Learning Reward Models from Human Feedback"** (*Anonymous Submission*).

---

## Overview

Reinforcement Learning in multi-stage environments is fundamentally constrained by sparse rewards and delayed feedback. Standard trajectory-level RLHF suffers from credit assignment ambiguity and cold-start exploration bottlenecks. This framework introduces an action-level reward modeling paradigm:

* **Reward Model (RM):** A compact encoder-only transformer (**DeBERTa-v3-small**, 44M parameters) fine-tuned via **LoRA** ($r=8, \alpha=16$) to evaluate serialized state-action transitions ($\langle s, a \rangle \rightarrow r$) into discrete, calibrated reward tiers.
* **Anti-Farming Safeguards:** Synthetic negative transition data augmentation explicitly penalizes cyclical or regressive behaviors (e.g., repeated drop-pickup farming loops).
* **Policy Architecture:** Proximal Policy Optimization (PPO) paired with specialized perception:
  * 3-layer NatureCNN observation encoder for sequential grid navigation (`MiniGrid-DoorKey-8x8`).
  * 8-bit multi-head relational comparator architecture for compositional, language-conditioned manipulation (`MiniGrid-PutNear-8x8-N3`).
* **LRU Semantic Cache:** Sub-millisecond step-level evaluation during online RL training.

---

## Hardware & Requirements

The reward model is based on **DeBERTa-v3-small** (44M parameters / ~170 MB), eliminating the need for heavy multi-GPU quantization (`bitsandbytes`). Both RM fine-tuning and policy training run natively on:
* **CUDA GPUs:** Linux / Cloud clusters.
* **Apple Silicon (M1/M2/M3/M4):** Accelerated via Metal Performance Shaders (`mps`).
* **CPU:** Reward model fine-tuning completes in under 10 minutes on standard CPUs.

---

## Installation & Setup

1. **Clone the repository and enter the directory:**
```bash
git clone <repository-url>
cd action-level-rm
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration System

All experimental hyperparameters are modularly managed under `configs/`:

| Config File | Benchmark | Description |
| :--- | :--- | :--- |
| `configs/doorkey_config.yaml` | DoorKey-8x8 | Master config & RM fine-tuning setup |
| `configs/doorkey_rm_ppo.yaml` | DoorKey-8x8 | Action-Level RM-guided PPO |
| `configs/doorkey_baseline_ppo.yaml` | DoorKey-8x8 | Unshaped sparse reward baseline |
| `configs/doorkey_intrinsic_ppo.yaml` | DoorKey-8x8 | Count-based curiosity novelty baseline |
| `configs/doorkey_reward_shaping_ppo.yaml` | DoorKey-8x8 | Heuristic milestone reward shaping baseline |
| `configs/putnear_config.yaml` | PutNear-8x8-N3 | Master config & PutNear RM fine-tuning setup |
| `configs/putnear_rm_ppo.yaml` | PutNear-8x8-N3 | Action-Level RM-guided PPO |
| `configs/putnear_baseline_ppo.yaml` | PutNear-8x8-N3 | Unshaped sparse reward baseline |
| `configs/putnear_intrinsic_ppo.yaml` | PutNear-8x8-N3 | Count-based curiosity novelty baseline |
| `configs/putnear_reward_shaping_ppo.yaml` | PutNear-8x8-N3 | Heuristic milestone reward shaping baseline |

---

## Reproduction Guide

### Part 1: MiniGrid-DoorKey-8x8 Benchmark

#### 1. Fine-Tune the DeBERTa-v3-small Reward Model:
```bash
python -m src.train_rm --config configs/doorkey_config.yaml
```
*Outputs: Fine-tuned LoRA adapter weights saved to `./reward-model-lora-doorkey/`.*

#### 2. Train RL Agents (1,000,000 Timesteps):
```bash
# Action-Level RM-PPO (Ours)
python -m src.train_rl --config configs/doorkey_rm_ppo.yaml

# Baselines
python -m src.train_rl --config configs/doorkey_baseline_ppo.yaml
python -m src.train_rl --config configs/doorkey_intrinsic_ppo.yaml
python -m src.train_rl --config configs/doorkey_reward_shaping_ppo.yaml
```

#### 3. Evaluate Held-Out Benchmark (Table 1):
```bash
python -m src.evaluate --config configs/doorkey_rm_ppo.yaml --seeds 100 --start-seed 5000
```

---

### Part 2: MiniGrid-PutNear-8x8-N3 Benchmark (Language-Conditioned Manipulation)

#### 1. Fine-Tune the PutNear Reward Model:
```bash
python -m src.train_rm --config configs/putnear_config.yaml
```
*Outputs: Fine-tuned LoRA adapter weights saved to `./reward-model-lora-putnear/`.*

#### 2. Train RL Agents (2,000,000 Timesteps):
```bash
# Action-Level RM-PPO (Ours)
python -m src.train_rl --config configs/putnear_rm_ppo.yaml

# Baselines
python -m src.train_rl --config configs/putnear_baseline_ppo.yaml
python -m src.train_rl --config configs/putnear_intrinsic_ppo.yaml
python -m src.train_rl --config configs/putnear_reward_shaping_ppo.yaml
```

#### 3. Evaluate Held-Out Benchmark (Table 2):
```bash
python -m src.evaluate --config configs/putnear_rm_ppo.yaml --seeds 100 --start-seed 5000
```

---

## Repository Structure

```
action-level-rm/
├── configs/                     # Modular experiment configurations
│   ├── doorkey_config.yaml      # Master DoorKey configuration
│   ├── doorkey_rm_ppo.yaml      # DoorKey Action-Level RM-PPO
│   ├── doorkey_baseline_ppo.yaml
│   ├── doorkey_intrinsic_ppo.yaml
│   ├── doorkey_reward_shaping_ppo.yaml
│   ├── putnear_config.yaml      # Master PutNear configuration
│   ├── putnear_rm_ppo.yaml      # PutNear Action-Level RM-PPO
│   ├── putnear_baseline_ppo.yaml
│   ├── putnear_intrinsic_ppo.yaml
│   └── putnear_reward_shaping_ppo.yaml
├── data/                        # Stratified transition datasets (CSV)
│   ├── doorkey_reward_dataset_balanced.csv
│   └── putnear_reward_dataset_balanced.csv
├── reward-model-lora-doorkey/   # Pre-trained LoRA adapter weights (DoorKey)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── label_mapping.json
│   └── tokenizer_config.json
├── reward-model-lora-putnear/   # Pre-trained LoRA adapter weights (PutNear)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── label_mapping.json
│   └── tokenizer_config.json
├── src/                         # Core Python package
│   ├── config.py                # YAML loader and runtime device resolver
│   ├── features_extractor.py    # CNN & Relational Slot policy extractors
│   ├── prompt_utils.py          # DoorKey state-action text serializer
│   ├── prompt_utils_putnear.py  # PutNear state-action text serializer
│   ├── rm_model.py              # DeBERTa reward model inference & LRU cache
│   ├── train_rm.py              # DeBERTa LoRA fine-tuning pipeline
│   ├── train_rl.py              # PPO policy training pipeline
│   ├── wrappers.py              # DoorKey Gym wrappers
│   ├── wrappers_putnear.py      # PutNear Gym wrappers
│   └── evaluate.py              # Benchmark rollout evaluation & metrics
├── LICENSE                      # Open-source MIT License
├── README.md                    # Project documentation & reproduction instructions
└── requirements.txt             # Pinned dependency specifications
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
