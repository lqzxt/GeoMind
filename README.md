# GeoMind

A reinforcement learning-based agent for automated lithology classification from well log data, powered by the [agentlightning](https://github.com/microsoft/agent-lightning) framework.

## Overview

This project implements an intelligent agent that combines multiple prediction methods (Neural Networks, Large Language Models) to classify lithology from well log data. The agent uses reinforcement learning to optimize its decision-making process across multiple geological datasets.

### Key Features

- **Reinforcement Learning Training**: Uses VERL (Variational Experience Replay Learning) for continuous improvement
- **Multi-Dataset Support**: Works with 4 different geological datasets (Facies, FORCE-2020, Geolink, SEAM)
- **Geological Reasoning**: Incorporates trend analysis, transition probability checks, and disagreement analysis
- **Comprehensive Logging**: Tracks all LLM calls with detailed prompt/response logging

### Supported Datasets

1. **Facies Dataset**: lithofacies classification
2. **FORCE-2020 Dataset**: lithology classification
3. **Geolink Dataset**: lithology classification
4. **SEAM Dataset**: geological layer classification


## Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for training)
- 32GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the agent using reinforcement learning:

```bash
python train_well_log.py train \
    --model qwen3-4b \
    --epochs 10 \
    --runners 64 \
    --exp_name my_experiment \
    --datasets facies \
    --model_name gpt4ts \
    --gpus 8 \
    --nodes 1 \
    --accelerator npu
```

## Acknowledgments

- Built with [agentlightning](https://github.com/microsoft/agent-lightning) framework
