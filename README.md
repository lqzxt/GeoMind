
# GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation

[![KDD 2027](https://img.shields.io/badge/KDD-2027%20Accepted-blue)]()

> 🎉 **News:** Our paper, *"GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation"*, has been accepted to **ACM SIGKDD 2027 (KDD '27)**!

## 📖 Introduction

**GeoMind** is a novel agentic framework that treats lithology classification not as a static pattern-matching task, but as a **sequential multi-step reasoning process**.

Traditional methods (Classical ML or standard LLMs) often struggle with noisy well-log data ("salt-and-pepper" noise) or fail to adhere to stratigraphic consistency. GeoMind bridges this gap by coordinating specialized tools through a **Planner-Executor-Reflector** architecture, optimized via a novel **Module-Aware Group Relative Policy Optimization (MA-GRPO)** algorithm.


## 🚀 Key Features

* **🤖 Agentic Workflow**:
    * **Planner**: Dynamically decomposes tasks and orchestrates tool invocation based on log characteristics.
    * **Executor**: Utilizing a hierarchical toolkit including *Trend Pattern Extractor*, *Neighbor Vote Aggregator*, and *Neural Probability Interpreter*.
    * **Reflector**: Performs self-correction and resolves conflicting signals using geological constraints (Stratigraphic Sequence Validator).

* **🧠 MA-GRPO Training**:
    * Implements **Module-Aware Group Relative Policy Optimization** to solve the credit assignment problem in multi-step reasoning.
    * Optimizes intermediate reasoning steps (trend analysis, reasoning traces) alongside final prediction accuracy.

## 🛠️ Environment Setup

### Prerequisites

* Linux OS
* Python 3.8+
* CUDA-capable GPU (Recommended: A800/H800 for training)

### Installation

1.  **Clone the repository**

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

We support four standard benchmark datasets. Please ensure your data is preprocessed (windowing, normalization, and outlier removal) as described in the paper.

* **Facies**: Council Grove gas reservoir logs.
* **FORCE**: 2020 Lithology prediction challenge data.
* **GeoLink**: Large-scale North Sea dataset.
* **SEAM**: SEG Wiki open data.

Structure your data directory as follows:

```

data/
├── facies/
├── force/
├── geolink/
└── seam/

```

## 🚅 Training

To train the GeoMind agent using the **MA-GRPO** strategy with process rewards:

```bash
python train_well_log.py train \
    --exp_name geomind_kdd_experiment \
    --model qwen3-4b \
    --model_name gpt4ts \
    --datasets facies \
    --epochs 10 \
    --runners 64 \
    --gpus 8 \
    --nodes 1 \

```
