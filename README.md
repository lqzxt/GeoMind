# GeoMind: A Context-Aware Agentic Workflow for Lithology Classification with Reasoned Tool Invocation

![KDD 2027](https://img.shields.io/badge/KDD-2027%20Accepted-blue)

> 🎉 **News:** Our paper, **"GeoMind: A Context-Aware Agentic Workflow for Lithology Classification with Reasoned Tool Invocation,"** has been accepted to **KDD 2027**.
>
> **Note:** This repository contains the official implementation of GeoMind.

## 📖 Overview

**GeoMind** is a context-aware, tool-augmented agentic framework for **lithology classification from well logs**. Instead of treating multivariate well-log interpretation as a static, single-step prediction task, GeoMind formulates it as a **sequential reasoning and decision-making process**.

Existing numerical classifiers can be sensitive to local noise and produce geologically implausible, fragmented predictions, while standalone language models may struggle with fine-grained numerical signals. GeoMind integrates numerical predictions, retrieved neighborhood evidence, geological trend interpretation, and stratigraphic constraints through a **Planner–Executor–Reflector** workflow.

The framework combines **domain-adaptive supervised fine-tuning (SFT)** with **process-supervised reinforcement learning**. Its **Module-Aware Group Relative Policy Optimization (MAPO)** method provides module-specific learning signals for intermediate reasoning, helping address credit assignment across multi-step agent interactions.

## 🚀 Key Features

- **Adaptive agentic workflow:** A Planner dynamically chooses analytical tools; an Executor produces evidence and candidate predictions; and a Reflector resolves disagreements and revises predictions under geological constraints.
- **Hierarchical geological toolkit:** Seven specialized tools span **perception**, **reasoning**, and **analysis**, enabling evidence-grounded lithology interpretation.
- **Fine-grained process rewards:** Dedicated rewards supervise trend quality, intermediate LLM classification accuracy, and reflection-based correction, rather than relying only on final correctness.
- **MAPO training:** Module-local group-relative advantages and state-dispersion-based attenuation support more targeted, stable policy updates, with KL regularization against a reference policy.
- **Inference-consistent training:** Well-level **K-fold stacking** supplies out-of-fold numerical predictions and prevents held-out well information from leaking into retrieval or stratigraphic statistics used during agent training.

## 🧩 GeoMind Framework

### Planner–Executor–Reflector

1. **Planner:** Examines well-log characteristics and constructs an adaptive sequence of tool invocations.
2. **Executor:** Invokes specialized tools to derive numerical and semantic evidence, generate candidate lithology predictions, and detect potential inconsistencies.
3. **Reflector:** Integrates neural, neighbor-based, and LLM predictions with confidence and geological validation signals to produce a final classification and explanation.

### Toolkit

| Layer | Tool | Purpose |
| --- | --- | --- |
| Perception | Case Retriever | Retrieves similar historical well-log windows and their lithology labels. |
| Perception | Trend Pattern Extractor | Converts log curves into depth-aware descriptions of trends, turning points, and magnitude changes. |
| Reasoning | Neighbor Vote Aggregator | Forms lithology confidence estimates via similarity-weighted neighbor voting. |
| Reasoning | Neural Probability Interpreter | Converts neural classifier probabilities into interpretable evidence. |
| Reasoning | Semantic Reasoning Engine | Combines well-log measurements, trend descriptions, and retrieved cases to infer candidate lithology labels. |
| Analysis | Consensus Conflict Scanner | Detects disagreements among neural, neighbor-based, and LLM predictions. |
| Analysis | Stratigraphic Sequence Validator | Flags implausible lithological transitions using a transition model learned from training labels. |

## 🧠 Process-Supervised Training with MAPO

GeoMind uses **domain-adaptive SFT followed by MAPO-based agentic reinforcement learning**.

**Domain-adaptive SFT.** A Qwen3-4B model is adapted to lithology interpretation with geological instruction data.

**Process reward design.** MAPO uses three complementary signals:

| Reward | Supervised behavior |
| --- | --- |
| Trend Analysis Reward | Accurate, complete, clear, and depth-aligned trend descriptions, together with classification correctness. |
| LLM Classification Accuracy Reward | Correct intermediate lithology predictions by the semantic reasoning engine. |
| Reflection Correction Reward | Correctly resolving disagreements between candidate predictors when the correct label is among the candidates. |

The **Planner** is optimized using terminal classification correctness. MAPO computes **module-specific group-relative advantages** rather than sharing one trajectory-level return across all steps. A **state-dispersion gate** down-weights updates when sampled trajectories reach substantially different module contexts, and a KL term constrains deviation from the reference policy.

For inference-consistent policy learning, numerical predictor outputs used in RL observations are generated **out of fold at the well level**. The case-retrieval index and stratigraphic transition statistics are also built without the held-out wells for each fold.

## 📊 Benchmark Results

GeoMind is evaluated on four public well-log benchmarks. The table reports **weighted F1** as presented in the paper.

| Dataset | GeoMind (Weighted F1) |
| --- | ---: |
| SEAM | **0.8821** |
| Facies | **0.4582** |
| FORCE | **0.3741** |
| GeoLink | **0.4401** |

GeoMind achieves the highest weighted F1 among the evaluated baselines on all four datasets. It also reduces anomalously short lithology fragments by **up to 27.9%** relative to XGBoost in the reported analysis.

## 🛠️ Environment Setup

1. **Clone this repository.**
2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 📂 Data Preparation

| Dataset | Source | Wells (Train / Val / Test) | Samples | Channels | Classes |
| --- | --- | ---: | ---: | ---: | ---: |
| [SEAM](https://wiki.seg.org/wiki/Open_data#Well_logs) | SEG Wiki open well-log resources | 3 / 1 / 1 | 7,092 | 9 | 7 |
| [Facies](https://www.kaggle.com/datasets/imeintanis/well-log-facies-dataset) | Council Grove gas reservoir well logs | 4 / 1 / 2 | 3,164 | 7 | 9 |
| [FORCE](https://www.kaggle.com/datasets/faresazzam/well-logs-dataset-for-machine-learning) | FORCE 2020 lithology prediction context | 4 / 2 / 5 | 52,766 | 9 | 5 |
| [GeoLink](https://github.com/LukasMosser/geolink_dataset) | GEOLINK-S2 North Sea well logs | 80 / 23 / 25 | 580,205 | 9 | 11 |

Organize the prepared datasets as follows:

```text
data/
├── facies/
├── force/
├── geolink/
└── seam/
```

## 🚅 Training


```bash
python train_well_log.py train \
    --exp_name geomind_kdd27_facies \
    --model qwen3-4b \
    --model_name gpt4ts \
    --datasets facies \
    --gpus 4 \
    --nodes 1
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{cheng2026geomind,
  title={GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation},
  author={Cheng, Mingyue and Zhou, Yitong and Wang, Jiahao and Mao, Qingyang and Liu, Qi},
  journal={arXiv preprint arXiv:2604.21501},
  year={2026}
}
```
