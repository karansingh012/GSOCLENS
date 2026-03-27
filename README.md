# Gravitational Lens Detection using Deep Learning and Domain Adaptation

## Overview
This project focuses on **binary gravitational lens classification** from astronomical images (**Lens vs Non-Lens**) using deep learning.  
It targets two core challenges in practical astronomy ML pipelines:

1. **Class imbalance** between lens and non-lens samples  
2. **Domain shift** between simulated data and real-world observations

The repository provides a strong supervised baseline (**ResNet18**) and a domain adaptation variant (**WDGRL**) to improve robustness under distribution changes.

---

## Repository Structure
```text
.
├── src/                    # Core modules: data loading, models, utilities
├── train.py                # Main training/evaluation entry point
├── plot_results.py         # Generates baseline vs WDGRL comparison plot
├── outputs_baseline/       # Artifacts from baseline training/evaluation
├── outputs_wdgrl/          # Artifacts from WDGRL training/evaluation
├── comparison.png          # Final comparison graph
├── requirements.txt
└── README.md
```

---

### Prediction Task

The model performs binary classification on astronomical images and outputs:

- 1 → Gravitational Lens detected  
- 0 → Non-Lens  

For each input image, the model produces a probability score (via softmax) indicating the likelihood of the presence of a gravitational lens.

## Approach
### 1) Baseline Model
- **Backbone:** ResNet18 (PyTorch)
- **Task:** Binary classification (2 output classes)
- **Objective:** Class-weighted cross-entropy to mitigate imbalance

### 2) Domain Adaptation Model
- **Method:** **WDGRL** (Wasserstein Distance Guided Representation Learning)
- **Goal:** Learn domain-invariant representations and improve performance under noisy/shifted conditions

---

## Dataset Structure
Expected directory layout:

```text
data/
├── train_lenses/
├── train_nonlenses/
├── test_lenses/
└── test_nonlenses/
```

---

## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train baseline
```bash
python train.py --data-root data --output-dir outputs_baseline
```

### 3. Train WDGRL
```bash
python train.py --data-root data --use-wdgrl --output-dir outputs_wdgrl
```

### 4. Evaluate without retraining
```bash
python train.py --data-root data --epochs 0
```

---

## Results

| Condition  | Baseline ROC-AUC | WDGRL ROC-AUC |
|------------|------------------:|--------------:|
| Clean      | 0.9839            | 0.9804        |
| Noise      | 0.8947            | 0.8991        |
| Blur       | 0.9840            | 0.9797        |
| Low Light  | 0.9833            | 0.9818        |

---

## Key Insights
WDGRL shows **improved robustness under noisy conditions** while introducing a small reduction on clean-domain performance.  
This trade-off indicates stronger **generalization under domain shift**, which is often more relevant for real observational pipelines.

---

## Outputs
Each run saves reproducible artifacts, including:
- **Model weights**
- **ROC curves**
- **`history.csv`** (training/evaluation metrics)
- **Training logs**
- **Robustness evaluation results**

---

## Future Work
- Explore stronger backbones (e.g., ConvNeXt, ViT)
- Add threshold calibration for precision-recall trade-offs
- Investigate hard-negative mining for false positives
- Extend adaptation to multi-source real telescope domains
- Move from binary detection to lens morphology classification
