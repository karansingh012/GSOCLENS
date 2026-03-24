# Gravitational Lens Classification Using Deep Learning

## Repository Structure

```
.
├── notebooks/
│   └── experiment_template.ipynb
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── train.py
├── requirements.txt
└── README.md
```

Core logic lives in `src/`, while notebooks are kept in `notebooks/` for experiments and visualization.

## Project Overview
This project detects gravitational lenses from astronomical images using deep learning.
The task is a binary classification problem:
- 1 -> Lens
- 0 -> Non-Lens

A key challenge is class imbalance, where non-lens samples outnumber lens samples.
To reduce bias toward the majority class, training uses class-weighted cross entropy loss.

## Dataset Structure
Expected directory layout:

```
.
├── train_lenses/
├── train_nonlenses/
├── test_lenses/
└── test_nonlenses/
```

Supported file types in each folder:
- NumPy arrays: `.npy`, `.npz`
- Image formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

Input images are expected to represent 3 observational channels, commonly with shape `(3, 64, 64)`.
The loader also accepts HWC images and converts them to CHW.

## Model
- Backbone: ResNet18 (PyTorch)
- Initialization: ImageNet pretrained weights by default
- Classification head: final fully connected layer replaced for 2 classes

## Training Pipeline
1. Load train/test images from class folders using `src/data.py`
2. Split training into train/validation with stratified sampling
3. Apply augmentation (random flips and rotation) on training data
4. Build ResNet18 binary classifier from `src/model.py`
5. Handle class imbalance using weighted `CrossEntropyLoss`
6. Train with Adam optimizer and log train/validation metrics
7. Generate probabilities using softmax
8. Evaluate with ROC curve and AUC score

## Evaluation
Because of class imbalance, ROC-AUC is used as the main metric instead of raw accuracy.

The script saves:
- Trained weights: `outputs/resnet18_lens.pt`
- ROC plot: `outputs/roc_curve.png` (when both classes exist in test set)
- Training history: `outputs/history.csv`
- Training logs: `outputs/training.log`

## Installation
```bash
pip install -r requirements.txt
```

## Run Training
From the project root:

```bash
python train.py --data-root . --epochs 15 --batch-size 32
```

Optional flags:
- `--no-pretrained` to disable ImageNet initialization
- `--lr` learning rate (default `1e-4`)
- `--weight-decay` weight decay (default `1e-4`)
- `--num-workers` dataloader workers
- `--output-dir` where artifacts are saved

## Future Work
- Compare with Vision Transformers (ViT)
- Use threshold optimization to reduce false positives
- Analyze false positives and hard negatives
- Apply domain adaptation for real observational shifts
- Extend to multi-class lens morphology tasks
