from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


# -------------------------------
# 1. BASELINE MODEL (ResNet18)
# -------------------------------
def build_resnet18_binary(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    return model


# -------------------------------
# 2. FEATURE EXTRACTOR (WDGRL / ADDA)
# -------------------------------
def build_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Remove final FC layer
    layers = list(model.children())[:-1]

    # 🔥 Add flatten here (important)
    feature_extractor = nn.Sequential(
        *layers,
        nn.Flatten()
    )

    return feature_extractor


# -------------------------------
# 3. CLASSIFIER HEAD
# -------------------------------
def build_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )


# -------------------------------
# 4. DOMAIN CRITIC (WDGRL / ADDA)
# -------------------------------
class DomainCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)