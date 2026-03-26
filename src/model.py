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
# 2. FEATURE EXTRACTOR (WDGRL)
# -------------------------------
def build_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Remove final FC layer
    layers = list(model.children())[:-1]  # remove classifier
    feature_extractor = nn.Sequential(*layers)

    return feature_extractor


# -------------------------------
# 3. CLASSIFIER HEAD
# -------------------------------
def build_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )


# -------------------------------
# 4. DOMAIN CRITIC (WDGRL CORE)
# -------------------------------
class DomainCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten (B, 512, 1, 1) → (B, 512)
        return self.net(x)