"""Protocol-frozen raw-sequence neural architectures."""

from __future__ import annotations

import torch
from torch import nn


class CompactCNN(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values).squeeze(-1))


class DeepConvLSTM(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        layers = []
        input_channels = 6
        for _ in range(4):
            layers.extend(
                [nn.Conv1d(input_channels, 64, kernel_size=5, padding="same"), nn.ReLU()]
            )
            input_channels = 64
        self.convolution = nn.Sequential(*layers)
        self.recurrent = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = self.convolution(values).transpose(1, 2)
        output, _ = self.recurrent(sequence)
        return self.classifier(output[:, -1])


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_channels),
        )
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channels),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(values) + self.shortcut(values))


class ResNet1D(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.stages = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(256, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stages(self.stem(values)).squeeze(-1))


def build_neural_model(name: str, classes: int) -> nn.Module:
    if name == "compact_cnn":
        return CompactCNN(classes)
    if name == "deepconvlstm":
        return DeepConvLSTM(classes)
    if name == "resnet1d":
        return ResNet1D(classes)
    raise ValueError(f"Unknown neural architecture: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
