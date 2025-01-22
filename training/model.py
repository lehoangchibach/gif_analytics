import numpy as np
import torch
import torch.nn as nn

from constants import *


class VideoCNN1(nn.Module):  # Best threshold 10
    name = "VideoCNN1"

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
        )

        self.flat_features = self._calculate_flat_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flat_features(self) -> int:
        x = torch.randn(1, 3, NUM_FRAMES, *IMAGE_SIZE)
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_frames, height, width, channels)
        # Need to permute to: (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv3d(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


class VideoCNN2(nn.Module):
    name = "VideoCNN2"

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
        )

        # Calculate the size of the flattened features
        self.flat_features = self._calculate_flat_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flat_features(self) -> int:
        x = torch.randn(1, 3, NUM_FRAMES, *IMAGE_SIZE)
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_frames, height, width, channels)
        # Need to permute to: (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv3d(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


class VideoCNN3(nn.Module):  # 1st big model
    name = "VideoCNN3"

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
        )

        self.flat_features = self._calculate_flat_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flat_features(self) -> int:
        x = torch.randn(1, 3, NUM_FRAMES, *IMAGE_SIZE)
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_frames, height, width, channels)
        # Need to permute to: (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv3d(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


VideoCNN = VideoCNN1
