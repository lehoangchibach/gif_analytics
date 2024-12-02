import numpy as np
import torch
import torch.nn as nn

from constants import *


class VideoCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
        )

        # Calculate the size of the flattened features
        self.flat_features = self._calculate_flat_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 4096),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flat_features(self) -> int:
        x = torch.randn(1, 3, NUM_FRAMES, *IMAGE_SIZE)
        # x = torch.randn(1, NUM_FRAMES, 512, 512, 3)
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, num_frames, height, width, channels)
        # Need to permute to: (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv3d(x)
        # print(f"Shape after conv3d: {x.shape}")

        x = x.view(x.size(0), -1)
        # print(f"Shape after flatten: {x.shape}")

        x = self.classifier(x)
        return x
