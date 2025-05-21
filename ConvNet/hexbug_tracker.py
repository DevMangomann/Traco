import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class HexbugTracker(nn.Module):
    def __init__(self, max_bugs=11):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=4),  # 512 * 512 images
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=3),  # 256 * 256 images
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=2),  # 128 * 128 images
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  # 64 * 64 images
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * max_bugs),  # Output number of hexbugs
            nn.ReLU()
        )

    def forward(self, num_bugs, image_tensor):
        x = self.conv(image_tensor)
        x = x.view(x.size(0), -1)
        num_bugs = num_bugs.view(num_bugs.size(0), -1).float()
        x = torch.cat([x, num_bugs], 1)
        return self.fc(x)
