import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.init as init


class HexbugTracker(nn.Module):
    def __init__(self, max_bugs=11):
        super().__init__()
        self.max_bugs = max_bugs
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * max_bugs),
            nn.Tanh()
        )

    def forward(self, num_bugs, image_tensor):
        x = self.conv(image_tensor)
        num_bugs = num_bugs.view(num_bugs.size(0), -1).float()
        x = x.view(num_bugs.size(0), -1)
        x = torch.cat([x, num_bugs], 1)
        x = self.fc(x)
        x = x.view(x.size(0), self.max_bugs, 2)
        return x


def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
