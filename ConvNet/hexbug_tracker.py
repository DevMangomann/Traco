import torch
import torch.nn.init as init
from torch import nn


class HexbugTracker(nn.Module):
    def __init__(self, max_bugs=11):
        super().__init__()
        self.max_bugs = max_bugs
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, padding=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(96, 128, 5, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2 * max_bugs),
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


def init_weights_alexnet(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 1)
