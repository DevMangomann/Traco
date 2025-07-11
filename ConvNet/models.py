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


class HexbugPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 12)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class HexbugHeatmapTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.final_conv = nn.Conv2d(64, 1, 1, padding=0, stride=1)

    def forward(self, image_tensor):
        x = self.conv(image_tensor)
        x = self.upsampling(x)
        return self.final_conv(x)


class BigHexbugHeatmapTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.final_conv = nn.Conv2d(64, 1, 1, padding=0, stride=1)

    def forward(self, image_tensor):
        x = self.conv(image_tensor)
        x = self.upsampling(x)
        return self.final_conv(x)


class BigHexbugHeatmapTracker_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 192, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(192, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 192, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.final_conv = nn.Conv2d(64, 1, 1, padding=0, stride=1)

    def forward(self, image_tensor):
        x = self.conv(image_tensor)
        x = self.upsampling(x)
        return self.final_conv(x)


def init_weights_alexnet(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 1)
