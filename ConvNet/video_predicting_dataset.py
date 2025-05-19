import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class VideoPredictingDataset(Dataset):
    def __init__(self, lables_dir, video_dir, transform=None, target_transform=None):
        self.lables_dir = lables_dir
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith('.mp4'):
                label_file = os.path.splitext(video_file)[0] + '.csv'
                label_path = os.path.join(lables_dir, label_file)
                labels = pd.read_csv(label_path)
                num_bugs = sum(labels["t"] == 0)
                label = float(num_bugs)

                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                for frame_idx in range(frame_count):
                    self.data.append((video_path, frame_idx, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, frame_idx, label = self.data[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            label = self.target_transform(label)

        return frame, torch.tensor(label, dtype=torch.float32)
