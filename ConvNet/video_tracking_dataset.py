import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoTrackingDataset(Dataset):
    def __init__(self, lables_dir, video_dir, transform=None):
        self.lables_dir = lables_dir
        self.video_dir = video_dir
        self.transform = transform

        self.data = []
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith('.mp4'):
                label_file = os.path.splitext(video_file)[0] + '.csv'
                label_path = os.path.join(lables_dir, label_file)
                labels = pd.read_csv(label_path)
                grouped = labels.groupby('t')

                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                for frame_idx in range(frame_count):
                    if frame_idx in grouped.groups:
                        frame_labels = grouped.get_group(frame_idx)[["x", "y"]].values  # (n_objects, 2)
                        self.data.append((video_path, frame_idx, frame_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, frame_idx, frame_labels = self.data[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        positions = torch.tensor(frame_labels, dtype=torch.float32)  # (n_objects, 2)
        num_bugs = torch.tensor(positions.shape[0])

        return frame, positions, num_bugs
