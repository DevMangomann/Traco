import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

import traco.ConvNet.augmentations as augmentations
from traco.ConvNet.helper import get_image_size, denormalize_positions, generate_heatmap

video_path = "../leaderboard_data/test001.mp4"
label_path = "../leaderboard_data/test001.csv"
cap = cv2.VideoCapture(video_path)
labels = pd.read_csv(label_path)

transform = augmentations.JointCompose([augmentations.JointStretch(0.33, 0.1),
                                        augmentations.ResizeImagePositions((512, 512)),
                                        # augmentations.JointWrapper(transforms.ToTensor()),
                                        augmentations.JointRandomFlip(0.5, 0.5),
                                        augmentations.JointRotation(180.0),
                                        augmentations.JointWrapper(transforms.ToTensor()),
                                        augmentations.JointWrapper(
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                   hue=0.02)),
                                        augmentations.JointWrapper(
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
                                        ])

paused = False
frame_idx = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV: BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_positions = labels[labels["t"] == frame_idx][["x", "y"]].values

        # transforms
        frame_rgb, label_positions = transform(frame_rgb, label_positions)
        height, width = get_image_size(frame_rgb)
        heatmap = generate_heatmap(get_image_size(frame_rgb), label_positions)
        label_positions = denormalize_positions(label_positions, (height, width), (256, 256))

        if isinstance(frame_rgb, torch.Tensor):
            frame_rgb = frame_rgb.permute(1, 2, 0).numpy()
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        heatmap = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)
        if isinstance(label_positions, torch.Tensor):
            label_positions = label_positions.numpy()
        for (x, y) in label_positions.astype(int):
            cv2.circle(frame_bgr, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

        cv2.imshow('Video mit transforms', frame_bgr)
        cv2.moveWindow('Video mit transforms', 0, 0)
        cv2.imshow('HEATMAP', heatmap)
        cv2.moveWindow('HEATMAP', 500, 0)

        frame_idx += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
