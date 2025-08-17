import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

import traco.ConvNet.augmentations as augmentations
from traco.ConvNet.helper import get_image_size, denormalize_positions, generate_heatmap

video_path = "../training/training07.mp4"
label_path = "../training/training07.csv"
cap = cv2.VideoCapture(video_path)
labels = pd.read_csv(label_path)
resize = (512, 512)

transform = augmentations.JointCompose([
    augmentations.JointStretch(0.33, 0.1),
    augmentations.ResizeImagePositions(resize),
    augmentations.JointRandomFlip(0.5, 0.5),
    augmentations.JointRotation(180.0),
    augmentations.JointWrapper(transforms.ToTensor()),
    augmentations.JointWrapper(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)),
    augmentations.JointWrapper(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
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

        # Original sichern (als RGB)
        original_rgb = frame_rgb.copy()

        # Lade Keypoints für aktuellen Frame
        label_positions = labels[labels["t"] == frame_idx][["x", "y"]].values

        # Transformieren (bild + keypoints)
        transformed_rgb, transformed_positions = transform(frame_rgb, label_positions)

        # Für Heatmap: Höhe/Breite holen
        height, width = get_image_size(transformed_rgb)
        heatmap = generate_heatmap((height, width), transformed_positions)

        # Keypoints wieder von Normalisierung zurückskalieren
        denorm_positions = denormalize_positions(transformed_positions, (height, width), (height, width))

        # Tensor zu NumPy (HWC)
        if isinstance(transformed_rgb, torch.Tensor):
            transformed_rgb = transformed_rgb.permute(1, 2, 0).numpy()
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.numpy()

        # RGB -> BGR (für OpenCV)
        transformed_bgr = cv2.cvtColor(np.array(transformed_rgb), cv2.COLOR_RGB2BGR)
        original_rgb = frame_rgb.copy()
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        original_bgr_small = cv2.resize(original_bgr, resize)
        heatmap_gray = (heatmap * 255).astype(np.uint8)

        # Keypoints einzeichnen (nur auf transformiertem Bild)
        if isinstance(denorm_positions, torch.Tensor):
            denorm_positions = denorm_positions.numpy()
        for (x, y) in denorm_positions.astype(int):
            cv2.circle(transformed_bgr, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

        # Anzeigen
        cv2.imshow('Original', original_bgr_small)
        cv2.moveWindow('Original', 0, 0)

        cv2.imshow('Transformed', transformed_bgr)
        cv2.moveWindow('Transformed', 550, 0)

        cv2.imshow('Heatmap', heatmap_gray)
        cv2.moveWindow('Heatmap', 1100, 0)

        frame_idx += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
