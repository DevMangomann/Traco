import numpy as np
import cv2
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import traco.ConvNet.augmentations as augmentations

video_path = "../training/training089.mp4"
label_path = video_path.replace(".mp4", ".csv")
cap = cv2.VideoCapture(video_path)
labels = pd.read_csv(label_path)

# Beispiel-Transformationskette
transform = augmentations.JointCompose([
    augmentations.ResizeImagePositions((512, 512)),
    augmentations.JointRandomFlip(1.0, 1.0),
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    # transforms.Normalize(0.5, 0.5),
    # transforms.Resize((512, 512)),
    # transforms.RandomHorizontalFlip(p=0.5),
    augmentations.JointWrapper(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3)),
    # augmentations.JointWrapper(transforms.ToPILImage),  # wieder zu PIL zur Anzeige
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

        # Transform anwenden
        frame_rgb, label_positions = transform(frame_rgb, label_positions)
        label_positions = augmentations.denormalize_positions(label_positions, frame_rgb.size)

        # Zurück zu NumPy und RGB -> BGR für OpenCV-Anzeige
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        for (x, y) in label_positions.numpy().astype(int):
            cv2.circle(frame_bgr, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

        cv2.imshow('Video mit transforms', frame_bgr)
        cv2.moveWindow('Video mit transforms', 0, 0)

        frame_idx += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
