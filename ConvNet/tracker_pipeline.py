import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from hungarian_tracking import update_tracks
from torchvision import transforms
import augmentations
from traco.ConvNet.helper import denormalize_positions

from traco.ConvNet.models import HexbugCounter, HexbugTracker

counting_model = HexbugCounter()
counting_model.load_state_dict(torch.load("model_weights/hexbug_counter_v70_training.pth", weights_only=True))
counting_model.eval()  # Setze das Modell in den Evaluierungsmodus

tracking_model = HexbugTracker()
tracking_model.load_state_dict(torch.load("model_weights/hexbug_tracker_70_training.pth", weights_only=True))
tracking_model.eval()

target_size = (256, 256)

test_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

video_path = "../training/training07.mp4"

cap = cv2.VideoCapture(video_path)

predictions = np.empty((0, 5))

frame_count = 0
row_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    image = test_transform(image)

    image = image.unsqueeze(0)

    # Führe die Vorhersage mit dem Modell aus
    with torch.no_grad():
        num_bugs = counting_model(image)
        num_bugs = torch.argmax(num_bugs, dim=1)
        positions = tracking_model(num_bugs, image)

    num_bugs = int(num_bugs.item())
    coords = positions[0, :num_bugs]
    coords = coords.cpu().numpy()

    original_height, original_width = augmentations.get_image_size(frame)

    coords = denormalize_positions(coords, (original_height, original_width), target_size)

    hexbug_tracks = update_tracks(coords, np.array(frame))
    for hexbug in hexbug_tracks:
        hex_id, hex_coords = hexbug
        new_row = np.array(
            [[int(row_count), int(frame_count), int(hex_id), float(hex_coords[0]), float(hex_coords[1])]],
            dtype=np.float32)
        predictions = np.vstack([predictions, new_row])
        row_count += 1

    frame_count += 1

# Schließe das Video
cap.release()

# Erstelle einen Pandas DataFrame aus den Vorhersagen
df = pd.DataFrame(predictions, columns=["", "t", "hexbug", "x", "y"])
df[""] = df[""].astype(int)
df["t"] = df["t"].astype(int)
df["hexbug"] = df["hexbug"].astype(int)

# Speichere den DataFrame in einer CSV-Datei
video_name = video_path.split("/")[-1]
video_name = video_name.split(".")[0]
df.to_csv(f"./predictions/{video_name}_prediction.csv", index=False)

print("Vorhersagen wurden in predictions.csv gespeichert.")
