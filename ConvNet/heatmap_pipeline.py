import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from traco.ConvNet import helper
from traco.ConvNet.hungarian_tracking import update_tracks
from traco.ConvNet.models import HeatmapTracker, BigHeatmapTracker, BiggerHeatmapTracker
from kalman_tracking import KalmanMultiObjectTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_model = BiggerHeatmapTracker()
tracking_model.load_state_dict(
    torch.load("model_weights/heatmap_tracker_v80_training.pth", weights_only=True, map_location=device))
tracking_model.eval()
tracking_model = tracking_model.to(device)

# transforms for input
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# path to video and maximum number of hexbugs in video
video_path = "../leaderboard_data/test001.mp4"
max_bugs = 3

cap = cv2.VideoCapture(video_path)

# list for predictions and used tracking algorithm
predictions = np.empty((0, 5))
kalman_tracker = KalmanMultiObjectTracker(max_bugs)

frame_count = 0
row_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:
        print(f"Frame {frame_count}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = tracking_model(image)[0, 0].cpu()

    original_height, original_width = helper.get_image_size(frame)
    coords = helper.coords_from_heatmap(heatmap, max_bugs, (original_height, original_width))

    hexbug_tracks = kalman_tracker.update(coords, np.array(frame))
    for hexbug in hexbug_tracks:
        hex_id, hex_coords = hexbug
        new_row = np.array(
            [[int(row_count), int(frame_count), int(hex_id), float(hex_coords[0]), float(hex_coords[1])]],
            dtype=np.float32)
        predictions = np.vstack([predictions, new_row])
        row_count += 1

    frame_count += 1

cap.release()

# panda dataframe for predictions
df = pd.DataFrame(predictions, columns=["", "t", "hexbug", "x", "y"])
df[""] = df[""].astype(int)
df["t"] = df["t"].astype(int)
df["hexbug"] = df["hexbug"].astype(int)

# store predictions in csv
video_name = video_path.split("/")[-1]
video_name = video_name.split(".")[0]
df.to_csv(f"./predictions/{video_name}_prediction.csv", index=False)

print(f"Predictions stored in /predictions/{video_name}_prediction.csv")
