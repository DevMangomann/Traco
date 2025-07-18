import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from traco.ConvNet import helper
from traco.ConvNet.id_tracking import update_tracks
from traco.ConvNet.models import HexbugHeatmapTracker, HexbugPredictor, BigHexbugHeatmapTracker, BigHexbugHeatmapTracker_v2
from kalman_tracking import KalmanMultiObjectTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lade das trainierte Modell
predictor_model = HexbugPredictor()
predictor_model.load_state_dict(
    torch.load("model_weights/hexbug_predictor_folds1_v80.pth", weights_only=True, map_location=device))
predictor_model.eval()  # Setze das Modell in den Evaluierungsmodus

tracking_model = BigHexbugHeatmapTracker_v2()
tracking_model.load_state_dict(
    torch.load("model_weights/big_hexbug_heatmap_tracker_v2_folds1_v70.pth", weights_only=True, map_location=device))
tracking_model.eval()

# Definiere die Transformationen für die Eingabebilder
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Pfad zum Video
video_path = "../leaderboard_data/test006_20s.mp4"
max_bugs = 4

# Lade das Video
cap = cv2.VideoCapture(video_path)

# Erstelle eine leere Liste, um die Vorhersagen zu speichern
predictions = np.empty((0, 5))
kalman_tracker = KalmanMultiObjectTracker(max_bugs)

# Verarbeite jeden Frame des Videos
frame_count = 0
row_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:
        print(frame_count)

    # Konvertiere den Frame in ein PIL-Bild
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    # Wende die Transformationen auf das Bild an
    image = transform(image)

    # Füge eine Batch-Dimension hinzu
    image = image.unsqueeze(0)

    # Führe die Vorhersage mit dem Modell aus
    with torch.no_grad():
        #num_bugs = predictor_model(image)
        #print(num_bugs)
        #num_bugs = torch.argmax(num_bugs, dim=1)
        #print(num_bugs)
        num_bugs = torch.tensor([max_bugs])
        heatmap = tracking_model(image)[0, 0]

    # Extrahiere die x- und y-Koordinaten aus der Vorhersage
    num_bugs = int(num_bugs.item())
    original_height, original_width = helper.get_image_size(frame)
    coords = helper.coords_from_heatmap(heatmap, num_bugs, (original_height, original_width))

    hexbug_tracks = kalman_tracker.update(coords, (original_height, original_width))
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
