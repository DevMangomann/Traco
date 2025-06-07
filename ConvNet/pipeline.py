import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
import augmentations
from traco.ConvNet.helper import denormalize_positions

from traco.ConvNet.models import HexbugPredictor
from traco.ConvNet.models import HexbugTracker

# Lade das trainierte Modell
predictor_model = HexbugPredictor()
predictor_model.load_state_dict(torch.load("model_weights/hexbug_predictor_folds1_v20.pth", weights_only=True))
predictor_model.eval()  # Setze das Modell in den Evaluierungsmodus

tracking_model = HexbugTracker()
tracking_model.load_state_dict(torch.load("model_weights/hexbug_tracker_background_v60.pth", weights_only=True))
tracking_model.eval()

target_size = (512, 512)

# Definiere die Transformationen für die Eingabebilder
test_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# Pfad zum Video
video_path = "../Background_training/training02.mp4"

# Lade das Video
cap = cv2.VideoCapture(video_path)

# Erstelle eine leere Liste, um die Vorhersagen zu speichern
predictions = np.empty((0, 5))

# Verarbeite jeden Frame des Videos
frame_count = 0
row_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertiere den Frame in ein PIL-Bild
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    # Wende die Transformationen auf das Bild an
    predict_image = test_transform(image)
    tracking_image = test_transform(image)

    # Füge eine Batch-Dimension hinzu
    predict_image = predict_image.unsqueeze(0)
    tracking_image = tracking_image.unsqueeze(0)

    # Führe die Vorhersage mit dem Modell aus
    with torch.no_grad():
        num_bugs = predictor_model(predict_image)
        num_bugs = torch.argmax(num_bugs, dim=1)
        num_bugs = torch.tensor([1])
        positions = tracking_model(num_bugs, tracking_image)

    # Extrahiere die x- und y-Koordinaten aus der Vorhersage
    num_bugs = int(num_bugs.item())
    coords = positions[0, :num_bugs]  # (num_bugs, 2)
    coords = coords.cpu().numpy()

    # Hole die Originalgröße des Bildes
    original_height, original_width = augmentations.get_image_size(frame)

    # Rechne die x- und y-Koordinaten zurück
    coords = denormalize_positions(coords, (original_height, original_width), target_size)

    if frame_count == 0:
        for i in range(num_bugs):
            new_row = np.array([[int(row_count), int(frame_count), int(i), float(coords[i, 0]), float(coords[i, 1])]],
                               dtype=np.float32)
            predictions = np.vstack([predictions, new_row])
            row_count += 1
    else:
        mask = predictions[:, 1] == frame_count - 1
        last_prediction = predictions[mask]
        prev_coords = last_prediction[:, 3:5]  # (N, 2)

        # Matching mit Hungarian
        cost_matrix = torch.cdist(torch.tensor(prev_coords, dtype=torch.float32), torch.tensor(coords, dtype=torch.float32), p=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # IDs übernehmen
        for i in range(len(row_ind)):
            prev_id = last_prediction[row_ind[i], 0]
            new_x, new_y = coords[col_ind[i]]
            new_row = np.array([[int(row_count), int(frame_count), int(i), float(new_x), float(new_y)]],
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
