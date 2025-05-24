import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from traco.ConvNet.hexbug_predictor import HexbugPredictor
from traco.ConvNet.hexbug_tracker import HexbugTracker

# Lade das trainierte Modell
predictor_model = HexbugPredictor()
predictor_model.load_state_dict(torch.load("models/hexbug_predictor_folds1_v10.pth", weights_only=True))
predictor_model.eval()  # Setze das Modell in den Evaluierungsmodus

tracking_model = HexbugTracker()
tracking_model.load_state_dict(torch.load("models/hexbug_tracker_folds1_v2.pth", weights_only=True))
tracking_model.eval()

# Definiere die Transformationen für die Eingabebilder
predict_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

tracking_transform = transforms.Compose([transforms.ToTensor()])

# Pfad zum Video
video_path = "../training/training01.mp4"

# Lade das Video
cap = cv2.VideoCapture(video_path)

# Erstelle eine leere Liste, um die Vorhersagen zu speichern
predictions = []

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
    predict_image = predict_transform(image)
    tracking_image = tracking_transform(image)

    # Füge eine Batch-Dimension hinzu
    predict_image = predict_image.unsqueeze(0)
    tracking_image = tracking_image.unsqueeze(0)

    # Führe die Vorhersage mit dem Modell aus
    with torch.no_grad():
        num_bugs = predictor_model(predict_image)
        positions = tracking_model(num_bugs, tracking_image)

    # Extrahiere die x- und y-Koordinaten aus der Vorhersage
    num_bugs = torch.round(num_bugs)
    num_bugs = int(num_bugs.item())
    num_bugs = 1
    coords = positions[0, :num_bugs]  # (num_bugs, 2)
    coords = coords.cpu().numpy()

    # Hole die Originalgröße des Bildes
    original_height, original_width, _ = frame.shape

    # Rechne die x- und y-Koordinaten zurück
    coords[:, 0] = ((coords[:, 0] + 1) / 2) * original_width  # x
    coords[:, 1] = ((coords[:, 1] + 1) / 2) * original_height  # y

    # Füge die Vorhersage zur Liste hinzu
    predictions.append([row_count, frame_count, 0, coords[:, 0], coords[:, 1]])

    frame_count += 1
    row_count += 1

# Schließe das Video
cap.release()

# Erstelle einen Pandas DataFrame aus den Vorhersagen
df = pd.DataFrame(predictions, columns=["", "t", "hexbug", "x", "y"])

# Speichere den DataFrame in einer CSV-Datei
video_name = video_path.split("/")[-1]
video_name = video_name.split(".")[0]
df.to_csv(f"./predictions/{video_name}_prediction", index=False)

print("Vorhersagen wurden in predictions.csv gespeichert.")
