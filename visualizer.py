import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

video_path = "./training/training089.mp4"
label_path = video_path.replace(".mp4", ".csv")

cap = cv2.VideoCapture(video_path)
labels = pd.read_csv(label_path)

previous_points = {}

# === Farbcodierung für IDs ===
id_colors = {}
def get_color(object_id):
    if object_id not in id_colors:
        np.random.seed(int(object_id))  # gleiche Farbe je ID
        id_colors[object_id] = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    return id_colors[object_id]

# === Initialisiere Frame-Index ===
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Alle Einträge dieses Frames holen
    frame_data = labels[labels["t"] == frame_index]

    for _, row in frame_data.iterrows():
        object_id = int(row["hexbug"])
        x, y = int(row["x"]), int(row["y"])

        color = get_color(object_id)

        # Linie von vorherigem Punkt zur aktuellen Position
        if object_id in previous_points:
            prev_x, prev_y = previous_points[object_id]
            cv2.line(frame, (prev_x, prev_y), (x, y), color, thickness=5)

        # Punkt zeichnen
        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

        # Update letzten Punkt
        previous_points[object_id] = (x, y)

    # Zeige Bild
    scale = 0.6 # z. B. 50 % der Originalgröße
    frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    cv2.imshow('Video mit Objektspuren', frame_resized)
    cv2.moveWindow('Video mit Objektspuren', 0, 0)
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    frame_index += 1

# Aufräumen
cap.release()
cv2.destroyAllWindows()

