import csv
import random
import cv2
import numpy as np
from collections import defaultdict

# Parameter
csv_path = "../leaderboard_data/test003.csv"
num_bugs = 4
video_path = csv_path.replace("csv", "mp4")

# Video öffnen
cap = cv2.VideoCapture(video_path)

# CSV-Datei einlesen
with open(csv_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)

# Header überspringen
if rows[0][1] == 't':
    rows = rows[1:]

# CSV-Daten in Dictionary: frame_id -> {hexbug_id: (x, y)}
frame_data = defaultdict(dict)
for row in rows:
    t = int(row[1])
    hexbug_id = int(row[2])
    x = int(float(row[3]))
    y = int(float(row[4]))
    frame_data[t][hexbug_id] = (x, y)

# Farben für Hexbugs
colors = {
    i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(num_bugs)
}

# Initialisierung für Spuren (Positionen merken)
bug_trails = defaultdict(list)  # hexbug_id -> Liste von Punkten

# Video abspielen mit Spuren
frame_id = 0
scale = 0.5  # z. B. auf 50 % der Originalgröße skalieren

trail_thickness = 2
circle_radius = 10

# Trail-Bild initialisieren
trail_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if trail_image is None:
        # Leeres Bild zum Einzeichnen der Trails, gleiche Größe wie Frame
        trail_image = np.zeros_like(frame)

    max_trail_length = 10  # z. B. nur die letzten 10 Positionen

    if frame_id in frame_data:
        for hexbug_id, (x, y) in frame_data[frame_id].items():
            # Punkt zur Spur hinzufügen
            bug_trails[hexbug_id].append((x, y))
            if len(bug_trails[hexbug_id]) > max_trail_length:
                bug_trails[hexbug_id].pop(0)

            # Spur zeichnen (basierend auf aktuellen bug_trails)
            color = colors[hexbug_id]
            for i in range(1, len(bug_trails[hexbug_id])):
                pt1 = bug_trails[hexbug_id][i - 1]
                pt2 = bug_trails[hexbug_id][i]
                cv2.line(frame, pt1, pt2, color, thickness=2)

            # Aktuelle Position als Kreis markieren
            cv2.circle(frame, (x, y), radius=circle_radius, color=color, thickness=-1)
            cv2.putText(frame, f"Bug {hexbug_id}", (x + 10, y - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=2)

    # Frame und Trail-Bild kombinieren
    frame_with_trail = cv2.addWeighted(frame, 1.0, trail_image, 1.0, 0)

    # Optional: Skalieren
    resized_frame = cv2.resize(frame_with_trail, None, fx=scale, fy=scale)

    # Anzeigen
    cv2.imshow(video_path, resized_frame)

    key = cv2.waitKey(60)
    if key == 27:  # ESC zum Beenden
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
