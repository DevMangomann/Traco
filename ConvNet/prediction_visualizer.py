import csv
import random
import cv2
import numpy as np
from collections import defaultdict

csv_path = "../leaderboard_data/test001.csv"
num_bugs = 3
video_path = csv_path.replace("csv", "mp4")

cap = cv2.VideoCapture(video_path)

with open(csv_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)

if rows[0][1] == 't':
    rows = rows[1:]

frame_data = defaultdict(dict)
for row in rows:
    t = int(row[1])
    hexbug_id = int(row[2])
    x = int(float(row[3]))
    y = int(float(row[4]))
    frame_data[t][hexbug_id] = (x, y)

# colors for hexbugs
colors = {
    i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(num_bugs)
}


bug_trails = defaultdict(list)

frame_id = 0
scale = 0.5

trail_thickness = 2
circle_radius = 10

trail_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if trail_image is None:
        trail_image = np.zeros_like(frame)

    max_trail_length = 10

    if frame_id in frame_data:
        for hexbug_id, (x, y) in frame_data[frame_id].items():
            bug_trails[hexbug_id].append((x, y))
            if len(bug_trails[hexbug_id]) > max_trail_length:
                bug_trails[hexbug_id].pop(0)

            color = colors[hexbug_id]
            for i in range(1, len(bug_trails[hexbug_id])):
                pt1 = bug_trails[hexbug_id][i - 1]
                pt2 = bug_trails[hexbug_id][i]
                cv2.line(frame, pt1, pt2, color, thickness=2)

            cv2.circle(frame, (x, y), radius=circle_radius, color=color, thickness=-1)
            cv2.putText(frame, f"Bug {hexbug_id}", (x + 10, y - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=2)

    frame_with_trail = cv2.addWeighted(frame, 1.0, trail_image, 1.0, 0)

    resized_frame = cv2.resize(frame_with_trail, None, fx=scale, fy=scale)

    cv2.imshow(video_path, resized_frame)

    key = cv2.waitKey(60)
    if key == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
