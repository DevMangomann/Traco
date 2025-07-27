import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import helper


def average_color_around(pos, frame, radius):
    x, y = int(pos[0]), int(pos[1])
    h, w = helper.get_image_size(frame)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return np.array([0, 0, 0])
    return region.reshape(-1, 3).mean(axis=0)


class Track:
    def __init__(self, track_id, initial_pos, color):
        self.id = track_id
        self.kf = self._init_kf(initial_pos)
        self.age = 0
        self.time_since_update = 0
        self.last_detection = initial_pos
        self.color = color

    def _init_kf(self, pos):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.x[:2] = np.array([[pos[0]], [pos[1]]])
        kf.P *= 1000.
        kf.R *= 10.0
        kf.Q *= 1.0
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:2].flatten()

    def update(self, pos, frame):
        self.kf.update(pos)
        self.last_detection = pos
        self.color = average_color_around(pos, frame, 2)
        self.time_since_update = 0


class KalmanMultiObjectTracker:
    def __init__(self, max_bugs, max_age=50, dist_thresh=300.0):
        self.max_bugs = max_bugs
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, detections, frame):
        height, width = helper.get_image_size(frame)
        detections = np.array(detections)

        # Step 1: Predict all tracks
        for track in self.tracks:
            track.predict()

        if len(detections) == 0:
            # No tracks -> everything mark missing
            for track in self.tracks:
                track.time_since_update += 1

            # delete older tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        elif len(self.tracks) == 0:
            # first frame tracks
            for det in detections:
                color = average_color_around(det, frame, 10)
                self.tracks.append(Track(self.next_id, det, color))
                self.next_id += 1

        else:
            last_dets = np.array([t.last_detection for t in self.tracks])

            # Compute cost matrix over distance and color
            detect_matrix = torch.cdist(
                torch.tensor(last_dets, dtype=torch.float32),
                torch.tensor(detections, dtype=torch.float32)
            ).numpy()

            track_colors = np.array([t.color for t in self.tracks])
            det_colors = np.array([average_color_around(det, frame, 10) for det in detections])

            if len(det_colors) == 0:
                color_matrix = np.zeros_like(detect_matrix)
            else:
                color_matrix = np.linalg.norm(track_colors[:, None, :] - det_colors[None, :, :], axis=2)

            combined_matrix = detect_matrix + 0.3 * color_matrix

            # Hungarian matching
            row_detect, col_detect = linear_sum_assignment(combined_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            for i, j in zip(row_detect, col_detect):
                if detect_matrix[i, j] < self.dist_thresh:
                    self.tracks[i].update(detections[j], frame)
                    assigned_tracks.add(i)
                    assigned_detections.add(j)

            # Unmatched tracks
            for i, track in enumerate(self.tracks):
                if i not in assigned_tracks:
                    track.time_since_update += 1

            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # Create new tracks for unmatched detections
            if len(self.tracks) < self.max_bugs:
                for i, det in enumerate(detections):
                    if i not in assigned_detections:
                        color = average_color_around(det, frame, 10)
                        self.tracks.append(Track(self.next_id, det, color))
                        self.next_id += 1

        # Prepare output with kalman as fallback
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.last_detection is not None:
                results.append((t.id, t.last_detection))
            else:
                kalman_detection = t.kf.x[:2].flatten()
                x, y = kalman_detection[0], kalman_detection[1]
                if x < 0 or y < 0 or x >= width or y >= height:
                    kalman_detection[0] = np.clip(kalman_detection[0], 0, width)
                    kalman_detection[1] = np.clip(kalman_detection[1], 0, height)
                else:
                    results.append((t.id, kalman_detection))
                t.last_detection = kalman_detection
        return results

