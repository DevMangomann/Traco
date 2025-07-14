import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, initial_pos):
        self.id = track_id
        self.kf = self._init_kf(initial_pos)
        self.age = 0
        self.time_since_update = 0
        self.last_detection = initial_pos

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

    def update(self, pos):
        self.kf.update(pos)
        self.last_detection = pos
        self.time_since_update = 0


class KalmanMultiObjectTracker:
    def __init__(self, max_bugs, max_age=20, dist_thresh=1000.0):
        self.max_bugs = max_bugs
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, detections, frame_size):
        height, width = frame_size
        detections = np.array(detections)
        last_dets = np.array([t.last_detection for t in self.tracks])

        # Step 1: Predict all tracks
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
        else:
            # Step 2: Compute cost matrix (euclidean distance)
            detect_matrix = torch.cdist(
                torch.tensor(last_dets, dtype=torch.float32),
                torch.tensor(detections, dtype=torch.float32)
            ).numpy()

            # Step 3: Hungarian matching
            row_detect, col_detect = linear_sum_assignment(detect_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            for i, j in zip(row_detect, col_detect):
                if detect_matrix[i, j] < self.dist_thresh:
                    self.tracks[i].update(detections[j])
                    assigned_tracks.add(i)
                    assigned_detections.add(j)

            # Step 5: Unmatched tracks (no update)
            for i, track in enumerate(self.tracks):
                if i not in assigned_tracks:
                    track.time_since_update += 1

            # Step 6: Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # Step 7: Create new tracks for unmatched detections
            if len(self.tracks) < self.max_bugs:
                for i, det in enumerate(detections):
                    if i not in assigned_detections:
                        self.tracks.append(Track(self.next_id, det))
                        self.next_id += 1

        # Step 8: Prepare output
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.last_detection is not None:
                results.append((t.id, t.last_detection))  # Detektion zugewiesen
            else:
                kalman_detection = t.kf.x[:2].flatten()
                kalman_detection[0] = np.clip(kalman_detection[0], 0, width)
                kalman_detection[1] = np.clip(kalman_detection[1], 0, height)
                results.append((t.id, kalman_detection))  # Prediction-Fallback
                t.last_detection = kalman_detection
        return results
