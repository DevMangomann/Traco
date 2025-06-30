import numpy as np
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
    def __init__(self, max_age=20, dist_thresh=300):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, detections, frame_id):
        # Step 1: Predict existing tracks
        predicted_positions = np.array([t.predict() for t in self.tracks])
        detections = np.array(detections)

        if len(self.tracks) == 0:
            # No existing tracks — initialize new ones
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
        else:
            # Step 2: Compute cost matrix (euclidean distances)
            cost_matrix = np.linalg.norm(predicted_positions[:, np.newaxis] - detections[np.newaxis, :], axis=2)

            # Step 3: Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            # Step 4: Match within threshold
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.dist_thresh:
                    self.tracks[r].update(detections[c])
                    assigned_tracks.add(r)
                    assigned_detections.add(c)

            # Step 5: Unmatched tracks
            for i, track in enumerate(self.tracks):
                if i not in assigned_tracks:
                    track.time_since_update += 1

            # Step 6: Remove old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            # Step 7: Create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in assigned_detections:
                    self.tracks.append(Track(self.next_id, det))
                    self.next_id += 1

        # Return current tracking results
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.last_detection is not None:
                # Neuste echte Messung verwenden
                results.append((t.id, t.last_detection))
            else:
                # Keine aktuelle Messung → Kalman Prediction
                results.append((t.id, t.kf.x[:2].flatten()))
        return results
