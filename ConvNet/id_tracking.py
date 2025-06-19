import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

max_missing = 50
tracks = []
next_track_id = 0

class Hexbug_Track:
    def __init__(self, track_id, coord, frame_id):
        self.track_id = track_id
        self.coord = coord
        self.last_frame = frame_id
        self.age = 1
        self.missing = 0

    def update(self, new_coord, frame_id):
        self.coord = new_coord
        self.last_frame = frame_id
        self.age += 1
        self.missing = 0

    def mark_missing(self):
        self.missing += 1
        self.age += 1


def update_tracks(current_coords, frame_id):
    global next_track_id, tracks

    max_distance = 300  # maximale Distanz zum Zuordnen

    # Nur aktive Tracks (nicht zu lange vermisst)
    active_tracks = [t for t in tracks if t.missing <= max_missing]
    track_coords = np.array([t.coord for t in active_tracks])

    matched_tracks = set()
    matched_coords = set()

    if len(track_coords) > 0 and len(current_coords) > 0:
        cost_matrix = torch.cdist(
            torch.tensor(track_coords, dtype=torch.float32),
            torch.tensor(current_coords, dtype=torch.float32)
        ).numpy()

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < max_distance:
                active_tracks[i].update(current_coords[j], frame_id)
                matched_tracks.add(i)
                matched_coords.add(j)

        # Nicht gematchte Tracks markieren
        for i in range(len(active_tracks)):
            if i not in matched_tracks:
                active_tracks[i].mark_missing()
    else:
        # Kein Match möglich → alle aktiven Tracks markieren
        for t in active_tracks:
            t.mark_missing()

    # Neue Tracks für ungematchte Punkte erzeugen
    for i, coord in enumerate(current_coords):
        if i not in matched_coords:
            new_track = Hexbug_Track(next_track_id, coord, frame_id)
            tracks.append(new_track)
            next_track_id += 1

    # Optional: inaktive Tracks vollständig entfernen
    # tracks = [t for t in tracks if t.missing <= max_missing]

    # Rückgabe: aktuelle gültige Tracks
    return [(t.track_id, t.coord) for t in tracks if t.missing == 0]

