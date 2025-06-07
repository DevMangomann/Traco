import numpy as np
import torch
import torch.nn.functional as F


def normalize_positions(frame_labels, original_size, target_size):
    oh, ow = original_size
    th, tw = target_size
    labels = torch.tensor(frame_labels, dtype=torch.float32)
    labels[:, 0] = labels[:, 0] * (tw / ow)
    labels[:, 1] = labels[:, 1] * (th / oh)
    labels[:, 0] = labels[:, 0] / tw * 2 - 1
    labels[:, 1] = labels[:, 1] / th * 2 - 1
    return labels


def denormalize_positions(labels, original_size, target_size):
    th, tw = target_size
    oh, ow = original_size

    # labels = norm_labels.clone()
    # Von [-1, 1] → [0, target_width/height]
    labels[:, 0] = (labels[:, 0] + 1) / 2 * tw
    labels[:, 1] = (labels[:, 1] + 1) / 2 * th

    # Rückskalierung von target_size → original_size
    labels[:, 0] = labels[:, 0] * (ow / tw)
    labels[:, 1] = labels[:, 1] * (oh / th)

    return labels


def coords_from_heatmap(heatmap, num_bugs, original_size):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    H_map, W_map = get_image_size(heatmap)
    H_orig, W_orig = original_size

    flat = heatmap.flatten()
    topk_indices = np.argpartition(flat, -num_bugs)[-num_bugs:]
    topk_indices = topk_indices[np.argsort(-flat[topk_indices])]  # Sortiert nach Score

    # Index zu 2D-Koordinate und skalieren
    coords = []
    for idx in topk_indices:
        y, x = np.unravel_index(idx, heatmap.shape)
        x_scaled = int(x * W_orig / W_map)
        y_scaled = int(y * H_orig / H_map)
        coords.append((x_scaled, y_scaled))

    return coords


def collate_padding(batch):
    images, positions_list, num_bugs = zip(*batch)

    # Maximale Höhe und Breite bestimmen
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []

    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # Padding: (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded)

    padded_images = torch.stack(padded_images)
    bug_counts = torch.tensor(num_bugs, dtype=torch.float32)

    return padded_images, list(positions_list), bug_counts


def get_image_size(img):
    """
    Gibt (Höhe, Breite) für ein Bild zurück – egal ob NumPy, PIL oder torch.Tensor.
    """
    import numpy as np
    from PIL import Image
    import torch

    if isinstance(img, np.ndarray):
        # NumPy: (H, W, C) oder (H, W)
        return img.shape[:2]
    elif isinstance(img, Image.Image):
        # PIL: .size gibt (W, H)
        w, h = img.size
        return h, w
    elif isinstance(img, torch.Tensor):
        # torch.Tensor: (C, H, W) oder (H, W)
        if img.ndim == 3:
            return img.shape[1], img.shape[2]
        elif img.ndim == 2:
            return img.shape[0], img.shape[1]
        else:
            raise ValueError(f"Unerwartete Tensor-Dimensionen: {img.shape}")
    else:
        raise TypeError(f"Nicht unterstützter Bildtyp: {type(img)}")


def generate_heatmap(size, normalized_points, sigma=3):
    H, W = size
    heatmap = np.zeros((H, W), dtype=np.float32)
    normalized_points = np.array(normalized_points)

    # Normierte Koordinaten in Pixelkoordinaten umwandeln
    # x in [-1, 1] => px = (x + 1) * W/2
    # y in [-1, 1] => py = (y + 1) * H/2
    for x_norm, y_norm in normalized_points:
        x = (x_norm + 1) * 0.5 * W
        y = (y_norm + 1) * 0.5 * H

        if x < 0 or y < 0 or x >= W or y >= H:
            continue  # außerhalb ignorieren

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)

    return torch.from_numpy(heatmap).float()
