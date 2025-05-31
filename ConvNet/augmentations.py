import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf


class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, positions):
        for t in self.transforms:
            image, positions = t(image, positions)

        return image, positions


class JointRandomFlip:
    def __init__(self, h_flip_prob=0.15, v_flip_prob=0.15):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def __call__(self, image, positions):
        image = np.array(image)
        image = transforms.ToPILImage()(image)
        if random.random() < self.h_flip_prob:
            image = tf.hflip(image)
            positions[:, 0] = -positions[:, 0]  # flip x-coordinate

        if random.random() < self.v_flip_prob:
            image = tf.vflip(image)
            positions[:, 1] = -positions[:, 1]  # flip y-coordinate

        return image, positions


class ResizeImagePositions:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, label_positions):
        height, width = get_image_size(image)
        image = transforms.ToPILImage()(image)
        image = transforms.Resize(self.target_size)(image)
        label_positions = normalize_positions(label_positions, (height, width), self.target_size)

        return image, label_positions


class JointWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, positions):
        return self.transform(image), positions


def normalize_positions(frame_labels, original_size, target_size):
    oh, ow = original_size
    th, tw = target_size
    labels = torch.tensor(frame_labels, dtype=torch.float32)
    labels[:, 0] = labels[:, 0] * (tw / ow)
    labels[:, 1] = labels[:, 1] * (th / oh)
    labels[:, 0] = labels[:, 0] / tw * 2 - 1
    labels[:, 1] = labels[:, 1] / th * 2 - 1
    return labels


def denormalize_positions(norm_labels, image_size):
    h, w = image_size
    labels = norm_labels.clone()
    labels[:, 0] = (labels[:, 0] + 1) / 2 * w
    labels[:, 1] = (labels[:, 1] + 1) / 2 * h
    return labels


def collate_padding(batch):
    images, positions_list, num_bugs = zip(*batch)

    # Alle Bilder haben Form [C, H, W], wir suchen das maximale H und W im Batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    norm_positions_list = []

    for img, positions in zip(images, positions_list):
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # Pad in Format (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded)

        positions_px = (positions + 1) / 2  # in [0, 1]
        positions_px[:, 0] *= w  # x
        positions_px[:, 1] *= h  # y

        # Schritt 2+3: Normieren auf neue Größe
        positions_px[:, 0] = (positions_px[:, 0] / max_w) * 2 - 1
        positions_px[:, 1] = (positions_px[:, 1] / max_h) * 2 - 1

        norm_positions_list.append(positions_px)

    padded_images = torch.stack(padded_images)
    bug_counts = torch.tensor(num_bugs, dtype=torch.float32)

    return padded_images, norm_positions_list, bug_counts


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
