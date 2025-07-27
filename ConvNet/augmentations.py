import math
import random

import PIL
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image

from traco.ConvNet.helper import get_image_size, normalize_positions


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


class JointRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, label_positions):
        # Zufälligen Drehwinkel wählen
        height, width = get_image_size(image)
        rotate_degrees = random.uniform(-self.degrees, self.degrees)
        rotate_rad = math.radians(rotate_degrees)

        # Bild rotieren um das Zentrum
        image = tf.rotate(image, angle=rotate_degrees, center=[width / 2, height / 2])

        # Punkte rotieren (Koordinaten im Bereich [-1, 1], also Ursprung = Bildzentrum)
        cos_r = math.cos(rotate_rad)
        sin_r = math.sin(rotate_rad)
        rot_mat = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        rotated_positions = label_positions @ rot_mat  # Matrixmultiplikation

        return image, rotated_positions


class JointStretch:
    def __init__(self, factor, prob):
        self.factor = factor
        self.prob = prob

    def __call__(self, image, label_positions):
        height, width = get_image_size(image)

        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)

        if random.random() < self.prob:
            # Stretch in der Höhe
            new_height = int(self.factor * height)
            stretched_image = transforms.Resize((new_height, width))(image)

            # Padding berechnen, um auf Originalgröße zurückzukommen
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

            # Nur wenn gestretcht wurde, sonst kein Padding nötig
            padded_image = transforms.Pad((0, pad_top, 0, pad_bottom), fill=0)(stretched_image)

            # Labels: zuerst Stretch, dann Offset durch Padding
            labels = torch.tensor(label_positions, dtype=torch.float32)
            labels[:, 1] *= (new_height / height)  # y-Koordinaten-Stretch
            labels[:, 1] += pad_top  # y-Offset durch Padding

            return padded_image, labels

        return image, label_positions


class ResizeImagePositions:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, label_positions):
        height, width = get_image_size(image)
        if not isinstance(image, PIL.Image.Image):
            image = transforms.ToPILImage()(image)
        image = transforms.Resize(self.target_size)(image)
        label_positions = normalize_positions(label_positions, (height, width), self.target_size)

        return image, label_positions


class JointWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, positions):
        return self.transform(image), positions
