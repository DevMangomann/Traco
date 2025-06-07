import random

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

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
