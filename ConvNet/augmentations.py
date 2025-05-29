import random

import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F


class JointRandomFlip:
    def __init__(self, h_flip_prob=0.15, v_flip_prob=0.15):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def __call__(self, image, positions):
        _, height, width = image.shape  # image is (C, H, W)
        if random.random() < self.h_flip_prob:
            image = tf.hflip(image)
            positions[:, 0] = -positions[:, 0]  # flip x-coordinate

        if random.random() < self.v_flip_prob:
            image = tf.vflip(image)
            positions[:, 1] = -positions[:, 1]  # flip y-coordinate

        return image, positions


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
