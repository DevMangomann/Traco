import random

import cv2
import numpy as np


def getPixelMode(x1, x2, x3):
    # pixel as np[r,g,b]
    x1 = x1.astype(np.uint8)
    x2 = x2.astype(np.uint8)
    x3 = x3.astype(np.uint8)

    t_xor = x1 ^ x2
    t_and = x3 & t_xor
    t_and2 = x1 & x2
    t_or = t_and | t_and2
    return t_or


def getImageMode(img0, img1, img2):
    img_mode = np.zeros_like(img0)
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            p0 = img0[i, j]
            p1 = img1[i, j]
            p2 = img2[i, j]
            img_mode[i, j] = getPixelMode(p0, p1, p2)
    return img_mode


def getBackground(video_path, level):
    files = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        files.append(frame)
    result_modes = [None] * (3 ** level)

    # First Level
    for i in range(3 ** (level - 1)):
        selected_files = random.sample(files, 3)
        img0 = selected_files[0]
        img1 = selected_files[1]
        img2 = selected_files[2]
        result_modes[i] = getImageMode(img0, img1, img2)
    # Rest Levels
    for i in range(1, level):
        for j in range(3 ** (level - i - 1)):
            img0 = result_modes[3 * j]
            img1 = result_modes[3 * j + 1]
            img2 = result_modes[3 * j + 2]

            result_modes[j] = getImageMode(img0, img1, img2)

    return result_modes[0]
