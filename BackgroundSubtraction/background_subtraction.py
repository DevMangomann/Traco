import cv2
import numpy as np


def subtractImage(background, img, threshold=30):
    if background.shape != img.shape:
        raise ValueError("Image shapes do not match!")

    diff = cv2.absdiff(background, img)
    mask = np.any(diff > threshold, axis=2).astype(np.uint8) * 255
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


def background_subtraction(background_image, video_path, output_path, threshold=30):
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    frame_height, frame_width, _ = first_frame.shape
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für MP4-Videos
    video_writer = cv2.VideoWriter(output_path, fourcc, 10, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        subtracted = subtractImage(background_image, frame, threshold)
        video_writer.write(subtracted)

    video_writer.release()
