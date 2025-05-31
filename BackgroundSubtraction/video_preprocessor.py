import numpy as np
import cv2
import os
import background_generator
import background_subtraction

videos = os.listdir("../training/")
for video in videos:
    if video.endswith(".mp4"):
        video_path = os.path.join("../training/", video)
        background = background_generator.getBackground(video_path, 3)
        output_path = f"../Background_training/{video}"
        background_subtraction.background_subtraction(background, video_path, output_path)
        print(output_path + " has been saved")
