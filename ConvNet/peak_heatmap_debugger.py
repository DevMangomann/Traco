import cv2
import matplotlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from traco.ConvNet import helper
from traco.ConvNet.models import HexbugHeatmapTracker

matplotlib.use('TkAgg')
plt.ion()  # Interaktiver Modus

video_path = "../leaderboard_data/test005.mp4"
cap = cv2.VideoCapture(video_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_model = HexbugHeatmapTracker()
tracking_model.load_state_dict(
    torch.load("model_weights/heatmap_tracker_training_plus_v60.pth", weights_only=True, map_location=device))
tracking_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Initiale Visualisierung vorbereiten
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
img1 = axs[0].imshow(np.zeros((256, 256, 3)))
axs[0].set_title("Frame (resized)")
axs[0].axis("off")

img2 = axs[1].imshow(np.zeros((256, 256)), cmap="hot", vmin=0, vmax=1)
axs[1].set_title("Predicted Heatmap")
axs[1].axis("off")

img3 = axs[2].imshow(np.zeros((256, 256, 3)))
scatter = axs[2].scatter([], [], c="cyan", s=40, marker="x", label="Peaks")
axs[2].set_title("Peaks on Frame")
axs[2].legend()
axs[2].axis("off")

paused = False
frame_idx = 0


def on_key(event):
    global paused
    if event.key == ' ':
        paused = not paused
    elif event.key == 'q':
        plt.close()


fig.canvas.mpl_connect('key_press_event', on_key)

while plt.fignum_exists(fig.number):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        predict_image = transform(image).unsqueeze(0)

        with torch.no_grad():
            heatmap = tracking_model(predict_image)[0, 0]

        num_bugs = torch.tensor([2])
        coords = helper.coords_from_heatmap(heatmap, num_bugs, (256, 256))
        print(coords)
        heatmap_np = heatmap.cpu().numpy()
        heatmap_vis = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

        # Bild vorbereiten
        image_np = np.asarray(image) / 255.0

        # Update der Plots
        img1.set_data(image_np)
        img2.set_data(heatmap_np)
        img3.set_data(image_np)

        # Scatterpunkte aktualisieren
        scatter.remove()
        scatter = axs[2].scatter(coords[:, 0], coords[:, 1], c="cyan", s=40, marker="x", label="Peaks")

        fig.canvas.draw()

        frame_idx += 1
        plt.pause(0.001)
    else:
        plt.pause(0.1)



cap.release()
cv2.destroyAllWindows()
