import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from traco.ConvNet import augmentations, helper
from traco.ConvNet.datasets import VideoTrackingDataset, HeatmapDataset
from traco.ConvNet.helper import get_image_size, denormalize_positions, collate_padding
from traco.ConvNet.models import HexbugHeatmapTracker

matplotlib.use('TkAgg')
target_size = (256, 256)

transform = augmentations.JointCompose([augmentations.ResizeImagePositions(target_size),
                                        augmentations.JointRandomFlip(0.5, 0.5),
                                        augmentations.JointWrapper(transforms.ToTensor()),
                                        augmentations.JointWrapper(
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)),
                                        augmentations.JointWrapper(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))])

model = HexbugHeatmapTracker()
model.load_state_dict(torch.load("model_weights/hexbug_heatmap_tracker_v50_original.pth"))
model.eval()

dataset = HeatmapDataset("../training", "../training", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for i, (frame, heatmap) in enumerate(dataloader):

    with torch.no_grad():
        heatmap_prediction = model(frame)  # (B, 1, H, W)

    # Daten von GPU holen und Tensoren auf erstes Bild im Batch beschränken
    image = frame[0]
    gt_heat = heatmap[0].squeeze()
    pred_heat = heatmap_prediction[0].squeeze()
    coords = helper.coords_from_heatmap(pred_heat, 3, (256, 256))
    print(coords)

    # Bild umwandeln: (C, H, W) -> (H, W, C)
    image_np = image.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)  # Optional: Normalisierung (falls nötig)

    # Plot
    plt.figure(figsize=(12, 4))

    # Originalbild
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original Frame")
    plt.axis("off")

    # Ground Truth Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(gt_heat.numpy(), cmap='hot')
    plt.title("Label Heatmap")
    plt.axis("off")

    # Predicted Heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(pred_heat.numpy(), cmap='hot')
    plt.title("Predicted Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if i == 100:
        break