import matplotlib
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from traco.ConvNet import augmentations
from traco.ConvNet.datasets import VideoTrackingDataset, HeatmapDataset
from traco.ConvNet.helper import get_image_size, denormalize_positions, collate_padding

matplotlib.use('TkAgg')
target_size = (512, 512)

transform = augmentations.JointCompose([augmentations.ResizeImagePositions(target_size),
                                        augmentations.JointRandomFlip(0.5, 0.5),
                                        augmentations.JointWrapper(transforms.ToTensor()),
                                        augmentations.JointWrapper(
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)),
                                        augmentations.JointWrapper(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))])

dataset = HeatmapDataset("../training", "../training", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for i, (frame, heatmap) in enumerate(dataloader):
    # Entferne Batch-Dimension: [1, C, H, W] -> [C, H, W]
    image = frame[0]
    heat = heatmap[0]  # Tensor mit [N, 2], angenommen normiert auf [-1, 1]

    # In numpy + Kanal nach hinten für matplotlib: [C, H, W] → [H, W, C]
    image_np = image.permute(1, 2, 0).numpy()

    # Positionen zurücktransformieren: [-1, 1] → Pixelkoordinaten
    h, w = get_image_size(image_np)
    #pos = denormalize_positions(pos, (h, w), target_size)

    # Bild + Punkte plotten
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    #plt.scatter(pos[:, 0], pos[:, 1], c='red', s=40, label="Label Positionen")
    #plt.title(f"Sample {i} mit {len(pos)} Objekten")
    plt.legend()
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(heat)
    # plt.scatter(pos[:, 0], pos[:, 1], c='red', s=40, label="Label Positionen")
    # plt.title(f"Sample {i} mit {len(pos)} Objekten")
    plt.legend()
    plt.axis('off')
    plt.show()

    if i == 100:  # Nur die ersten 4 Bilder zeigen
        break
