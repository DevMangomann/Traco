import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from traco.ConvNet import augmentations
from traco.ConvNet.video_tracking_dataset import VideoTrackingDataset
import matplotlib
matplotlib.use('TkAgg')


transform = augmentations.JointCompose([augmentations.ResizeImagePositions((512, 512)),
                                            augmentations.JointRandomFlip(0.5, 0.5),
                                            augmentations.JointWrapper(transforms.ToTensor())])

dataset = VideoTrackingDataset("../training", "../training", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for i, (frame, positions, num_bugs) in enumerate(dataloader):
    # Entferne Batch-Dimension: [1, C, H, W] -> [C, H, W]
    image = frame[0]
    pos = positions[0]  # Tensor mit [N, 2], angenommen normiert auf [-1, 1]

    # In numpy + Kanal nach hinten für matplotlib: [C, H, W] → [H, W, C]
    image_np = image.permute(1, 2, 0).numpy()

    # Positionen zurücktransformieren: [-1, 1] → Pixelkoordinaten
    h, w = augmentations.get_image_size(image_np)
    pos = augmentations.denormalize_positions(pos, (h, w))

    # Bild + Punkte plotten
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.scatter(pos[:, 0], pos[:, 1], c='red', s=40, label="Label Positionen")
    plt.title(f"Sample {i} mit {len(pos)} Objekten")
    plt.legend()
    plt.axis('off')
    plt.show()

    if i == 3:  # Nur die ersten 4 Bilder zeigen
        break

