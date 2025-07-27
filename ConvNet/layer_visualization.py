import matplotlib
import matplotlib.pyplot as plt
import torch

from models import HeatmapTracker

matplotlib.use('TkAgg')

model = HeatmapTracker()
model.load_state_dict(torch.load("model_weights/heatmap_tracker_v50.pth", weights_only=True))

first_conv = model.conv[0]  # first conv layer
filters = first_conv.weight.data.clone()

num_filters = 16
filters_mean = filters[32:32 + num_filters].mean(dim=1)


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    f = filters_mean[i]
    f = normalize(f)
    ax.imshow(f.numpy(), cmap='gray')
    ax.axis("off")

plt.suptitle("First Conv Layer Filters", fontsize=16)
plt.tight_layout()
plt.show()
