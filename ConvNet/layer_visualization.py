import matplotlib
import matplotlib.pyplot as plt
import torch
from models import HexbugTracker, HexbugPredictor, HexbugHeatmapTracker

matplotlib.use('TkAgg')

model = HexbugHeatmapTracker()
model.load_state_dict(torch.load("model_weights/hexbug_heatmap_tracker_v50_original.pth", weights_only=True))

first_conv = model.conv[0]  # nn.Conv2d(3, 64, 11, ...)
filters = first_conv.weight.data.clone()  # Tensor mit Shape (64, 3, 11, 11)

# Wähle Anzahl der Filter, die du anzeigen willst
num_filters = 16  # z.B. 16 von 64
filters_mean = filters[32:32+num_filters].mean(dim=1)


# Normalisieren für die Anzeige
def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


# Plotte die Filter
fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # 4x4 Grid
for i, ax in enumerate(axes.flat):
    # Jeder Filter hat 3 Kanäle (RGB) → umwandeln in HWC für plt.imshow
    f = filters_mean[i]
    f = normalize(f)
    ax.imshow(f.numpy(), cmap='gray')
    ax.axis("off")

plt.suptitle("First Conv Layer Filters", fontsize=16)
plt.tight_layout()
plt.show()
