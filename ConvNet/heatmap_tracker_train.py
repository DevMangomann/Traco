import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torchvision.transforms import transforms

import augmentations
import helper
from traco.ConvNet.datasets import HeatmapTrackingDataset
from traco.ConvNet.models import HeatmapTracker, BigHeatmapTracker, BiggerHeatmapTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 16


def train_loop(dataloader, model, loss_fn, optimizer):
    training_loss = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (frame, heatmap) in enumerate(dataloader):
        frame, heatmap = frame.to(device), heatmap.to(device)
        pred = model(frame).squeeze()

        loss = loss_fn(pred, heatmap)
        training_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * batch_size + len(frame):>5d}/{size:>5d}]")

    return training_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = []

    with torch.no_grad():
        for (frame, heatmap) in dataloader:
            frame, heatmap = frame.to(device), heatmap.to(device)
            pred = model(frame).squeeze()

            loss = loss_fn(pred, heatmap)

            test_loss.append(loss.item())

    return test_loss


def Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path):
    if kfolds == 1:
        train_size = int(1.0 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                      prefetch_factor=4)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                    prefetch_factor=4)

        epoch_training_loss, epoch_validation_loss = epoch_training(epochs, train_dataloader, val_dataloader, model,
                                                                    loss_fn, optimizer, scheduler)
        print_losscurve(epoch_training_loss, epoch_validation_loss, kfolds, loss_save_path)
        torch.save(model.state_dict(), model_save_path)

        print("Done!")
    else:
        indices = list(range(len(dataset)))
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        fold_train_loss = [0.0] * epochs
        fold_val_loss = [0.0] * epochs
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            model = BiggerHeatmapTracker().to(device)
            # model.apply(init_weights_he)

            print(f"\n=== Fold {fold + 1} / {kfolds} ===")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True,
                                          prefetch_factor=4)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                        prefetch_factor=4)

            epoch_training_loss, epoch_validation_loss = epoch_training(epochs, train_dataloader, val_dataloader, model,
                                                                        loss_fn, optimizer, scheduler)
            fold_train_loss[:] += epoch_training_loss[:]
            fold_val_loss[:] += epoch_validation_loss[:]

            print("Done!")

        fold_train_loss = np.array(fold_train_loss) / float(kfolds)
        fold_val_loss = np.array(fold_val_loss) / float(kfolds)
        print_losscurve(fold_train_loss, fold_val_loss, kfolds, loss_save_path)
        torch.save(model.state_dict(), model_save_path)


def epoch_training(epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler):
    epoch_training_loss = []
    epoch_validation_loss = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        training_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        validation_loss = test_loop(val_dataloader, model, loss_fn)
        avg_training_loss = sum(training_loss) / len(training_loss)
        avg_validation_loss = sum(validation_loss) / len(validation_loss)

        scheduler.step()
        print(scheduler.get_last_lr())

        print(f"Train Error: \n Avg Train loss: {avg_training_loss:.6f} \n")
        print(f"Test Error: \n Avg loss: {avg_validation_loss:.6f} \n")
        epoch_training_loss.append(avg_training_loss)
        epoch_validation_loss.append(avg_validation_loss)

        if t % 5 == 0:
            torch.save(model.state_dict(), f"model_weights/bigger_heatmap_tracker_v{t}.pth")

    return epoch_training_loss, epoch_validation_loss


def print_losscurve(training_losses, validation_losses, kfolds, save_path):
    plt.plot(training_losses, marker='o', label="Train Loss", color='red')
    # plt.plot(validation_losses, marker='o', label="Validation Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Average Loss per Epoch for {kfolds} folds")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


def main():
    torch.cuda.empty_cache()
    model = BiggerHeatmapTracker().to(device)
    learning_rate = 0.001
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95,
    )

    transform = augmentations.JointCompose([augmentations.JointStretch(0.33, 0.1),
                                            augmentations.ResizeImagePositions((512, 512)),
                                            # augmentations.JointWrapper(transforms.ToTensor()),
                                            augmentations.JointRandomFlip(0.5, 0.5),
                                            augmentations.JointRotation(180.0),
                                            augmentations.JointWrapper(transforms.ToTensor()),
                                            augmentations.JointWrapper(
                                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                       hue=0.02)),
                                            augmentations.JointWrapper(
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
                                            ])

    # tmpdir = os.environ.get("TMPDIR", "/tmp")  # fallback zu /tmp für lokale Tests
    # lable_path = os.path.join(tmpdir, "training")
    # data_path = os.path.join(tmpdir, "training")

    dataset = HeatmapTrackingDataset("../training", "../training", transform=transform)
    dataset = Subset(dataset, range(200))

    kfolds = 1
    epochs = 2

    model_save_path = f"./model_weights/bigger_heatmap_tracker_folds{kfolds}_v{epochs}.pth"
    loss_save_path = f"./plots/bigger_heatmap_tracker_folds{kfolds}_v{epochs}_loss.png"

    Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path)


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    main()
