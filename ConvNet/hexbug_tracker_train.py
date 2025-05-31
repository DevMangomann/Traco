import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms

import augmentations
from traco.ConvNet.hexbug_tracker import HexbugTracker
from traco.ConvNet.video_tracking_dataset import VideoTrackingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64


def match_predictions_to_targets(pred, target):
    """
    pred: Tensor der Form (max_objects, 2)
    target: Tensor der Form (n_objects, 2)

    Es werden nur die ersten n_objects Predictions zum Matching verwendet.
    """
    n_target = target.shape[0]
    pred = pred[:n_target]  # Nur die ersten n_target Predictions verwenden

    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros((0, 2)), target.new_zeros((0, 2))

    cost_matrix = torch.cdist(pred, target, p=2)

    if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
        print("Warnung: cost_matrix enthält NaN oder Inf.")
        print("Pred:", pred)
        print("Target:", target)
        raise ValueError("Ungültige cost_matrix")

    # Hungarian Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

    matched_pred = pred[row_ind]
    matched_target = target[col_ind]

    return matched_pred, matched_target


def train_loop(dataloader, model, loss_fn, optimizer):
    training_loss = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (frame, positions, num_bugs) in enumerate(dataloader):
        frame, num_bugs = frame.to(device), num_bugs.to(device)
        pred = model(num_bugs, frame).squeeze()

        batch_loss = torch.tensor(0.0, device=device)
        for i in range(frame.shape[0]):
            prediction = pred[i]
            target = positions[i].to(device)
            prediction, target = match_predictions_to_targets(prediction, target)
            batch_loss += loss_fn(prediction, target)

        batch_loss /= frame.shape[0]
        training_loss.append(batch_loss.item())

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            print(f"loss: {batch_loss.item():>7f}  [{batch * batch_size + len(frame):>5d}/{size:>5d}]")

    return training_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = []

    with torch.no_grad():
        for (frame, positions, num_bugs) in dataloader:
            frame, num_bugs = frame.to(device), num_bugs.to(device)
            pred = model(num_bugs, frame).squeeze()

            batch_loss = torch.tensor(0.0, device=device)
            for i in range(frame.shape[0]):
                prediction = pred[i]
                target = positions[i].to(device)
                prediction, target = match_predictions_to_targets(prediction, target)
                batch_loss += loss_fn(prediction, target)

            batch_loss /= frame.shape[0]
            test_loss.append(batch_loss.item())

    return test_loss


def Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path):
    if kfolds == 1:
        train_size = int(0.8 * len(dataset))
        train_set = Subset(dataset, range(train_size))
        val_set = Subset(dataset, range(train_size, len(dataset)))
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=augmentations.collate_padding)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=augmentations.collate_padding)

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
            model = HexbugTracker().to(device)
            # model.apply(init_weights_he)

            print(f"\n=== Fold {fold + 1} / {kfolds} ===")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

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

        scheduler.step(avg_validation_loss)
        print(scheduler.get_last_lr())

        print(f"Train Error: \n Avg Train loss: {avg_training_loss:.6f} \n")
        print(f"Test Error: \n Avg loss: {avg_validation_loss:.6f} \n")
        epoch_training_loss.append(avg_training_loss)
        epoch_validation_loss.append(avg_validation_loss)

        if t % 5 == 0:
            torch.save(model.state_dict(), f"models/hexbug_tracker_v{t}")

    return epoch_training_loss, epoch_validation_loss


def print_losscurve(training_losses, validation_losses, kfolds, save_path):
    plt.plot(training_losses, marker='o', label="Train Loss", color='red')
    plt.plot(validation_losses, marker='o', label="Validation Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Average Loss per Epoch for {kfolds} folds")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


def main():
    torch.cuda.empty_cache()
    model = HexbugTracker().to(device)
    # model.apply(init_weights_alexnet)
    learning_rate = 0.001
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    transform = augmentations.JointCompose([augmentations.ResizeImagePositions((512, 512)),
                                            augmentations.JointRandomFlip(0.5, 0.5),
                                            augmentations.JointWrapper(transforms.ToTensor())])

    dataset = VideoTrackingDataset("../training", "../training", transform=transform)
    dataset = Subset(dataset, range(1000))

    kfolds = 1
    epochs = 20

    model_save_path = f"./models/hexbug_tracker_folds{kfolds}_v{epochs}.pth"
    loss_save_path = f"./plots/tracking_loss_folds{kfolds}_v{epochs}.png"

    Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path)


if __name__ == '__main__':
    main()
