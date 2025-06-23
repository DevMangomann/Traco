import os

import numpy as np
import torch
import torch.optim.lr_scheduler
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torchvision import transforms

from traco.ConvNet.datasets import VideoPredictingDataset
from traco.ConvNet.models import HexbugPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64


def train_loop(dataloader, model, loss_fn, optimizer):
    training_loss = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        training_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(f"Output: {pred[0]}  Label: {y[0]}")
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return training_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    test_loss = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss.append(loss.item())

            predicted_class = torch.argmax(pred, dim=1)
            correct += (predicted_class == y).sum().item()

    correct /= size
    avg_test = sum(test_loss) / len(test_loss)
    print(device)
    print(f"Test Error: \n Accuracy: {100 * correct:.1f}%, Avg loss: {avg_test:.6f} \n")
    return test_loss, correct


def Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path):
    if kfolds == 1:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                      prefetch_factor=4, persistent_workers=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                    prefetch_factor=4, persistent_workers=True)

        epoch_training_loss, epoch_validation_loss, epoch_accuracy = epoch_training(epochs, train_dataloader,
                                                                                    val_dataloader, model,
                                                                                    loss_fn, optimizer, scheduler)
        print_losscurve(epoch_training_loss, epoch_validation_loss, epoch_accuracy, kfolds, loss_save_path)
        torch.save(model.state_dict(), model_save_path)

        print("Done!")
    else:
        indices = list(range(len(dataset)))
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        fold_train_loss = [0.0] * epochs
        fold_val_loss = [0.0] * epochs
        fold_accuracy = [0.0] * epochs
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            model = HexbugPredictor().to(device)
            # model.apply(init_weights_he)

            print(f"\n=== Fold {fold + 1} / {kfolds} ===")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            epoch_training_loss, epoch_validation_loss, epoch_accuracy = epoch_training(epochs, train_dataloader,
                                                                                        val_dataloader, model,
                                                                                        loss_fn, optimizer, scheduler)
            fold_train_loss[:] += epoch_training_loss[:]
            fold_val_loss[:] += epoch_validation_loss[:]
            fold_accuracy[:] += epoch_accuracy[:]

            print("Done!")

        fold_train_loss = np.array(fold_train_loss) / float(kfolds)
        fold_val_loss = np.array(fold_val_loss) / float(kfolds)
        fold_accuracy = np.array(fold_accuracy) / float(kfolds)
        print_losscurve(fold_train_loss, fold_val_loss, fold_accuracy, kfolds, loss_save_path)
        torch.save(model.state_dict(), model_save_path)


def epoch_training(epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler):
    epoch_training_loss = []
    epoch_validation_loss = []
    epoch_accuracy = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        training_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        validation_loss, accuracy = test_loop(val_dataloader, model, loss_fn)
        avg_training_loss = sum(training_loss) / len(training_loss)
        avg_validation_loss = sum(validation_loss) / len(validation_loss)

        scheduler.step()
        print(f"Train Error: \n Avg Train loss: {avg_training_loss:.6f} \n")
        print(f"last learning rate: {scheduler.get_last_lr()} \n")

        epoch_training_loss.append(avg_training_loss)
        epoch_validation_loss.append(avg_validation_loss)
        epoch_accuracy.append(accuracy)

        if t % 5 == 0:
            torch.save(model.state_dict(), f"model_weights/hexbug_predictor_v{t}")

    return epoch_training_loss, epoch_validation_loss, epoch_accuracy


def print_losscurve(training_losses, validation_losses, accuracy, kfolds, save_path):
    plt.plot(training_losses, marker='o', label="Train Loss", color='red')
    # plt.plot(validation_losses, marker='o', label="Validation Loss", color='blue')
    plt.plot(accuracy, marker='o', label="Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Average Loss per Epoch for {kfolds} folds")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


def main():
    torch.cuda.empty_cache()
    model = HexbugPredictor().to(device)
    # model.apply(init_weights_alexnet)
    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40, 60],
        gamma=0.1
    )

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
                                    transforms.RandomRotation(90),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # tmpdir = os.environ.get("TMPDIR", "/tmp")  # fallback zu /tmp für lokale Tests
    # lable_path = os.path.join(tmpdir, "training")
    # data_path = os.path.join(tmpdir, "training")

    dataset = VideoPredictingDataset("../training", "../training", transform=transform)
    # dataset = Subset(dataset, range(2000))

    kfolds = 1
    epochs = 80

    model_save_path = f"./model_weights/hexbug_predictor_folds{kfolds}_v{epochs}.pth"
    loss_save_path = f"./plots/predicting_loss_folds{kfolds}_v{epochs}.png"

    Kfold_training(kfolds, epochs, dataset, model, loss_fn, optimizer, scheduler, model_save_path, loss_save_path)


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    main()
