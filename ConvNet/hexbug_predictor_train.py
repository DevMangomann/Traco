import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.model_selection import KFold

from traco.ConvNet.hexbug_predictor import HexbugPredictor
from traco.ConvNet.video_predicting_dataset import VideoPredictingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#model = HexbugPredictor().to(device)
batch_size = 64
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.15),
    transforms.RandomVerticalFlip(0.15)
])
dataset = VideoPredictingDataset("../training", "../training", transform=transform)
#dataset = Subset(dataset, range(200))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

k_folds = 4
indices = list(range(len(dataset)))
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

loss_fn = nn.MSELoss()


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    training_loss = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        training_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return training_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()

            pred_rounded = torch.round(pred)
            correct += (pred_rounded == y).sum().item()  # z. B. 3 == 3

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100 * correct:.1f}%, Avg loss: {test_loss:.6f} \n")
    return test_loss


fold_training_losses = []
fold_validation_losses = []
epochs = 5

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n=== Fold {fold + 1} / {k_folds} ===")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = HexbugPredictor().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    fold_train_loss = []
    fold_val_loss = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        training_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        avg_training_loss = sum(training_loss) / len(training_loss)
        fold_train_loss.append(avg_training_loss)
        avg_validation_loss = test_loop(val_dataloader, model, loss_fn)
        fold_val_loss.append(avg_validation_loss)

    fold_training_losses.append(fold_train_loss)
    fold_validation_losses.append(fold_val_loss)

    torch.save(model.state_dict(), f'models/hexbug_predictor_fold{fold}_v{epochs}.pth')

print("Done!")

avg_train = np.mean(fold_training_losses, axis=0)
avg_val = np.mean(fold_validation_losses, axis=0)
plt.plot(avg_train, marker='o', label="Train Loss", color='red')
plt.plot(avg_val, marker='o', label="Validation Loss", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("./plots/train_cross_validation_loss.png")
