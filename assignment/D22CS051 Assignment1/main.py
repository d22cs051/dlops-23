# pytorch libs
import torch
from torch import nn
import torchvision
import os

# numpy
import numpy as np

# torch metrics
try:
    import torchmetrics
except:
    os.system("pip3 -q install torchmetrics")
finally:
    from torchmetrics import Accuracy


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] current used device: {device}")


# Getting DATASET
from sklearn.datasets import load_iris
data = load_iris()
X,y = data.data, data.target
class_names = data.target_names
# print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}, class_names shape: {class_names.shape}")


# Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Converting data into torch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


# converting data into torch dataloader
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train,y_train) # creates train dataset
train_dataloader = DataLoader(train_dataset) # create your train dataloader

test_dataset = TensorDataset(X_test,y_test) # creates test dataset
test_dataloader = DataLoader(test_dataset) # create your test dataloader


# Importing model
from models import IRISModel0, EarlyStopping
model0 = IRISModel0(in_channels=4,out_channels=3).to(device)

# Early stopping
early_stopping = EarlyStopping(tolerance=3, min_delta=0.005)

# Training model on train data
from engine import train
train_results = train(
    model=model0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model0.parameters(), lr=0.001),
    acc_fn=Accuracy('multiclass', num_classes=3).to(device),
    device=device,
    early_stopping = early_stopping,
    epochs=100
)

# plotting results
from plotting import plot_curves
plot_curves(train_results)