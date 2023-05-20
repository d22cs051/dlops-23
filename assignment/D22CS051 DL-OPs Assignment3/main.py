# pytorch libs

from plotting import plot_curves
import torch
from torch import nn
import torchvision
import os

# numpy
import numpy as np

# torch metrics

from torchmetrics import Accuracy

from torch.utils.data import DataLoader

from torchvision import transforms


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] current used device: {device}")


# Getting DATASET

# defining transform
transform_train = transforms.Compose([
    # transforms.Resize(224),
    transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(
    download=True,
    root='data/',
    train=True,
    transform=transform_train,
)

test_dataset = torchvision.datasets.CIFAR10(
    download=True,
    root='data/',
    train=False,
    transform=transform_test,
)

# creating the subset for train and test datasets
# following tutorial:- https://ravimashru.dev/blog/2021-09-26-pytorch-subset/
idx0, idx2, idx4, idx6, idx8 = torch.Tensor(train_dataset.targets) == 0, torch.Tensor(train_dataset.targets) == 2, torch.Tensor(train_dataset.targets) == 4, torch.Tensor(train_dataset.targets) == 6, torch.Tensor(train_dataset.targets) == 8
idx0_test, idx2_test, idx4_test, idx6_test, idx8_test = torch.Tensor(test_dataset.targets) == 0, torch.Tensor(test_dataset.targets) == 2, torch.Tensor(test_dataset.targets) == 4, torch.Tensor(test_dataset.targets) == 6, torch.Tensor(test_dataset.targets) == 8
train_mask = idx0 | idx2 | idx4 | idx6 | idx8
test_mask = idx0_test | idx2_test | idx4_test | idx6_test | idx8_test

train_indices = train_mask.nonzero().reshape(-1)
test_indices = test_mask.nonzero().reshape(-1)

# defining subset
from torch.utils.data import Subset
train_subset = Subset(train_dataset,train_indices)
test_subset = Subset(test_dataset,test_indices)

# converting data into torch dataloader
import os
BATCH_SIZE = 64
NUM_WORKERS = 2

train_dataloader = DataLoader(
    train_subset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    test_subset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers=NUM_WORKERS
)

# Testing one batch from train dataloader
# print("\n\n1st batch form train dataloader")
# print(next(iter(train_dataloader))[1])

# Importing model
from models import EarlyStopping, MyViT
def get_model():
    # model = ViT(
    #     in_channels=3,
    #     img_size=32,
    #     patch_size=8,
    #     num_classes=5,
    #     num_heads=6,
    #     num_transformer_layers=4,
    #     mlp_size=128,
    #     embedding_dim=24,
    #     mlp_dropout=0.2,
    #     embedding_dropout=0.2
    # )
    model = MyViT()
    return model
    


# Train Info
# Early stopping
early_stopping = EarlyStopping(tolerance=3, min_delta=0.001)

# Training model on train data
from engine import train
from timeit import default_timer as timer 

# Hyperparms
lr = [1e-3,1e-4] # learning rate
betas=[(0.8, 0.888)] # coefficients used for computing running averages of gradient and its square
eps = [1e-8] # term added to the denominator to improve numerical stability
weight_decay = [1e-3] # weight decay (L2 penalty)

# init. epochs
NUM_EPOCHS = [10,15]

parms_combs = [(l,b,e,w_d,epochs) for l in lr for b in betas for e in eps for w_d in weight_decay for epochs in NUM_EPOCHS]

# init. loss function, accuracy function and optimizer
loss_fn = nn.CrossEntropyLoss()
acc_fn = Accuracy(task="multiclass", num_classes=10).to(device=device)

cur,total = 1, len(lr)*len(betas)*len(eps)*len(weight_decay)*len(NUM_EPOCHS)
for h_parms in parms_combs:
  ### INIT MODEL STARTS ###
  # traning same model for each parms
  model = get_model().to(device=device)
  ### INIT MODEL END ###

  optimizer = torch.optim.Adam(
      params=model.parameters(), lr=h_parms[0], betas=h_parms[1], eps=h_parms[2],weight_decay=h_parms[3]
  )

  # importing and init. the timer for checking model training time
  from timeit import default_timer as timer

  start_time = timer()
  print(f"current exp / total: {cur} / {total}")
  print(f"Training with: lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")
  
  model_results = train(
      model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      acc_fn=acc_fn,
      epochs=h_parms[4],
      save_info=f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}",
      device=device
  )

  # end timer
  end_time = timer()
  # printing time taken
  print(f"total training time: {end_time-start_time:.3f} sec.")
  # print("model stats:")
  # print(model_0_results)
  print(f"LOSS & Accuracy Curves\n"
        f"lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")
  plot_curves(model_results,f"{model.__class__.__name__}_epoch_{h_parms[4]}_optim_adam_"
              +
              f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}")
  cur+=1
  print()