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
transform = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(
    download=True,
    root='data/',
    train=True,
    transform=transform,
)

test_dataset = torchvision.datasets.CIFAR10(
    download=True,
    root='data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)


# converting data into torch dataloader
import os
BATCH_SIZE = 64
NUM_WORKERS = 2 # setting the numworkers to half of cpu counts

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers=NUM_WORKERS
)

# Importing model
from models import EarlyStopping, CustomArchModified

def get_model():
    return CustomArchModified()

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

  # creating profiler
#   prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(f'log/custom_arch_{cur}'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#         )
  
  # importing and init. the timer for checking model training time
  from timeit import default_timer as timer

#   start_time = timer()
  print(f"current exp / total: {cur} / {total}")
  print(f"Training with: lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")
  
  # profiling only does not train the model
#   print(f"\n### Profiling the model ###\n")
#   model_results_prof = train(
#       model=model,
#       train_dataloader=train_dataloader,
#       test_dataloader=test_dataloader,
#       optimizer=optimizer,
#       loss_fn=loss_fn,
#       acc_fn=acc_fn,
#       epochs=h_parms[4],
#       save_info=f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}_profiler_curves",
#       device=device,
#       profiler=prof
#   )
  # end timer
#   end_time = timer()
#   print(f"total profiling time: {end_time-start_time:.3f} sec.")

  # actual train the model
  start_time = timer()
  print(f"\n### Training the model ###\n")
  model_results = train(
      model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      acc_fn=acc_fn,
      epochs=h_parms[4],
      save_info=f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}",
      device=device,
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