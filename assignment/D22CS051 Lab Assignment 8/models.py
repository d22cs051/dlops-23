from typing import List
import torch
from torch import nn
import numpy as np
from torchvision.models import resnet18

def get_resnet_18():
    model = resnet18()
    model.fc = nn.Linear(in_features=512,out_features=10,bias=True)
    return model


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta: 
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
