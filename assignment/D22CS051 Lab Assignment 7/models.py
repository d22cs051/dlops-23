from typing import List
import torch
from torch import nn

class IRISModel0(nn.Module):
  def __init__(
      self,
      in_channels:int = 4,
      hidden_units:List[int] = [4,5],
      out_channels:int = 3,
      num_hidden_layers:int = 2,
      activation_funtion:nn.Module = nn.ReLU()
      ) -> None:
    '''
    args:
      in_channels: input image shape
      hidden_units: list number of hidden in neural net.
      out_channels: number of classes in the data
      num_hidden_layers: number of layers not including i/p and o/p layers
      actication_function: activation function of your choice
    '''
    super().__init__()
    self.num_hidden_layers = num_hidden_layers # no of layers in the neural net.
    layer_list = [nn.Flatten()]
    assert len(hidden_units) == num_hidden_layers, f"hidden_units: {len(hidden_units)} and num_hidden_layers: {num_hidden_layers} are not compatible"
    for i in range(num_hidden_layers):
        if i == 0:
            layer_list.append(nn.Linear(in_channels,hidden_units[i]))
            layer_list.append(activation_funtion)
        elif i == num_hidden_layers - 1:
            layer_list.append(nn.Linear(hidden_units[i-1],out_channels))
        else:
            layer_list.append(nn.Linear(hidden_units[i-1],hidden_units[i]))
            layer_list.append(activation_funtion)
    # print("layer list:")
    # print(*layer_list,sep = '\n')
    self.block = nn.Sequential(*layer_list)

  def forward(self,x:torch.Tensor):
    return self.block(x)

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