import torch
from torch import nn

import math
import matplotlib.pyplot as plt

#Random generator seed for replicability
torch.manual_seed(111)

#Preparing training data
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

#Examine training data
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()