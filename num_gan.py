import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

#Random seed generation for experiment
torch.manual_seed(111)

#GPU configuration
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

#Preaparing training data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

#Loading the training data
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

#Creating the data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)