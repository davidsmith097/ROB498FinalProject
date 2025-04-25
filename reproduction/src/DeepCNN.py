import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import math
from torchvision.utils import make_grid
import time
import pickle
import torch.nn.functional

DEVICE = torch.device("cuda")


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='valid').to(DEVICE)
        relu1 = nn.ReLU().to(DEVICE)

        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to(DEVICE)

        conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='valid').to(DEVICE)
        relu2 = nn.ReLU().to(DEVICE)

        conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid').to(DEVICE)
        relu3 = nn.ReLU().to(DEVICE)

        dropout = nn.Dropout(0.4).to(DEVICE)

        flatten = nn.Flatten().to(DEVICE)

        fcn1 = nn.Linear(in_features=7200, out_features=16).to(DEVICE)
        relu4 = nn.ReLU().to(DEVICE)

        fcn2 = nn.Linear(in_features=16, out_features=4).to(DEVICE)
        softmax = nn.Softmax().to(DEVICE)

        self.layers = nn.Sequential(conv1, relu1, pool, conv2, relu2, pool, conv3, relu3, pool, dropout, flatten, fcn1, relu4, fcn2, softmax)

    def forward(self, x):
        return self.layers(x)