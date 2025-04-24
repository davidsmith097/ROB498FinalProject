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


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()

        conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1).to(DEVICE)
        relu1 = nn.LeakyReLU().to(DEVICE)
        pool1 = nn.MaxPool2d(kernel_size=2).to(DEVICE)

        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1).to(DEVICE)
        relu2 = nn.LeakyReLU().to(DEVICE)
        pool2 = nn.MaxPool2d(kernel_size=2).to(DEVICE)

        conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1).to(DEVICE)
        relu3 = nn.LeakyReLU().to(DEVICE)
        Apool = nn.AdaptiveAvgPool2d(1).to(DEVICE)
        
        flatten = nn.Flatten().to(DEVICE)

        fcn1 = nn.Linear(128, 38).to(DEVICE)
        # softmax = nn.Softmax().to(DEVICE)

        self.layers = nn.Sequential(conv1, relu1, pool1, conv2, relu2, pool2, conv3, relu3, Apool, flatten, fcn1).to(DEVICE)

    def forward(self, x):
        return self.layers(x).to(DEVICE)