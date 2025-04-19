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

class DiseasedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device=DEVICE)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2]).to(device=DEVICE) # Get everything up to the avg pooling layer
        # Freeze the base
        for param in resnet50.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        # for param in resnet50.layer4.parameters():
        #     param.requires_grad = True

        conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1).to(device=DEVICE)
        relu1 = nn.ReLU().to(device=DEVICE)

        conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1).to(device=DEVICE)
        batchnorm = nn.BatchNorm2d(256).to(device=DEVICE)
        relu2 = nn.ReLU().to(device=DEVICE)

        pool = nn.AdaptiveAvgPool2d((1, 1)).to(DEVICE)
        flatten = nn.Flatten().to(DEVICE)

        fcn1 = nn.Linear(256, 64).to(device=DEVICE)
        relu3 = nn.ReLU().to(device=DEVICE)

        fcn2 = nn.Linear(64, 38).to(device=DEVICE)
        softmax = nn.Softmax().to(DEVICE)

        self.layers = nn.Sequential(conv1, relu1, conv2, batchnorm, relu2, pool, flatten, fcn1, relu3, fcn2, softmax).to(DEVICE)

        # print(self.backbone)
        # print()
        # print()
        # print(self.layers)

    def forward(self, x):
        x = self.backbone(x).to(DEVICE)
        return self.layers(x).to(DEVICE)