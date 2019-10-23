# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:17:10 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from   torchsummaryX import summary
from   PIL import Image
import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet

def show_image(img):
    '''
    Display a tensor image with (height, width, 1).
    '''
    img_cp = img.squeeze(0)
    plt.imshow(img_cp, cmap='gray')
    plt.axis('off')
    return

model = EfficientNet.from_pretrained('efficientnet-b5')
model = nn.Sequential(model, nn.Linear(1000,14))
img   = Image.open('./00000001_000.png').convert(mode="RGB")
transform = transforms.Compose([transforms.Resize(456),
                                transforms.ToTensor()])
img   = transform(img)
img   = img / 255
img   = img.unsqueeze(0)
print(img.shape)
pred  = model(img)

