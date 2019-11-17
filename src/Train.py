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
import math

from efficientnet_pytorch import EfficientNet

def show_image(img):
    '''
    Display a tensor image with (3, height, width).
    '''
    plt.figure()
    img_cp = img.transpose(0,-1).transpose(0,1)
    plt.imshow(img_cp, cmap='gray')
    return

# --- Pretrained Model ---
"""
For torch.hub usages, change C:\ProgramData\Anaconda3\Lib\site-packages\torch\hub.py
from os.remove(path) to pass
"""
#model = EfficientNet.from_pretrained('efficientnet-b5')
#model  = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')

# --- New Model ---
model = EfficientNet.from_name('efficientnet-b5')

# --- Transforms ---
transform = transforms.Compose([transforms.RandomResizedCrop(456),
                                transforms.ToTensor()])
transform2 = transforms.Compose([transforms.ToTensor()])

img   = Image.open('./00000001_000.png').convert(mode="RGB")
show_image(transform2(img))
img   = transform(img)
show_image(img)
img   = img / 255
img   = img.unsqueeze(0)


