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

def add_bias_channel(x, dim=1):
    one_size = list(x.size())
    one_size[dim] = 1
    one = x.new_ones(one_size)
    return torch.cat((x, one), dim)

def flatten(x, keepdims=False):
    """
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y

def gem(x, p=3, eps=1e-6, clamp=True, add_bias=False, keepdims=False):
    if p == math.inf or p is 'inf':
        x = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    elif p == 1 and not (torch.is_tensor(p) and p.requires_grad):
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    else:
        if clamp:
            x = x.clamp(min=eps)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    if add_bias:
        x = add_bias_channel(x)
    if not keepdims:
        x = flatten(x)
    return x


# --- Pretrained Model ---
#model = EfficientNet.from_pretrained('efficientnet-b5')
model  = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
# For torch.hub usages, change C:\ProgramData\Anaconda3\Lib\site-packages\torch\hub.py
# from os.remove(path) to pass

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
print(img.shape)
pred  = model(img)

