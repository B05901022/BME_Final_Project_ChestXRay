# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:25:37 2019

@author: Austin Hsu
"""

import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os

def ImageDataLoader(transform_args,
                    image_dir = '../../',
                    label_dir = '../../CheXpert-v1.0-small/train.csv',
                    batchsize = 64,
                    numworker = 12,
                    res       = 456
                    ):

    """
    train_dataset (images):
        [0]: Minibatch of Images. torch.Tensor of size (batchsize, 3, H, W)
        [1]: What subfolder the data come from. list of size (batchsize)
    train_label:
         1: Positive
         0: Negative + Unmentioned
        -1: Uncertain
    """
    
    # --- Transforms ---
    # TODO:
    # 1. Fix getattr problem
    transform_list = [transforms.RandomResizedCrop(res),
                      transforms.RandomHorizontalFlip(),
                      *[getattr(transforms, single_transform)() for single_transform in transform_args],
                      transforms.ToTensor()]
    img_transform = transforms.Compose(transform_list)

    # --- Data Collection ---
    train_dataset = ImageDataset(image_dir=image_dir,
                                 label_dir=label_dir,
                                 transform=img_transform)
    
    # --- DataLoader ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=numworker, pin_memory=True)
    return train_loader, len(train_dataset)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.transform = transform
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label = self._load_label(self.label_dir)
    def _load_label(self, label_dir):
        label = pd.read_csv(label_dir)
        label = label.fillna(0)
        label = label.replace(-1, 0.5)
        label = label.values
        return label
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.label[index][0])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = img / 255
        lbl = torch.Tensor(self.label[index][5:].astype(np.float16))
        return img, lbl
    def __len__(self):
        return self.label.shape[0]