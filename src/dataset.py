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
                    res       = 456,
                    train     = True
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
    transform_list = [getattr(transforms, transform_args[0])(res),
                      *[getattr(transforms, single_transform)() for single_transform in transform_args[1:]]]
    img_transform = transforms.Compose(transform_list)

    # --- Data Collection ---
    train_dataset = ImageDataset(image_dir=image_dir,
                                 label_dir=label_dir,
                                 transform=img_transform,
                                 train=train)
    
    # --- DataLoader ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=numworker, pin_memory=True)
    return train_loader, len(train_dataset)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform, train=True):
        self.transform = transform
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.trainmode = train
        self.label = self._load_label(self.label_dir)
    def _load_label(self, label_dir):
        label = pd.read_csv(label_dir)
        label['Lung Lesion'] = label['Lung Lesion'].fillna(1)
        label = label.fillna(0)
        label = label.replace(-1, 1)
        label = label.values
        if self.trainmode:
            label = self._label_fix(label)
        return label
    def _label_fix(self, label):
        """
        fix conflicts from:
            (1) Lung Opacity
            (2) Enlarged Cardiomediastinum
        """
        for sample in range(label.shape[0]):
            if (label[sample][10] == 1 or label[sample][11] == 1 or label[sample][12] == 1 or label[sample][13] == 1):
                # (1) Lung Lesion
                # (2) Edema
                # (3) Consolidation
                # (4) Pneumonia
                # (5) Atelectasis
                # --> Lung Opacity
                label[sample][8] = 1
            if (label[sample][7] == 1):
                # (1) Cardiomegaly
                # --> Enlarged Cardiomediastinum
                label[sample][6] = 1
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

# =================================================================================================

def TestImageDataLoader(image_dir, input_dir, res):
    
    # --- Transforms ---
    transform_list = [transforms.Resize(res),
                      transforms.ToTensor()]
    img_transform = transforms.Compose(transform_list)

    # --- Data Collection ---
    train_dataset = TestImageDataset(image_dir=image_dir, input_dir=input_dir, transform=img_transform)
    
    # --- DataLoader ---
    train_loader = torch.utils.data.DataLoader(train_dataset)
    
    return train_loader, train_dataset.path[:,0]
  
class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_dir, transform):
        self.transform = transform
        self.image_dir = image_dir
        self.input_dir = input_dir
        self.path      = self._load_path(self.input_dir)
    def _load_path(self, path_dir):
        path = pd.read_csv(path_dir)
        path = path.values
        return path
    def __getitem__(self, index):
        img_path = self.path[index][0]
        img = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = img / 255
        return img
    def __len__(self):
        return self.path.shape[0]