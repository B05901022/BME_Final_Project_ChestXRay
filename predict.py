#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:53:34 2019

@author: austinhsu
"""

import torch
import os
import warnings
import sys
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Executing on:', device)

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

def main(input_data_csv, output_pred_csv):
    
    warnings.filterwarnings("ignore")
                
    # --- MODEL / DATA_LOADER ---
    test_dir = 'model/base23.pkl'
    test_model = torch.load(test_dir, map_location=device).to(device)
    test_loader, test_path = TestImageDataLoader(image_dir='../', input_dir=input_data_csv, res=test_model.image_size)
        
    # --- TEST ---
    test_model = test_model.eval()
    pred_list  = []
    for b_num, image in enumerate(test_loader):
        image = image.to(device)
        pred  = test_model(image)
        for one_row in pred.cpu().data.numpy():
            pred_list.append((one_row>=0.5).astype(int))
    
    # --- only for study ---
    new_test_path = []
    new_pred_list = []
    for i in range(len(test_path)):
        study = '{}/{}/{}/{}'.format(*test_path[i].split('/')[:-1])
        if study in new_test_path:
            if 'frontal' in test_path[i].split('/')[-1]:
                new_pred_list.pop()
                new_pred_list.append(pred_list[i])
        else:
            new_test_path.append(study)
            new_pred_list.append(pred_list[i])
    test_path = new_test_path
    pred_list  = new_pred_list
    
    # --- WRITE FILE ---
    with open(output_pred_csv, 'w') as f:
        print('Study,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion', file=f)
        for index in range(len(pred_list)):
            print(test_path[index], end=',', file=f)
            print(pred_list[index][8], end=',', file=f)
            print(pred_list[index][2], end=',', file=f)
            print(pred_list[index][6], end=',', file=f)
            print(pred_list[index][5], end=',', file=f)
            if index < len(pred_list)-1:
                print(pred_list[index][10],          file=f)
            else:
                print(pred_list[index][10], end=',', file=f)
    
    return

if __name__ == '__main__':
    input_data_csv  = sys.argv[1]
    output_pred_csv = sys.argv[2]
    main(input_data_csv, output_pred_csv)