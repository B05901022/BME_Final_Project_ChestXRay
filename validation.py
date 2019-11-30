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

from src.dataset import TestImageDataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(input_data_csv, output_pred_csv):
    
    warnings.filterwarnings("ignore")
                
    # --- MODEL / DATA_LOADER ---
    test_dir = os.path.join('./model/', 'base7.pkl')
    test_model = torch.load(test_dir).to(device)
    test_loader, test_path = TestImageDataLoader(image_dir='../', input_dir=input_data_csv, res=test_model.image_size)
        
    # --- TEST ---
    test_model = test_model.eval()
    pred_list  = []
    for b_num, image in enumerate(test_loader):
        image = image.to(device)
        pred  = test_model(image)
        for one_row in pred.cpu().data.numpy():
            pred_list.append((one_row>=0.5).astype(int))
    
    # --- WRITE FILE ---
    with open(output_pred_csv, 'w') as f:
        print('Study,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion', file=f)
        for index in range(len(pred_list)):
            print(test_path[index], end=',', file=f)
            print(pred_list[index][8], end=',', file=f)
            print(pred_list[index][2], end=',', file=f)
            print(pred_list[index][6], end=',', file=f)
            print(pred_list[index][5], end=',', file=f)
            print(pred_list[index][10],         file=f)
    
    return

if __name__ == '__main__':
    input_data_csv  = sys.argv[1]
    output_pred_csv = sys.argv[2]
    main(input_data_csv, output_pred_csv)