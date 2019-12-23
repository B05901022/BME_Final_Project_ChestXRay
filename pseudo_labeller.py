#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:53:34 2019

@author: austinhsu
"""

import torch
import warnings
import sys
from tqdm import tqdm
from src.dataset import UnlabeledImageDataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Executing on:', device)
headers='Path,Sex,Age,Frontal/Lateral,AP/PA,' + \
        'No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,' + \
        'Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,' + \
        'Pleural Effusion,Pleural Other,Fracture,Support Devices'

def main(output_pred_csv):
    
    warnings.filterwarnings("ignore")
                
    # --- MODEL / DATA_LOADER ---
    test_dir = 'model/base23.pkl'
    test_model = torch.load(test_dir, map_location=device).to(device)
    test_loader, test_path = UnlabeledImageDataLoader(transform_args=['Resize', 'ToTensor'],
                                                      randaug=False,
                                                      image_dir = '../ChestX-ray14/images/',
                                                      label_dir_list = ['../ChestX-ray14/train_val_list.txt', '../ChestX-ray14/test_list.txt'],
                                                      batchsize=6,
                                                      res=224,
                                                      train=False)
        
    # --- TEST ---
    test_model = test_model.eval()
    pred_list  = []
    for b_num, image in tqdm(enumerate(test_loader)):
        image = image.to(device)
        pred  = test_model(image)
        for one_row in pred.cpu().data.numpy():
            pred_list.append((one_row>=0.5).astype(int))
    
    # --- WRITE FILE ---
    with open(output_pred_csv, 'w') as f:
        print(headers, file=f)
        for index in range(len(pred_list)):
            print('ChestX-ray14/images/'+test_path[index], end=',', file=f)
            print('<UNK>,<UNK>,<UNK>,<UNK>', end=',', file=f)
            if index < len(pred_list)-1:
                print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*pred_list[index]), file=f)
            else:
                print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*pred_list[index]), end='',  file=f)
    return

if __name__ == '__main__':
    output_pred_csv = sys.argv[1]
    main(output_pred_csv)