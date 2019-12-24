#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:53:34 2019

@author: austinhsu
"""

import torch
import numpy as np
import pandas as pd
import gc
import warnings
import sys
from tqdm import tqdm
from src.dataset import UnlabeledImageDataLoader, ImageDataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
headers='Path,Sex,Age,Frontal/Lateral,AP/PA,' + \
        'No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,' + \
        'Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,' + \
        'Pleural Effusion,Pleural Other,Fracture,Support Devices'

def get_threshold(model_dir, use_cached, search_space,
                  image_dir='../', label_dir='../CheXpert-v1.0-small/valid.csv',
                  batchsize=1, numworker=12):
    
    if use_cached:
        try:
            threshold_file = pd.read_csv('Threshold/'+model_dir.split('/')[-1][:-4]+'_threshold.csv').values
            return threshold_file[0], threshold_file[1]
        except:
            print('Cached file not found')
            
    print('Generating new threshold file...')
    
    model = torch.load(model_dir).to(device)
    test_loader, _ = ImageDataLoader(transform_args=['Resize', 'ToTensor'],
                                     pseudo_label_args={'use_pseudo_label':False},
                                     randaug=False,
                                     image_dir=image_dir,
                                     label_dir=label_dir,
                                     batchsize=batchsize,
                                     numworker=numworker,
                                     res=model.image_size,
                                     train=False,
                                     label_smooth=False)
    model = model.eval()
    pred_list  = []
    label_list = []
    for b_num, (image, label) in enumerate(test_loader):
        image = image.to(device)
        pred  = model(image)
        for one_row in pred.cpu().data.numpy():
            pred_list.append(one_row)
        for one_row in label.cpu().data.numpy():
            label_list.append(one_row)
    pred_list  = np.array(pred_list)
    label_list = np.array(label_list)
    
    thresholds = np.ones((14,))/2
    acc_all    = np.zeros((14,))
    for pathology in [8,2,6,5,10]:
        acc_list = []
        for thresh in search_space:
            threshold_list = (pred_list[:,pathology] > thresh).astype(int)
            acc = np.equal(threshold_list, label_list[:,pathology]).mean()
            acc_list.append(acc)
        thresholds[pathology] = search_space[np.argmax(acc_list)]
        acc_all[pathology]   = acc_list[np.argmax(acc_list)]
    
    thresholds = thresholds.tolist()
    acc_all   = acc_all.tolist()
    head = 'No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,' + \
           'Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,' + \
           'Pleural Effusion,Pleural Other,Fracture,Support Devices'
    with open('Threshold/'+model_dir.split('/')[-1][:-4]+'_threshold.csv', 'w') as f:
        print(head, file=f)
        print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*thresholds), file=f)
        print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*acc_all), file=f)
    
    del model, image
    torch.cuda.empty_cache()
    gc.collect()
        
    return np.array(thresholds), np.array(acc_all)

def main(output_pred_csv):
    
    warnings.filterwarnings("ignore")
    
    # --- THRESHOLD ---
    test_dir = 'model/base23.pkl'
    threshold, _ = get_threshold(model_dir=test_dir, use_cached=True, search_space=np.linspace(0,1,101))
    
    # --- MODEL / DATA_LOADER ---
    test_model = torch.load(test_dir, map_location=device).to(device)
    test_loader, test_path = UnlabeledImageDataLoader(transform_args=['Resize', 'ToTensor'],
                                                      randaug=False,
                                                      image_dir = '../ChestX-ray14/images/',
                                                      label_dir_list = ['../ChestX-ray14/train_val_list.txt', '../ChestX-ray14/test_list.txt'],
                                                      batchsize=6,
                                                      res=320,
                                                      train=False)
        
    # --- TEST ---
    test_model = test_model.eval()
    pred_list  = []
    for b_num, image in tqdm(enumerate(test_loader)):
        image = image.to(device)
        pred  = test_model(image)
        for one_row in pred.cpu().data.numpy():
            pred_list.append((one_row>=threshold).astype(int))
    
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