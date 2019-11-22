# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:48:24 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import yaml

from model import EfficientNet_Model
from dataset import ImageDataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(config_dir):
    
    # --- CONFIG ---
    config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    
    # --- MODEL / OPTIMIZER / DATA_LOADER ---
    train_model = EfficientNet_Model(**config['model'])
    train_loader, train_num = ImageDataLoader(**config['train_loader'], res=train_model._global_params.image_size)
    train_optim = torch.optim.AdamW(train_model.parameters(), **config['optimizer'])
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=train_optim,
                                                                 T_max=int(config['train_param']['epoch']/config['train_param']['batchsize'])*train_num)
    test_loader, _ = ImageDataLoader(**config['test_loader'], res=train_model._global_params.image_size)
    
    # --- CRITERION ---
    train_criterion = nn.BCELoss()
    
    # --- TRAIN / VALID ---
    for e in range(config['train_param']['epoch']):
        print('Epoch ', e)
        
        # TODO: 
        # 1. Extra transform
        # 2. AUCROC validation
        
        # --- TRAIN ---
        train_model = train_model.train()
        epoch_loss  = 0
        epoch_acc   = 0
        for b_num, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            train_optim.zero_grad()
            pred = train_model(image)
            loss = train_criterion(pred, label)
            loss.backward()
            train_optim.step()
            train_scheduler.step()
            batch_acc   = torch.sum(torch.eq((pred>=0.5), label.byte())).item()/14
            epoch_loss += loss.item()
            epoch_acc  += batch_acc
            print('[TRAIN] Batch: %4d | Loss: %.8f | Accuracy: %.4f'%(b_num, loss.item(), batch_acc), end='\r')
        
        # --- VALID ---
        train_model = train_model.eval()
        valid_loss = 0
        valid_acc  = 0
        for b_num, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            pred  = train_model(image)
            loss  = train_criterion(pred, label)
            batch_acc   = torch.sum(torch.eq((pred>=0.5), label.byte())).item()/14
            valid_loss += loss.item()
            valid_acc  += batch_acc
            print('[VALID] Batch: %4d | Loss: %.8f | Accuracy: %.4f'%(b_num, loss.item(), batch_acc), end='\r')
        
    return

if __name__ == '__main__':
    main(config_dir='../config/base_config.yaml')