# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:48:24 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import yaml
import os
import numpy as np
import argparse

from src.model import EfficientNet_Model
from src.dataset import ImageDataLoader
from src.utils import AUROC

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def main(config_dir):
    
    # --- CONFIG ---
    config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    
    # --- LOG ---
    log_dir = os.path.join(config['log']['log_dir'], config['log']['config_name'])
    logger  = SummaryWriter(logdir=log_dir)
    train_step = 0
        
    # --- MODEL / OPTIMIZER / DATA_LOADER ---
    train_model = EfficientNet_Model(**config['model']).to(device)
    train_loader, train_num = ImageDataLoader(transform_args=config['train_param']['train_transform'],
                                              **config['train_loader'],
                                              batchsize=config['train_param']['batchsize'],
                                              numworker=config['train_param']['numworker'],
                                              res=train_model._global_params.image_size)
    train_optim = torch.optim.Adam(train_model.parameters(), **config['optimizer'])
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=train_optim,
                                                                 T_max=(train_num//config['train_param']['batchsize'])*config['train_param']['epoch'])
    test_loader, _ = ImageDataLoader(transform_args=[],
                                     **config['test_loader'],
                                     res=train_model._global_params.image_size)
    
    # --- CRITERION ---
    train_criterion = nn.BCELoss().to(device)
    auroc_evaluator = AUROC(**config['label_info'])
    best_auroc = 0
    train_batch = len(train_loader)
    valid_batch = len(test_loader)
    
    # --- TRAIN / VALID ---
    for e in range(config['train_param']['epoch']):
        print('Epoch %2d'%e)

        # --- TRAIN ---
        train_model = train_model.train()
        epoch_loss  = 0
        epoch_acc   = 0
        for b_num, (image, label) in enumerate(train_loader):
            train_step += 1
            image = image.to(device)
            label = label.to(device, non_blocking=True)
            train_optim.zero_grad()
            pred = train_model(image)
            loss = train_criterion(pred, label)
            loss.backward()
            train_optim.step()
            train_scheduler.step()
            batch_acc   = torch.sum(torch.eq((pred>=0.5), label.byte())).item() / (config['label_info']['num_classes']*config['train_param']['batchsize'])
            epoch_loss += loss.item()
            epoch_acc  += batch_acc
            logger.add_scalars('loss', {'train_loss': loss.item()}, train_step)
            logger.add_scalars('acc', {'train_acc': batch_acc}, train_step)
            print('[TRAIN] Batch: %4d/%4d | Loss: %.8f | Accuracy: %.4f'%(b_num, train_batch, loss.item(), batch_acc), end='\r')
        print('')

        # --- VALID ---
        train_model = train_model.eval()
        valid_loss = 0
        valid_acc  = 0
        pred_list  = []
        label_list = []
        for b_num, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device, non_blocking=True)
            pred  = train_model(image)
            loss  = train_criterion(pred, label)
            batch_acc   = torch.sum(torch.eq((pred>=0.5), label.byte())).item() / (config['label_info']['num_classes']*config['train_param']['batchsize'])
            valid_loss += loss.item()
            valid_acc  += batch_acc
            for one_row in pred.cpu().data.numpy():
                pred_list.append(one_row)
            for one_row in label.cpu().data.numpy():
                label_list.append(one_row)
            print('[VALID] Batch: %4d/%4d | Loss: %.8f | Accuracy: %.4f'%(b_num, valid_batch, loss.item(), batch_acc), end='\r')
        print('')
        auroc_list, fpr_tpr_list, auroc = auroc_evaluator.auroc(np.array(label_list), np.array(pred_list))
        roc_fig = auroc_evaluator.draw_curve(fpr_tpr_list)
        logger.add_scalars('loss', {'valid_loss': valid_loss/valid_batch}, train_step)
        logger.add_scalars('acc', {'valid_acc': valid_acc/valid_batch}, train_step)
        logger.add_scalars('auroc', {'valid_auroc': auroc}, train_step)
        logger.add_scalars('auroc', {config['label_info']['class_names'][detection]: auroc_list[detection] \
                                     for detection in range(config['label_info']['num_classes'])}, train_step)
        logger.add_figure('auroc', roc_fig, train_step)

        # --- UPDATE ---
        if auroc > best_auroc:
            print('Update best model.')
            torch.save(train_model, os.path.join(config['train_param']['model_dir'], config['log']['config_name']+'.pkl'))
            torch.save(train_optim.state_dict(), os.path.join(config['log']['model_dir'], config['log']['config_name']+'.optim'))
            best_auroc = auroc
        print('Epoch: %2d | Train Loss: %.8f | Train Accuracy: %.4f | Valid Loss: %.8f | Valid Accuracy: %.4f | AUROC: %.6f'\
              %(e, epoch_loss/train_batch, epoch_acc/train_batch, valid_loss/valid_batch, valid_acc/valid_batch, auroc))
    
    # --- DRAW MODEL GRAPH ---
    #logger.add_graph(train_model, (torch.zeros((1,3,456,456)).to(device), ))
    
    # --- CLOSE LOGGER ---
    logger.close()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BME_Final')
    parser.add_argument('--config_dir', '-c', type=str, default='./config/base_config.yaml')
    args = parser.parse_args()
    main(config_dir=args.config_dir)