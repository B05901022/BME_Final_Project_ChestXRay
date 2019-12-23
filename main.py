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
import gc
import warnings

from src.model import Train_Model
from src.dataset import ImageDataLoader
from src.utils import AUROC, AUPRC, ACC_SCORE, WarmupLR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def main(config_dir):
    
    warnings.filterwarnings("ignore")
    
    # --- CONFIG ---
    config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    
    # --- LOG ---
    log_dir = os.path.join(config['log']['log_dir'], config['log']['config_name'])
    logger  = SummaryWriter(logdir=log_dir)
    train_step = 0
        
    # --- MODEL / OPTIMIZER / DATA_LOADER ---
    train_model = Train_Model(**config['model']).to(device)
    train_loader, train_num = ImageDataLoader(**config['train_loader'], res=train_model.image_size, train=True)
    train_optim = getattr(torch.optim, config['optimizer']['optimizer_type'])
    train_optim = train_optim(train_model.parameters(), **config['optimizer']['optimizer_param'])
    train_scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['scheduler_type'])
    train_scheduler = train_scheduler(optimizer=train_optim,
                                      T_max=(train_num//config['train_loader']['batchsize'])*config['log']['epoch'])
    if config['scheduler']['warmup']['use_warmup']:
        train_scheduler = WarmupLR(train_optim, config['scheduler']['warmup']['warmup_step'], train_scheduler)
    test_loader, _ = ImageDataLoader(**config['test_loader'], res=train_model.image_size, train=False)
    
    # --- CRITERION ---
    loss_weight = torch.Tensor(config['criterion']['weight'])
    train_criterion = nn.BCELoss(weight=loss_weight).to(device)
    auroc_evaluator = AUROC(**config['label_info'])
    auprc_evaluator = AUPRC(**config['label_info'])
    acc_evaluator   = ACC_SCORE(**config['label_info'])
    best_auroc = 0
    train_batch = len(train_loader)
    valid_batch = len(test_loader)
    
    # --- TRAIN / VALID ---
    for e in range(config['log']['epoch']):
        print('Epoch %2d'%e)

        epoch_loss  = 0
        epoch_acc   = 0
        for b_num, (image, label) in enumerate(train_loader):
            
            # --- TRAIN ---
            train_model = train_model.train()
            train_step += 1
            image = image.to(device)
            label = label.to(device, non_blocking=True)
            train_optim.zero_grad()
            pred = train_model(image)
            loss = train_criterion(pred, label)
            loss.backward()
            train_optim.step()
            train_scheduler.step()
            batch_acc   = torch.sum(torch.eq((pred>=0.5), (label>0.5))).item() / (config['label_info']['num_classes']*config['train_loader']['batchsize'])
            epoch_loss += loss.item()
            epoch_acc  += batch_acc
            logger.add_scalars('loss', {'train_loss': loss.item()}, train_step)
            logger.add_scalars('acc', {'train_acc': batch_acc}, train_step)
            print('[TRAIN] Batch: %5d/%5d | Loss: %.8f | Accuracy: %.4f'%(b_num+1, train_batch, loss.item(), batch_acc), end='\r')
            
            if train_step % config['log']['record_step'] == 0:
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
                    batch_acc   = torch.sum(torch.eq((pred>=0.5), (label>0.5))).item() / (config['label_info']['num_classes']*config['test_loader']['batchsize'])
                    valid_loss += loss.item()
                    valid_acc  += batch_acc
                    for one_row in pred.cpu().data.numpy():
                        pred_list.append(one_row)
                    for one_row in label.cpu().data.numpy():
                        label_list.append(one_row)
                    print('[VALID] Batch: %5d/%5d | Step: %6d | Loss: %.8f | Accuracy: %.4f'%(b_num+1, valid_batch, train_step, loss.item(), batch_acc), end='\r')
                
                # --- AUROC ---
                auroc_list, fpr_tpr_list, auroc = auroc_evaluator.auroc(np.array(label_list), np.array(pred_list))
                roc_fig_list = auroc_evaluator.draw_curve(fpr_tpr_list)
                
                # --- AUPRC ---
                auprc_list, pr_list, auprc = auprc_evaluator.auprc(np.array(label_list), np.array(pred_list))
                prc_fig_list = auprc_evaluator.draw_curve(pr_list)
                
                # --- ACC ---
                acc_score_list = acc_evaluator.acc(np.array(label_list), np.array(pred_list)) 
                
                print('[VALID] Batch: %5d/%5d | Step: %6d | Valid Loss: %.8f | Valid Accuracy: %.4f | AUROC: %.6f | AUPRC: %.6f'\
                      %(b_num+1, valid_batch, train_step, valid_loss/valid_batch, valid_acc/valid_batch, auroc, auprc))
                
                # --- LOGGING ---
                logger.add_scalars('loss', {'valid_loss': valid_loss/valid_batch}, train_step)
                logger.add_scalars('acc', {'valid_acc': valid_acc/valid_batch}, train_step)
                logger.add_scalars('auroc', {'valid_auroc': auroc}, train_step)
                logger.add_scalars('auroc', {config['label_info']['class_names'][detection]: auroc_list[detection] \
                                             for detection in range(config['label_info']['num_classes'])}, train_step)
                logger.add_scalars('auprc', {config['label_info']['class_names'][detection]: auprc_list[detection] \
                                             for detection in range(config['label_info']['num_classes'])}, train_step)
                logger.add_scalars('acc', {config['label_info']['class_names'][detection]: acc_score_list[detection] \
                                             for detection in range(config['label_info']['num_classes'])}, train_step)
                for roc_fig in roc_fig_list:
                    logger.add_figure('auroc/'+roc_fig[0], roc_fig[1], train_step)
                for prc_fig in prc_fig_list:
                    logger.add_figure('auprc/'+prc_fig[0], prc_fig[1], train_step)
        
                # --- UPDATE ---
                if auroc > best_auroc:
                    print('Update best model.')
                    torch.save(train_model, os.path.join(config['log']['model_dir'], config['log']['config_name']+'.pkl'))
                    torch.save(train_optim.state_dict(), os.path.join(config['log']['model_dir'], config['log']['config_name']+'.optim'))
                    best_auroc = auroc
                    for roc_fig in roc_fig_list:
                        logger.add_figure('best_auroc/'+roc_fig[0], roc_fig[1], train_step)
                
            # --- CLEAR MEMORY ---
            del loss, pred, label, image
            torch.cuda.empty_cache()
            gc.collect()
        
        print('')
        print('Epoch: %2d | Train Loss: %.8f | Train Accuracy: %.4f'%(e, epoch_loss/train_batch, epoch_acc/train_batch))

        
        
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