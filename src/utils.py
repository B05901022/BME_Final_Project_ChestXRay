#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:28:20 2019

@author: austinhsu
"""

import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

class AUROC():
    
    def __init__(self, num_classes, class_names, eval_metrics):
        self.num_classes = num_classes
        self.class_names = class_names
        self.eval_metrics = np.array(eval_metrics)
        
    def auroc(self, label, pred):
        """
        input:
            label: [np.array] (num_samples, num_classes)
            pred:  [np.array] (num_samples, num_classes)
        output:
            total_auroc: [np.array] (num_classes, )
            total_fpr_tpr: [list] list of (fpr, tpr) tuples
            total_auroc.mean(): mean for total_auroc
        """
        total_auroc = []
        total_fpr_tpr = []
        for detection in range(self.num_classes):
            fpr, tpr, threshold = metrics.roc_curve(label[:,detection], pred[:,detection])
            if len(np.unique(label[:,detection])) != 1:
                roc_auc = metrics.auc(fpr, tpr)
            else:
                if np.isnan(fpr[0]): fpr = np.zeros(fpr.shape)
                if np.isnan(tpr[0]): tpr = np.zeros(tpr.shape)
                roc_auc = metrics.accuracy_score(label[:,detection], np.rint(pred[:,detection]))
            total_auroc.append(roc_auc)
            total_fpr_tpr.append((fpr, tpr))
        total_auroc = np.array(total_auroc)
        eval_auroc  = total_auroc[self.eval_metrics]
        return total_auroc, total_fpr_tpr, eval_auroc.mean()
    
    def draw_curve(self, total_fpr_tpr):
        fig_list = []
        fig_pos = 0
        for detection in self.eval_metrics:
            fig = plt.figure()
            plt.title('ROC for: ' + self.class_names[detection])
            plt.plot(*total_fpr_tpr[detection])
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            fig_pos += 1
            fig_list.append((self.class_names[detection], fig))
        return fig_list

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, delay_epochs, after_scheduler):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return [base_lrs * (self.last_epoch / self.delay_epochs) for base_lrs in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(WarmupScheduler, self).step(epoch)

def WarmupLR(optimizer, delay_epochs, base_scheduler):
    return WarmupScheduler(optimizer, delay_epochs, base_scheduler)
