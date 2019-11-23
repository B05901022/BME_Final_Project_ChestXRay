# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:28:14 2019

@author: Austin Hsu
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNet_Model(nn.Module):
    
    def __init__(self, pretrained=True, model_name='efficientnet-b5', num_classes=14):
        super(EfficientNet_Model, self).__init__()
        
        # --- LOAD MODEL ---
        if pretrained:
            self.conv_net = EfficientNet.from_pretrained(model_name)
        else:
            self.conv_net = EfficientNet.from_name(model_name)
        self._global_params = self.conv_net._global_params
        # batch_norm_momentum
        # batch_norm_epsilon
        # dropout_rate
        # num_classes ( should not use here )
        # width_coefficient
        # depth_coefficient
        # min_depth
        # drop_connect_rate
        # image_size
        
        # --- OUTPUT LAYER ---
        self.conv_net._fc = nn.Sequential(nn.Linear(self.conv_net._fc.in_features, num_classes),
                                          nn.Sigmoid())
        
    def forward(self, x):
        x = self.conv_net(x)
        return x