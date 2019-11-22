# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:28:14 2019

@author: Austin Hsu
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNet_Model(nn.Module):
    
    def __init__(self, pretrained=True, model_name='efficientnet-b5'):
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
        self._avgpool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(self.conv_net._fc.in_features, 14)
        
    def forward(self, x):
        x = self.conv_net.extract_features(x)
        x = self._avgpool(x)
        x = x.view(x.size(0),-1)
        x = self._dropout(x)
        x = self._fc(x)
        return x