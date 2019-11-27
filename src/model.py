# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:28:14 2019

@author: Austin Hsu
"""

import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class Train_Model(nn.Module):
    
    def __init__(self, pretrained=True, model_name='efficientnet-b5', num_classes=14):
        super(Train_Model, self).__init__()
        
        # --- efficientnet._global_params ---
        # batch_norm_momentum
        # batch_norm_epsilon
        # dropout_rate
        # num_classes ( should not use here )
        # width_coefficient
        # depth_coefficient
        # min_depth
        # drop_connect_rate
        # image_size
        
        # --- LOAD MODEL ---
        self.pretrained = pretrained
        self.model_name = model_name
        self.num_classes = num_classes
        if model_name[:12] == 'efficientnet':
            if int(model_name[-1]) > 7:
                raise ImportError('Only efficientnet-b0 to efficientnet-b7 models are implemented')
            if pretrained:
                self.conv_net = EfficientNet.from_pretrained(model_name)
            else:
                self.conv_net = EfficientNet.from_name(model_name)
            self.image_size = self.conv_net._global_params.image_size
            self.conv_net._fc = nn.Sequential(nn.Linear(self.conv_net._fc.in_features, num_classes),
                                              nn.Sigmoid())
        else:
            if model_name == 'inception_v3':
                raise ImportError('inception_v3 model is not implemented in current design')
            self.conv_net = getattr(models, model_name)(pretrained=pretrained)
            self.image_size = 224
            if model_name[:6] == 'resnet' or model_name[:7] == 'resnext' or model_name[:11] == 'wide_resnet':
                self.conv_net.fc = nn.Sequential(nn.Linear(self.conv_net.fc.in_features, num_classes),
                                                  nn.Sigmoid())
            else:
                self.conv_net.classifier = nn.Sequential(nn.Linear(self.conv_net.classifier.in_features, num_classes),
                                                  nn.Sigmoid())
            print('Model: ', model_name)
        
    def forward(self, x):
        x = self.conv_net(x)
        return x