# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:08:37 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
from PIL import Image
import gc
import os
import numpy as np
import pandas as pd
import argparse
import warnings
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import NoiseTunnel
from captum.attr import DeepLift
from captum.attr import visualization as viz

from pseudo_labeller import get_threshold

def model_transform(model):
    def backward_hook(module, grad_input, grad_output):
        return grad_input
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.saved_grad = m.register_backward_hook(backward_hook)
    return model

def main(args):
    
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Executing on device:', device)
    
    # --- Get Threshold ---
    model = './model/base'+str(args.model_index)+'.pkl'
    threshold, _ = get_threshold(model_dir=model, use_cached=args.use_cached, search_space=np.linspace(0,1,args.search_num))
    
    # --- Load Model ---
    model = torch.load(model, map_location=device).to(device)
    model = model.eval()
    
    # --- Fix Seed ---
    torch.manual_seed(123)
    np.random.seed(123)
    
    # --- Label ---
    label_dir = '../CheXpert-v1.0-small/valid.csv'
    label = pd.read_csv(label_dir)
    target_label = np.array(list(label.keys())[5:])
    target_obsrv = np.array([8,2,6,5,10])
    label = label.values
    label_gd = label[:,5:]
    
    # --- Image ---
    img_index = args.image_index
    img_dir   = '../'
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    ])
    print('Patient:',label[img_index][0])
    img = Image.open(os.path.join(img_dir,label[img_index][0])).convert('RGB')
    img = transform(img)
    img_transformed = img / 255
    img_transformed = img_transformed.unsqueeze(0).to(device)
    
    # --- Predict ---
    pred = model(img_transformed)
    print()
    print('[Prediction]')
    print('{:17s}|{:4s}|{:4s}|{:4s}|{:3s}'.format('Pathology','Prob','Thrs','Pred','Ans'))
    for lbl, prd, thrsh, gd in zip(target_label[target_obsrv], pred[0][target_obsrv], threshold[target_obsrv], label_gd[img_index][target_obsrv]):
        print('{:17s}:{:4.2f} {:4.2f} {:4d} {:3d}'.format(lbl, prd.item(), thrsh, int(prd.item()>thrsh), int(gd)))
    
    print()
    del pred
    torch.cuda.empty_cache()
    gc.collect()
    model = model.to(torch.device('cpu'))
    img_transformed = img_transformed.to(torch.device('cpu'))
    
    # --- Visualization ---
    pathology_input = input('Please enter which pathology to visualize:\n[0]Atelectasis\n[1]Cardiomegaly\n[2]Consolidation\n[3]Edema\n[4]Pleural Effusion\n[5]Exit\n')
    if pathology_input == '0':
        pathology = 8
        print('Diagnosis on Atelectasis')
    elif pathology_input == '1':
        pathology = 2
        print('Diagnosis on Cardiomegaly')
    elif pathology_input == '2':
        pathology = 6
        print('Diagnosis on Consolidation')
    elif pathology_input == '3':
        pathology = 5
        print('Diagnosis on Edema')
    elif pathology_input == '4':
        pathology = 10
        print('Diagnosis on Pleural Effusion')
    elif pathology_input == '5':
        print('Exiting...')
        return
    else:
        raise NotImplementedError('Only 0-5 are valid input values')
        
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')],
                                                       N=256)
    print()
    method_input = input('Please enter which method to visualize:\n[0]GradientShap\n[1]DeepLift\n[2]Exit\n')
    if method_input == '0':
        print('Using GradientShap')
        # --- Gradient Shap ---
        gradient_shap = GradientShap(model)
        
        # === baseline distribution ===
        rand_img_dist = torch.cat([img_transformed*0, img_transformed*1])
        
        attributions_gs = gradient_shap.attribute(img_transformed,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pathology)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=default_cmap,
                                              show_colorbar=True)
        del attributions_gs
    elif method_input == '1':
        print('Using DeepLIFT')
        # --- Deep Lift ---
        model = model_transform(model)
        dl = DeepLift(model)
        attr_dl = dl.attribute(img_transformed, target=pathology, baselines=img_transformed * 0)
        _ = viz.visualize_image_attr_multiple(np.transpose(attr_dl.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap=default_cmap,
                                              show_colorbar=True)
        del attr_dl
    elif method_input == '2':
        print('Exiting...')
        return
    else:
        raise NotImplementedError('Only 0-2 are valid input values')
        
    """
    elif method_input == '2':
        print('Using Integrated Gradients')
        # --- Integrated Gradients ---
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(img_transformed, target=pathology, n_steps=200)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              method=["original_image", "heat_map"],
                                              cmap=default_cmap,
                                              show_colorbar=True,
                                              sign=["all", "positive"])
        del attributions_ig
    elif method_input == '3':
        print('Using Noise Tunnel')
        # --- Noise Tunnel ---
        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig_nt = noise_tunnel.attribute(img_transformed, n_samples=10, nt_type='smoothgrad_sq', target=pathology)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap=default_cmap,
                                              show_colorbar=True)
        del attributions_ig_nt
    """
    
    gc.collect()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BME_Final')
    parser.add_argument('--image_index', '-i', help='Which patient from the validation set.', type=int, default=93)
    parser.add_argument('--model_index', '-m', help='Which model to be visualized.(using config#)', type=int, default=23)
    parser.add_argument('--use_cached',  '-u', help='(optional) If using the cached prediction threshold in ./Threshold/', type=bool, default=True)
    parser.add_argument('--search_num',  '-s', help='(optional) How many numbers between [0,1] to be searched for threshold. \'-u\' must be False for working', type=int, default=101)                                                                                                                                                              
    args = parser.parse_args()
    main(args)