#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:34:51 2020

@author: lokeshkvn
"""

import piq

import torch
from torch import nn
import torchvision.models as models

from skimage.metrics import structural_similarity
from math import log10
from utils import denorm, tensor2numpy, RGB2BGR

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            #'3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
            '25': "relu5_1"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs
vgg16 = VGG(models.vgg16(pretrained=True).features[:26])
# vgg16 = vgg16.cuda()
epsilon = 1e-5
def mean_std(features):
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + epsilon)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1)  # 【mean, ..., std, ...] to [mean, std, ...]
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features
    
def PixelLoss():
    return torch.nn.L1Loss()

class ContentLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
class SSIM_Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.ssimLoss = piq.SSIMLoss()
        
    def forward(self,logits,targets):
        loss = self.ssimLoss(denorm(logits), denorm(targets)) 
        return loss
    
class StyleLoss(nn.Module):
    def __init__(
            self,
            eps: float = 1e-5,
        ):
            super().__init__()
            self.loss = torch.nn.MSELoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
            """Calculates loss between ``logits`` and ``target`` tensors.
    
            Args:
                logits: model logits
                targets: ground truth labels
    
            Returns:
                computed loss
            """
            # print(logits.shape,  targets.shape)
            fake_mean_std = mean_std(logits)
            real_mean_std = mean_std(targets)
            
            return self.loss(fake_mean_std, real_mean_std)
        
class PSNR(nn.Module):
    def __init__(
            self,
            eps: float = 1e-5,
        ):
            super().__init__()
            self.eps = eps
            
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        
        psnr_index = piq.psnr(denorm(targets), denorm(logits) , data_range=1.)
        return psnr_index
            
class SSIM(nn.Module):
    def __init__(
            self,
            eps: float = 1e-5,
        ):
            super().__init__()
            self.eps = eps
            
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # print(denorm(logits).max(), denorm(logits).min(), denorm(targets).max(), denorm(targets).min())
        ssim = piq.ssim(denorm(targets), denorm(logits) , data_range=1.)
        return ssim