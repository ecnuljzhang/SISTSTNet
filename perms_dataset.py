#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:52:08 2021

@author: lokesh
"""

import glob, os
import itertools
from typing import Callable, List, Tuple
import numpy as np

font_styles = glob.glob("/Volumes/Recursion/Work/ICCV Work/LITST-Dataset/train/*/")

len_images = []

for fs in font_styles:
    images = [os.path.basename(x)[:-4] for x in glob.glob(fs + "/*.jpg")]
    len_images.append((len(images)))
    
dir_ = {key: [] for key in map(str,np.unique(len_images))}
set_ = {key: [] for key in map(str,np.unique(len_images))}

for fs in font_styles:
    images = [os.path.basename(x)[:-4] for x in glob.glob(fs + "/*.jpg")]        
    dir_[str(len(images))].append(fs.split('/')[-2])
    
    if len(set_[str(len(images))]) == 0: 
        set_[str(len(images))] = images
    else: 
        assert set(set_[str(len(images))]) & set(images)
        
param_ = {key: [] for key in map(str,np.unique(len_images))}

for p in param_.keys():
    param_[p] = list(itertools.product(set_[p],
                                set_[p],
                                dir_[p]))
prems_train = []

for p in param_.keys():
    prems_train += param_[p]
    

font_styles = glob.glob("/Volumes/Recursion/Work/ICCV Work/LITST-Dataset/valid/*/")

len_images = []


for fs in font_styles:
    images = [os.path.basename(x)[:-4] for x in glob.glob(fs + "/*.jpg")]
    len_images.append((len(images)))
    
dir_ = {key: [] for key in map(str,np.unique(len_images))}
set_ = {key: [] for key in map(str,np.unique(len_images))}

for fs in font_styles:
    images = [os.path.basename(x)[:-4] for x in glob.glob(fs + "/*.jpg")]        
    dir_[str(len(images))].append(fs.split('/')[-2])
    
    if len(set_[str(len(images))]) == 0: 
        set_[str(len(images))] = images
    else: 
        assert set(set_[str(len(images))]) & set(images)
        
param_ = {key: [] for key in map(str,np.unique(len_images))}

for p in param_.keys():
    param_[p] = list(itertools.product(set_[p],
                                set_[p],
                                dir_[p]))
            

perms_valid = []

for p in param_.keys():
    perms_valid += param_[p]
    
# color_masks = [os.path.basename(x)[:-4] for x in glob.glob("/Volumes/Recursion/DL_Resource/STEFANN_CVPR/colornet/train/_color_filters/*.jpg")]
# import random

# picks = [];
# for i in range(6000):
#     p = random.choice(color_masks)
#     picks.append(p)
        
# picks.sort()
# u,c = np.unique(np.array(picks), return_counts=True)