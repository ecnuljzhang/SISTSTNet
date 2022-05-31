#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:27:00 2020

@author: lokeshkvn
"""


import datetime
import itertools
import numpy as np
import os
import torchvision.transforms as transforms
from typing import Callable, List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import random
import glob



TRAIN_IMAGES_DIR = '.../LITST-Dataset/train/'
VALID_IMAGES_DIR = '.../LITST-Dataset/valid/'
PAIRS_IMAGES_DIR = '.../LITST-Dataset/pairs/'
STANDARD_IMAGES_DIR = '.../LITST-Dataset/std_font/'
MASKS_DIR = '.../LITST-Dataset/color_masks/'


IMAGE_FILE_EXT   = '.jpg'
IMAGE_READ_MODE  = 'L'


def get_data_multi(imgs_dir):
    
    font_styles = glob.glob(imgs_dir +"/*/")
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
        
    return perms_valid
    

def get_data(imgs_dir):
    
    font_styles = glob.glob(imgs_dir +"/*/")
    
    len_images = []
    
    set_en = [];
    set_en_dev = []
    
    dir_en = [];
    dir_en_dev = [];
    
    for fs in font_styles:
        images = [os.path.basename(x)[:-4] for x in glob.glob(fs + "/*.jpg")]
        len_images.append((len(images)))
        if len(images) == 126:
            dir_en_dev.append(fs.split('/')[-2])
            if len(set_en_dev) == 0: 
                set_en_dev = images
            else: 
                assert set(set_en_dev) & set(images)
        if len(images) == 62:
            dir_en.append(fs.split('/')[-2])
            if len(set_en) == 0: 
                set_en = images
            else: 
                assert set(set_en) & set(images)
                
    perms_en = list(itertools.product(set_en,
                                    set_en,
                                    dir_en))
    
    perms_en_dev = list(itertools.product(set_en_dev,
                                    set_en_dev,
                                    dir_en_dev, ))
    
    perms_train = perms_en + perms_en_dev
    
    return perms_train


class GeneratorDataset(Dataset):
    def __init__(
        self,
        image_dir, 
        std_font_dir,
        masks_dir,
        transform,
        image_ext='.jpg',
        mode='RGB', 
        target_shape=(64, 64), 
        rescale=1.0,    
    ) -> None:
        
        dir_lst = os.listdir(image_dir)
        if '.DS_Store' in dir_lst:
            dir_lst.remove('.DS_Store')
            
        self._perms = get_data_multi(image_dir)
        self._imdir = image_dir
        self._imext = image_ext
        self._std_dir = std_font_dir
        self._imtyp = mode
        self._shape = target_shape
        self._scale = rescale
        self.transforms = transform
        self.color_masks = glob.glob(masks_dir + "/*.jpg")
        self.target_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    def __len__(self) -> int:
        return len(self._perms)

    def __getitem__(self, idx: int) -> dict:
        ch_src = self._perms[idx][0]
        ch_dst = self._perms[idx][1]
        ch_fnt = self._perms[idx][2]
        # print(ch_src,ch_fnt, ch_dst)
        im_src = os.path.join(self._imdir, ch_fnt, ch_src + self._imext)
        im_dst = os.path.join(self._imdir, ch_fnt, ch_dst + self._imext)
        im_target = os.path.join(self._std_dir, ch_dst + self._imext)
        mask = random.choice(self.color_masks)
        img_m = Image.open(mask).resize(self._shape)
        img_m = np.asarray(img_m, dtype=np.uint8)
        # print(img_m.shape)
        img_x0 = Image.open(im_src).convert(self._imtyp).resize(self._shape)
        img_x0 = np.asarray(img_x0, dtype=np.uint8)
        img_x0 =  1 * (img_x0 > 127)
        img_x0 = np.repeat(img_x0[...,None],3,axis=2)
        img_x0 = np.asarray(np.multiply(img_x0, img_m), dtype=np.uint8)
        # print(img_x0.shape)
        img_y0 = Image.open(im_dst).convert(self._imtyp).resize(self._shape)
        img_y0 = np.asarray(img_y0, dtype=np.uint8)
        img_y0 = 1 * (img_y0 > 127)
        img_yb = img_y0.copy()
        img_yb = np.atleast_3d(img_yb)
        # print(img_y0.shape)
        img_y0 = np.repeat(img_y0[...,None],3,axis=2)
        img_y0 = np.asarray(np.multiply(img_y0, img_m), dtype=np.uint8)
        # print(img_y0.shape)
        img_tr = Image.open(im_target).convert(self._imtyp).resize(self._shape)
        img_tr = np.asarray(img_tr, dtype=np.uint8)
        # img_tr = 255 * (img_tr > 64)
        img_tr = np.atleast_3d(img_tr)

        if self.transforms:
            img_x0 = self.transforms((img_x0))
            img_y0 = self.transforms((img_y0))
            img_tr = self.target_transform(np.array(img_tr, dtype = np.float32))
            img_yb = self.target_transform(np.array(img_yb, dtype = np.float32))
        
        result = {"source": img_x0}
        result["target"] = img_tr
        result["output"] = img_y0
        result["output_binary"] = img_yb

        return result
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
#     ])  

# train_dataset = GeneratorDataset(
#         image_dir=VALID_IMAGES_DIR,
#         std_font_dir = STANDARD_IMAGES_DIR,
#         masks_dir = MASKS_DIR,
#         transform = transform,
#         image_ext=IMAGE_FILE_EXT,
#         mode=IMAGE_READ_MODE,
#         target_shape=(64,64),
#         rescale=1,
#     ) 

# z = get_data_multi(VALID_IMAGES_DIR)
