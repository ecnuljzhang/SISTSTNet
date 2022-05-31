#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:40:45 2020

@author: lokeshkvn
"""

from utils_tet_dataset import load_trainset_batchfnames, load_trainset_fnames, load_validset_fnames, prepare_batch, load_valid_batchfnames
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import cv2


class GeneratorTETDataset(Dataset):
    def __init__(
        self,
        train_path, 
        datasize = 128000,
        datarange = 708,
        centercropratio = 0.5,
        augementratio =0.25,
        outer_iter = 50,
        dataset_type = "train"
    ) -> None:
        
        
        self._train_path = train_path
        self._datasize= datasize
        self._datarange = datarange
        if dataset_type == "train":
            # load_trainset_fnames(filepath, batch_size, usetrainnum=708, trainnum=100000)
            self._file_names = load_trainset_fnames(self._train_path, 1, self._datarange, self._datasize*2)
        elif dataset_type == "valid":
            self._file_names = load_validset_fnames(self._train_path, 1, self._datarange, self._datasize*2)
        else: 
            assert dataset_type == "train" or dataset_type == "valid"
        self._centercropratio = centercropratio
        self._augementratio = augementratio
        self._outer_iter = outer_iter
        self._binary_output_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5],std=[0.5])
        ])
    def __len__(self) -> int:
        return len(self._file_names)

    def __getitem__(self, idx: int) -> dict:
        fade_iter = max(1.0, float(self._outer_iter  / 2))  # fade = 25.0
        jitter = min(1.0, idx / fade_iter)  # jitter = 1/0
        target, output, source = prepare_batch(self._file_names[idx], 2, jitter, self._centercropratio, self._augementratio, 1)
        r = random.randrange(2)
        img_yb = cv2.cvtColor((((output[0][r,:,:,:]).cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # img_yb = 1 * (img_yb > 127)
        img_yb = self._binary_output_transform(np.array( np.atleast_3d(img_yb), dtype = np.float32))
        result = {"source": source[0][r, :, :, :]}
        result["target"] = target[0][r, :, :, :]
        result["output"] = output[0][r, :, :, :]
        result["output_binary"] = img_yb
        return result

# import matplotlib.pyplot as plt
# train_dataset = GeneratorTETDataset(
#         train_path = "/home/lokesh/research/CVPR/TET-E/E", dataset_type = "train",
#     ) 
# z = train_dataset[12796]

# fname = train_dataset._file_names