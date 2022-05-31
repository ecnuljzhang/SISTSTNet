#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:46:12 2020

@author: lokeshkvn
"""


import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from catalyst.core.callback import Callback, CallbackOrder

def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    
def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return (1 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch))   
    
class InferMaskCallback(Callback):
    def __init__(
        self,
        out_dir=None,
        out_prefix=None,
        max_images = 10,
        save = True,
        show = False,
    ):
        super().__init__(CallbackOrder.Internal)
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.counter = 0
        self.max_images = max_images
        self.show = show
        self.save = save
        self._keys_from_runner = ["out_dir", "out_prefix"]
        
    def on_stage_start(self, runner):
        
        for key in self._keys_from_runner:
            value = getattr(runner, key, None)
            if value is not None:
                setattr(self, key, value)
        self.out_prefix = (
            self.out_prefix if self.out_prefix is not None else ""
        )
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, runner):
        """Loader start hook.
        Args:
            runner: current runner
        """
        lm = runner.loader_name
        os.makedirs(f"{self.out_prefix}", exist_ok=True)
    
    def on_epoch_end(self, runner):
        """Batch end hook.
        Args:
            runner: current runner
        """
        lm = runner.loader_name
        current_batch = next(iter(runner.loaders["valid"]))
        prediction = runner.predict_batch(current_batch)
        input_images = (current_batch['source']).cpu()
        output_images = (current_batch['output']).cpu()
        predicted_images = (prediction['generated']).cpu()
        predicted_binary_images = (prediction['generated']).cpu().sigmoid()
        # predicted_binary_images = torch.argmax(predicted_binary_images, dim=1)
        predicted_binary_images = predicted_binary_images.repeat(1, 3, 1, 1)
        # print(predicted_binary_images.shape, predicted_images.shape, output_images.shape, input_images.shape)
        combined_images = []
        for i in range(0,self.max_images):
            combined = torch.cat([denorm(input_images[i]), denorm(output_images[i]), denorm(predicted_images[i])], 1)
            combined_images.append(combined)
        predictions = torch.cat(combined_images, 2)
        # print(predictions.shape)
        im_prd = transforms.ToPILImage()(predictions)
        if self.show:
            plt.figure(figsize=(im_prd.width/100, im_prd.height/100), dpi=100)
            plt.axis('off')
            plt.imshow(im_prd)
            plt.show()
        
        if self.save: 
            im_prd.save(f"{self.out_prefix}/epoch_{runner.epoch}.jpg")