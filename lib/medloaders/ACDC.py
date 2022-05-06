#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import scipy.ndimage
from elasticdeform import _deform_grid
import numpy as np
import cv2 as cv
import os
import glob
from torch.utils.data import Dataset
from torch.nn.functional import grid_sample
import random
from lib.utils.torch_deform import deform_grid
from matplotlib import pyplot as plt




#from elasticdeform import deform_random_grid
class ACDC_loader(Dataset):
    def __init__(self, mode, data_path):
        self.mode = mode
        self.data_path = data_path

        if self.data_path != None:
            if (self.mode == 'train'):
                self.imgs_path = data_path + "noisy_train100_image.npy"
                self.label_path = data_path + "noisy_train100_label.npy"

            if (self.mode == 'val'):
                self.imgs_path = data_path + "val_image.npy"
                self.label_path = data_path + "val_label.npy"


            self.image = np.load(self.imgs_path)
            self.label = np.load(self.label_path)



    def augment(self, image, flipCode):
        flip = cv.flip(image, flipCode)
        return flip

    def __getitem__(self, index):

        image = self.image[index]
        label = self.label[index]


        if self.mode == 'train':
            flipCode = random.choice([-1, 0, 1, 2])
            if flipCode != 2:
                image = cv.flip(image, flipCode)
                label = cv.flip(label, flipCode)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        image = image.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.FloatTensor)

        return image, label


    def __len__(self):
        return len(self.image)

if __name__ == "__main__":

    Train_LIDC_dataset = ACDC_loader(mode='train',data_path="/home/qwang/Desktop/code/datasets/ACDC/")
    Test_LIDC_dataset = ACDC_loader(mode='test',data_path="/home/qwang/Desktop/code/datasets/ACDC/")
    print("the number of train data : ",len(Train_LIDC_dataset))
    print("the number of test data : ", len(Test_LIDC_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=Train_LIDC_dataset,
                                               batch_size=5)
    test_loader = torch.utils.data.DataLoader(dataset=Test_LIDC_dataset,
                                               batch_size=5)
    for image, label in train_loader:
        print(image.shape)


