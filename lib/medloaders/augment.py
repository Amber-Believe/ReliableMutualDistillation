import numpy as np
import torch
import random
from lib.utils.torch_deform import deform_grid
import cv2 as cv

def gasuss_noise(img,mean=0, std=0.001):
    '''
        添加高斯噪声
        mean : 均值
        std : 标准差
    '''

    # image = np.array(image / 255, dtype=float)
    #noise = np.random.normal(mean, var ** 0.5, image1.shape)
    #noise = torch.Tensor(noise).cuda()
    noise = torch.normal(mean,std,size=img.size()).cuda()
    return torch.clamp(img + noise,0,1.0)


def flip(img,flipCode=None):
    '''
        img: input ,tensor
        flipCode: flip or not
    '''
    if flipCode is None:
        flipCode = random.choice([ 0, 2, 3, 4])
    if flipCode > 0:
        if flipCode==4:
            img = torch.flip(img, [2,3])
        else:
            img = torch.flip(img,[flipCode])
    return img, flipCode

def deform(img, displacements=None, rotates=None, zooms=None):
    n,ch,h,w=img.shape
    if displacements is None:
        displacements = []
    if rotates is None:
        rotates = []
    if zooms is None:
        zooms = []
    imgnew = []
    for i in range(n):
        imgtmp = img[i]
        if  len(displacements) < n :
            num = random.random() * 25 + 1
            displacement = np.random.randn(2,3,3) * num
            displacements.append(displacement)
        else:
            displacement = displacements[i]
        if len(rotates) < n :
            rotate = np.random.uniform(0, 60)
            rotates.append(rotate)
        else:
            rotate = rotates[i]
        if len(zooms) < n :
            zoom = np.random.uniform(1, 2)
            zooms.append(zoom)
        else:
            zoom = zooms[i]
        imgnewtmp = deform_grid(imgtmp, torch.Tensor(displacement), order=3, mode='nearest', rotate=rotate, zoom=zoom, axis=(1,2))
        imgnew.append(imgnewtmp)
    return torch.stack(imgnew,0), displacements, rotates, zooms 

