import numpy as np
import torch
import random
from elasticdeform import deform_random_grid
from sklearn.model_selection import train_test_split
import cv2 as cv
"""deform ACDC/BraTS"""

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    ''' 
    # 有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)  # 限定范围numpy.clip(a, a_min, a_max, out=None)

    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9  # 黑色背景区域
    '''

    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]

    tmp = (slice - np.min(image_nonzero)) / (np.max(image_nonzero) - np.min(image_nonzero))
    # since the range of intensities is between 0 and 5000 ,
    # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
    # the min is replaced with -9 just to keep track of 0 intensities
    # so that we can discard those intensities afterwards when sampling random patches

    tmp[tmp == tmp.min()] = 0  # 黑色背景区域
    return tmp




import math
from lib.utils.torch_deform import deform_grid

image_train = np.load('/home/data/train_image.npy')
label_train = np.load('/home/data/train_gt.npy')
label_train = np.array(label_train, dtype='uint8')
image60_train, image40_train, label60_train, label40_train = train_test_split(image_train, label_train, test_size=0.6,random_state=1)
# image40_train = image_train
# label40_train = label_train

new_label40_train = np.zeros(label40_train.shape)
for i in range(len(label40_train)):
    num = random.random() * 15 + 1
    if num == 0:
        new_label40_train[i] = label40_train[i]
    else:

        displacement = np.random.randn(2, 5, 5) * num
        displacement = torch.Tensor(displacement)
        rotate = np.random.uniform(-45, 45)
        zoom = np.random.uniform(0.8, 2)
        new_label40_train[i] = np.ceil(
            deform_grid(torch.Tensor(label40_train[i]), displacement, order=0, mode='nearest',rotate=rotate, zoom=zoom))


noisy_train100_image = np.vstack((image60_train, image40_train))
noisy_train100_label = np.vstack((label60_train, new_label40_train))

noisy_train99_image, text1_image, noisy_train99_label, text1_label = train_test_split(noisy_train100_image,
                                                                                      noisy_train100_label, test_size=1,
                                                                                      random_state=1)
noisy_train100_image = np.vstack((noisy_train99_image, text1_image))
noisy_train100_label = np.vstack((noisy_train99_label, text1_label))

np.save("/home/data/noisy_train100_image.npy", torch.Tensor(noisy_train100_image))
np.save("/home/data/noisy_train100_label.npy", torch.Tensor(noisy_train100_label))
print("finish")



