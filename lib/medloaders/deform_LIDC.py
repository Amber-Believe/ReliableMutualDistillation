import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
import os
import cv2 as cv
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
import random
from elasticdeform import deform_random_grid, deform_grid
# from deform import deform_random_grid
from sklearn import preprocessing

# from pylab import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location='/home/data/LIDC-IDRI/')

y = []
z = np.zeros(shape=(128, 128))
x = dataset.images

for i in range(len(x)):
    for j in range(4):
        if dataset.labels[i][j].max() == 1:
            p = j
    z = dataset.labels[i][p]
    y.append(z)
image_train, image_test, label_train, label_test = train_test_split(x, y, test_size=0.1, random_state=1)

image_meta = image_train[:800]
image_train = image_train[800:]
label_meta = label_train[:800]
label_train = label_train[800:]

image60_train, image40_train, label60_train, label40_train = train_test_split(image_train, label_train, test_size=0.6,
                                                                              random_state=1)
# image40_train = image_train
# label40_train = label_train
image_train = image40_train + image60_train
label_train = label40_train + label60_train

new_label40_train = []

for i in range(len(label40_train)):
    num = random.random() * 10 + 1
    if num == 0:
        new_label40_train.append(label40_train[i])
    else:

        z = deform_random_grid(label40_train[i], num, 5, order=1, mode='nearest', rotate=np.random.uniform(-45, 45),
                               zoom=np.random.uniform(0.8, 2))
        new_label40_train.append(z)

        # z1 = deform_random_grid(label40_train[i], num, 3, order=1, mode='nearest',  rotate=60, zoom=np.random.uniform(0.8,2))
        '''
        new_label40_train[i] = new_label40_train[i] + label40_train[i]
        #new_label40_train[i] = preprocessing.minmax_scale(new_label40_train[i], feature_range=(0, 1), axis=0, copy=True)
        for m in range(128):
            for n in range(128):
                if new_label40_train[i][m][n]==2:
                    new_label40_train[i][m][n]=1
                #if new_label40_train[i][m][n]==3:
                #    new_label40_train[i][m][n]=1
        '''
# cv.namedWindow('Image')
# cv.namedWindow('Image1')
# p1 = label40_train[0]
# p2 = new_label40_train[0]
# p1 *= 255
# p2 = (p1/2)*255
# cv.imshow("Image",p1)
# cv.imshow("Image1",p2)
# cv.waitKey(0)
# print(p1)
# cv.destroyAllWindow()
# x1=torch.Tensor(x)
# y1=torch.Tensor(y)


noisy_train100_image = image40_train + image60_train
noisy_train100_label = new_label40_train + label60_train

# noisy_train99_image, text1_image, noisy_train99_label, text1_label = train_test_split(noisy_train100_image, noisy_train100_label, test_size=1,random_state=1)
# noisy_train100_image = noisy_train99_image + text1_image
# noisy_train100_label = noisy_train99_label+ text1_label

# noisy_train100_image =  image40_train
# noisy_train100_label =  new_label40_train
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/train100_image.npy", torch.Tensor(image_train))
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/train100_label.npy", torch.Tensor(label_train))

np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/noisy_train100_image.npy",
        torch.Tensor(noisy_train100_image))
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/noisy_train100_label.npy",
        torch.Tensor(noisy_train100_label))

# np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data/4_new_data/noisy_train40_label.npy", torch.Tensor(new_label40_train))

np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/meta_image.npy", torch.Tensor(image_meta))
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/meta_label.npy", torch.Tensor(label_meta))
# np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data/7_new_data/train80_image.npy", torch.Tensor(image60_train))
# np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data/7_new_data/train80_label.npy", torch.Tensor(label60_train))
# np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data/4_new_data/train40_image.npy", torch.Tensor(image40_train))
# np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data/4_new_data/train40_label.npy", torch.Tensor(label40_train))
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/test_image.npy", torch.Tensor(image_test))
np.save("/home/qianwang/wq/MedZoo/datasets/LIDC_data_1/data_25/test_label.npy", torch.Tensor(label_test))

print('finish')