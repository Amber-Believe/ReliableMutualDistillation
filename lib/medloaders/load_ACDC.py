import os
import cv2 as cv
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'test':
            with open(self._base_dir + '/test.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        new_size = 128
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/{}.h5".format(case), 'r')
            #h5f = h5py.File(self._base_dir +
             #               "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        n, h, w = np.shape(label)
        gt=[]
        img=[]
        for i in range (n):
            gt_slice = np.asarray(label[i], dtype=np.float32)
            left = int(np.ceil((w - new_size) / 2))
            right = w - int(np.floor((w - new_size) / 2))
            top = int(np.ceil((h - new_size) / 2))
            bottom = h - int(np.floor((h - new_size) / 2))
            gt_slice = gt_slice[top:bottom, left:right]
            gt.append(gt_slice)

            image_slice = np.asarray(image[i], dtype=np.float32)
            image_slice = image_slice[top:bottom, left:right]
            img.append(image_slice)
        return np.array(img),np.array(gt)






        #sample = {'image': img, 'label': gt}
        #if self.split == "train":
        #    sample = self.transform(sample)
        #sample["idx"] = idx
        #return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        # prinmary and secondary are 12 batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        # for (primary_batch, secondary_batch)in zip(grouper(primary_iter, self.primary_batch_size),
        #                                             grouper(secondary_iter, self.secondary_batch_size)):
        #     print('prinmary_batch',next(prinmary_batch))
        # return(primary_batch + secondary_batch)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )


    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)   #here i can understand yeild=return and stop 'while' to run
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__=="__main__":

    train_dataset = BaseDataSets(base_dir='/data/ACDC/', split='train')
    train_len = train_dataset.__len__()
    val_dataset = BaseDataSets(base_dir='/data/ACDC/', split='val')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    image = np.zeros([1,128,128])
    gt = np.zeros([1,128,128])
    for batch_idx, (input_tensor,target) in enumerate(train_loader):
        input_tensor_np = input_tensor.numpy().squeeze()
        target_np = target.numpy().squeeze()
        image = np.vstack((image,input_tensor_np))
        gt = np.vstack((gt, target_np))
    image_new = image[1:]
    gt_new = gt[1:]
    np.save('ACDC/clean/train_image.npy',image_new)
    np.save('/ACDC/clean/train_label.npy', gt_new)

    image = np.zeros([1, 128, 128])
    gt = np.zeros([1, 128, 128])
    for batch_idx, (input_tensor,target) in enumerate(val_loader):
        input_tensor_np = input_tensor.numpy().squeeze()
        target_np = target.numpy().squeeze()
        image = np.vstack((image,input_tensor_np))
        gt = np.vstack((gt, target_np))
    image_new = image[1:]
    gt_new = gt[1:]
    np.save('/ACDC/clean/test_image.npy',image_new)
    np.save('/ACDC/clean/test_label.npy', gt_new)
    x=1

