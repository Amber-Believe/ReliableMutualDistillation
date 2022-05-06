import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import torch.nn as nn
import numpy as np
import cv2 as cv
#from hausdorff import hausdorff_distance
from lib.medzoo.Unet2D_base import Unet
from lib.surface_distance.metrics import compute_surface_distances, \
    compute_average_surface_distance, compute_robust_hausdorff, compute_surface_overlap_at_tolerance
import matplotlib.pyplot as plt

"""LIDC/BraTS_flair"""

def dice_coef(output, target):
    smooth = 1e-6

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def batch_iou(output, target):

    ious = []
    for i in range(output.shape[0]):
        ious.append(iou_score(output[i], target[i]))

    #return np.mean(ious)
    return ious

def batch_dice(output, target):

    dice = []
    for i in range(output.shape[0]):
        dice.append(dice_coef(np.squeeze(output[i]), np.squeeze(target[i])))

    return dice

from lib.losses3D.basic import compute_per_channel_dsc
from lib.medloaders.augment import *
def main():
    args = get_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)



    _,  test_generator = medical_loaders.generate_datasets(args,path='/home/LIDC/')


    model = Unet(1,1)
    model.cuda()
    ckpt_dict = torch.load(args.pretrained)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    #
    asd_gt_to_pred = []
    asd_pred_to_gt = []
    hd_75 = []
    hd_100 = []
    assd_mean = []
    overlap_gt_1 = []
    overlap_pred_1 = []
    overlap_gt_3 = []
    overlap_pred_3 = []
    overlap_gt_5 = []
    overlap_pred_5 = []
    dsc_idx=[]
    #
    for batch_idx, (input_tensor, target) in enumerate(test_generator):

        model.eval()
        with torch.no_grad():
            input_tensor.requires_grad = False
            input_tensor = torch.unsqueeze(input_tensor, 1)
            target = torch.unsqueeze(target, 1)
            output = model(input_tensor)
            #

        output = nn.Sigmoid()(output)
        output = (output.detach() > 0.5).cpu().type(torch.FloatTensor).numpy()
        target = target.detach().cpu().numpy()
        dsc = (dice_coef(output, target ))
        dsc_idx.append(dsc)
        for i in range(len(target)):
            output_one = output[i]
            target_one = target[i]
            if output_one.max() == 1 and target_one.max() == 1:

                surface_distances = compute_surface_distances(np.squeeze(output_one),
                                                              np.squeeze(target_one), [1.0, 1.0])
                hd_75.append(compute_robust_hausdorff(surface_distances, 75))
                hd_100.append(compute_robust_hausdorff(surface_distances, 100))
                p1, p2 = compute_average_surface_distance(surface_distances)
                asd_gt_to_pred.append(p1)
                asd_pred_to_gt.append(p2)
                assd_mean.append(((p1 + p2) / 2.0))
                p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 1.0)
                overlap_gt_1.append(p1)
                overlap_pred_1.append(p2)
                p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 3.0)
                overlap_gt_3.append(p1)
                overlap_pred_3.append(p2)
                p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 5.0)
                overlap_gt_5.append(p1)
                overlap_pred_5.append(p2)


    print('dsc: %.4f' % np.mean(dsc_idx))
    print('hd_75: %.4f' % np.mean(hd_75))
    print('hd_100: %.4f' % np.mean(hd_100))
    print('asd_gt_to_pred : %.4f' % np.mean(asd_gt_to_pred))
    print('asd_pred_to_gt: %.4f' % np.mean(asd_pred_to_gt))
    print('assd_mean: %.4f' % np.mean(assd_mean))
    print('overlap_gt_1: %.4f' % np.mean(overlap_gt_1))
    print('overlap_pred_1: %.4f' % np.mean(overlap_pred_1))
    print('overlap_gt_3: %.4f' % np.mean(overlap_gt_3))
    print('overlap_pred_3: %.4f' % np.mean(overlap_pred_3))
    print('overlap_gt_5: %.4f' % np.mean(overlap_gt_5))
    print('overlap_pred_5: %.4f' % np.mean(overlap_pred_5))





def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default="LIDC")
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--pretrained',
                        default='/home/model.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()











