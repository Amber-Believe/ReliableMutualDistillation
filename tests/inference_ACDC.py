import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn.functional as F

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
from lib.medzoo.Unet2D import Unet
import torch.nn as nn
import numpy as np
import cv2 as cv
from lib.surface_distance.metrics import compute_surface_distances,\
    compute_average_surface_distance,compute_robust_hausdorff,compute_surface_overlap_at_tolerance
#from hausdorff import hausdorff_distance


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




def main():
    args = get_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    _, test_generator = medical_loaders.generate_datasets(args,path='/home/qwang/Desktop/ACDC/noisy_0.6_new2/')
    model = Unet(1,4)
    model.cuda()
    ckpt_dict = torch.load(args.pretrained)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    #

    lv_dices = []
    rv_dices = []
    myo_dices = []
    lv_hd_95 = []
    rv_hd_95 = []
    myo_hd_95 = []
    lv_asd = []
    rv_asd = []
    myo_asd = []
    lv_overlap_gt_3 = []
    lv_overlap_pred_3 = []
    rv_overlap_gt_3 = []
    rv_overlap_pred_3 = []
    myo_overlap_gt_3 = []
    myo_overlap_pred_3 = []

    #
    for batch_idx, (input_tensor, target) in enumerate(test_generator):

        model.eval()
        input_tensor = torch.unsqueeze(input_tensor, 1)
        #target = torch.unsqueeze(target, 1)
        output, _, _, _ = model(input_tensor)
        output_label = torch.max(output, 1)[1]
        lv_output = np.squeeze(((output_label==1).type(torch.FloatTensor)).data.cpu().numpy())
        lv_label = np.squeeze(((target == 1).type(torch.FloatTensor)).data.cpu().numpy())
        rv_output = np.squeeze(((output_label == 2).type(torch.FloatTensor)).data.cpu().numpy())
        rv_label = np.squeeze(((target == 2).type(torch.FloatTensor)).data.cpu().numpy())
        myo_output = np.squeeze(((output_label == 3).type(torch.FloatTensor)).data.cpu().numpy())
        myo_label = np.squeeze(((target == 3).type(torch.FloatTensor)).data.cpu().numpy())

        """Dice"""

        if lv_output.max()==0 and lv_label.max()==0:
            dice=1
        else:
            dice = dice_coef(lv_output, lv_label)
        lv_dices.append(dice)
        #
        if rv_output.max()==0 and rv_label.max()==0:
            dice=1
        else:
            dice = dice_coef(rv_output, rv_label)
        rv_dices.append(dice)
        #
        if myo_output.max()==0 and myo_label.max()==0:
            dice=1
        else:
            dice = dice_coef(myo_output, myo_label)
        myo_dices.append(dice)

        #
        """ASD Overlap"""
        if lv_output.max()==0 and lv_label.max()==0:
            Hausdorff=0
            lv_hd_95.append(Hausdorff)
            ASD=0
            lv_asd.append(ASD)
        elif lv_output.max()==0 or lv_label.max()==0:
            pass
        else:
            surface_distances = compute_surface_distances(lv_label,lv_output,[1.0, 1.0])
            Hausdorff=(compute_robust_hausdorff(surface_distances, 95))
            lv_hd_95.append(Hausdorff)
            p1, p2 = compute_average_surface_distance(surface_distances)
            ASD = (((p1 + p2) / 2.0))
            lv_asd.append(ASD)
            p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 1.0)
            lv_overlap_gt_3.append(p1)
            lv_overlap_pred_3.append(p2)
        #
        if rv_output.max()==0 and rv_label.max()==0:
            Hausdorff=0
            rv_hd_95.append(Hausdorff)
            ASD=0
            rv_asd.append(ASD)
        elif rv_output.max()==0 or rv_label.max()==0:
            pass
        else:
            surface_distances = compute_surface_distances(rv_label,rv_output,[1.0, 1.0])
            Hausdorff=(compute_robust_hausdorff(surface_distances, 95))
            rv_hd_95.append(Hausdorff)
            p1, p2 = compute_average_surface_distance(surface_distances)
            ASD = (((p1 + p2) / 2.0))
            rv_asd.append(ASD)
            p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 1.0)
            rv_overlap_gt_3.append(p1)
            rv_overlap_pred_3.append(p2)
        #
        if myo_output.max()==0 and myo_label.max()==0:
            Hausdorff=0
            myo_hd_95.append(Hausdorff)
            ASD=0
            myo_asd.append(ASD)
        elif myo_output.max()==0 or myo_label.max()==0:
            pass
        else:
            surface_distances = compute_surface_distances(myo_label,myo_output,[1.0, 1.0])
            Hausdorff=(compute_robust_hausdorff(surface_distances, 95))
            myo_hd_95.append(Hausdorff)
            p1, p2 = compute_average_surface_distance(surface_distances)
            ASD = (((p1 + p2) / 2.0))
            myo_asd.append(ASD)
            p1, p2 = compute_surface_overlap_at_tolerance(surface_distances, 1.0)
            myo_overlap_gt_3.append(p1)
            myo_overlap_pred_3.append(p2)
        #
    mean_dice = (np.mean(lv_dices) + np.mean(rv_dices) + np.mean(myo_dices)) / 3.0
    mean_hd = (np.mean(lv_hd_95) + np.mean(rv_hd_95) + np.mean(myo_hd_95)) / 3.0
    mean_asd = (np.mean(lv_asd) + np.mean(rv_asd) + np.mean(myo_asd)) / 3.0
    mean_over_g = (np.mean(lv_overlap_gt_3) + np.mean(rv_overlap_gt_3) + np.mean(myo_overlap_gt_3)) / 3.0
    mean_over_p = (np.mean(lv_overlap_pred_3) + np.mean(rv_overlap_pred_3) + np.mean(myo_overlap_pred_3)) / 3.0
    print('lv Dice: %.4f' % np.mean(lv_dices))
    print('rv Dice: %.4f' % np.mean(rv_dices))
    print('myo Dice: %.4f' % np.mean(myo_dices))
    print('mean   : %.4f' % mean_dice)
    print("=============")
    print('lv HD: %.4f' % np.mean(lv_hd_95))
    print('rv HD: %.4f' % np.mean(rv_hd_95))
    print('myo HD: %.4f' % np.mean(myo_hd_95))
    print('mean : %.4f' % mean_hd)
    print("=============")
    print('lv ASD: %.4f' % np.mean(lv_asd))
    print('rv ASD: %.4f' % np.mean(rv_asd))
    print('myo ASD: %.4f' % np.mean(myo_asd))
    print('mean  : %.4f' % mean_asd)
    print("=============")
    print('lv overlap_gt_3: %.4f' % np.mean(lv_overlap_gt_3))
    print('rv overlap_gt_3: %.4f' % np.mean(rv_overlap_gt_3))
    print('myo overlap_gt_3: %.4f' % np.mean(myo_overlap_gt_3))
    print('mean           : %.4f' % mean_over_g)
    print("=============")
    print('lv overlap_pre_3: %.4f' % np.mean(lv_overlap_pred_3))
    print('rv overlap_pre_3: %.4f' % np.mean(rv_overlap_pred_3))
    print('myo overlap_pre_3: %.4f' % np.mean(myo_overlap_pred_3))
    print('mean            : %.4f' % mean_over_p)
    print("=============")



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="ACDC")

    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--pretrained',
                        default='/home/qwang/wq/code/saved_models/UNET2D_checkpoints/UNET2D_02_06___11_40_ACDC_/model2_UNET2D_02_06___11_40_ACDC__38_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()











'''

def overlap_3d_image():
    B, C, D, H, W = 2, 1, 144, 192, 256
    #B, C, D, H, W = 1, 1, 4, 4, 4
    x = torch.randn(B, C, D, H, W)
    print('IMAGE shape ', x.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    kernel_size = 32
    stride = 16
    patches = x.unfold(4, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, H, num_of_patches_W, kernel_size]
    patches = patches.unfold(3, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    patches = patches.unfold(2, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size]
    # patches = patches.unfold()
    # perform the operations on each patchff
    # ...
    B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size = patches.shape
    # # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    print('slice shape ',patches[:,:,0,:,:].shape)
    slices = []
    for i in range(num_of_patches_D * kernel_size):

        output = F.fold(
              patches[:,:,i,:,:].contiguous().view(B, C * kernel_size * kernel_size,-1), output_size=(H, W), kernel_size=kernel_size, stride=stride)
        #print(output.shape)  # [B, C, H, W]
        slices.append(output)
    image = torch.stack(slices)
    print(image.shape)
    print(image.is_contiguous())
    image = image.permute(1,2,0,3,4).contiguous().view(B,C,-1,H*W)
    print(image.shape)
    output = F.fold(
        image.contiguous().view(B*H*W, C*kernel_size, -1), output_size=(D, 1), kernel_size=kernel_size, stride=stride)
    print(output.shape)  # [B, C, H, W]


'''