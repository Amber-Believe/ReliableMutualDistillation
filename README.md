# ReliableMutualDistillation
The official pytorch implemention of the paper ”Reliable Mutual Distillation for Medical Image Segmentation under Imperfect Annotations”

# Introduction
We propose a novel pipeline to cope with noises in imperfect annotations for medical image segmentation, based on the reliable mutual distillation between dual segmentation models.We attempt to incorporate our method with other baseline models, including Unet, Unet++ and DenseUnet. We evaluate our method on three datasets with synthesized noisy annotations: LIDC-IDRI, ACDC and BraTS. We create imperfect annotations with moderate boundary offsets and shape distortions according to three strategies:Mask Distortion,Mask Erosion/Dilation and Real-world Imperfect Annotations.

# Useage
## Requirment
code/installation/requirements.txt
Python>=3.6
Pytorch>=1.8.1

## Training
For LIDC-IDRI dataset , run:
cd code/tests/
python train_with_trainer_class.py
## Testing
To test our model, please run inference_LIDC.py with the following setting:
1.change the model_path to your pre-trained model;
2.change the test_path to your testing data.
# Citation
@article{ReliableMutualDistillation,
  title={Reliable Mutual Distillation for Medical Image Segmentation under Imperfect Annotations},
  author={Fang, Chaowei and Wang, Qian and Cheng, Lechao and Gao, Zhifan and Pan, Chengwei and Cao, Zhen and Zhang, Dingwen},
  year={2022},
}
