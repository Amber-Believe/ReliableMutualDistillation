#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.utils.general import prepare_input
from lib.visual3D_temp.co_teaching_BaseWriter_2 import TensorboardWriter
from lib import ramps
import math
from scipy import *
from lib.metric import hd,asd,assd
from lib.surface_distance.metrics import compute_surface_distances,\
    compute_average_surface_distance,compute_robust_hausdorff,compute_surface_overlap_at_tolerance
from lib.medloaders.augment import *
from lib.losses3D.basic import compute_per_channel_dsc

class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model1, model2, rate_schedule, criterion, optimizer1, train_data_loader_weak,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        #self.optimizer2 = optimizer2
        self.criterion = criterion
        self.rate_schedule = rate_schedule
        self.train_data_loader_weak = train_data_loader_weak
        # self.train_data_loader_strong = train_data_loader_strong
        # epoch-based training
        self.len_epoch = len(self.train_data_loader_weak)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader_weak.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 1
        self.terminal_show_freq = 100
        #        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 0
        self.best_dsc = 0
        self.forget = 0
        self.epochs_since_improvement = 0
        self.iter_num = 0
        self.consist_factor = args.consist_factor
        self.num_aug = args.num_aug
        self.aug_mean = 0
        self.aug_std = 0.02

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.args.lr] * self.args.nEpochs
        self.beta1_plan = [mom1] * self.args.nEpochs
        for i in range(args.epoch_decay_start, self.args.nEpochs):
            self.alpha_plan[i] = float(self.args.nEpochs - i) / (
                        self.args.nEpochs - args.epoch_decay_start) * self.args.lr
            self.beta1_plan[i] = mom2


    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):

            adjust_learning_rate(self.optimizer1, epoch, self.alpha_plan, self.beta1_plan)
            #adjust_learning_rate(self.optimizer2, epoch, self.alpha_plan, self.beta1_plan)
            #if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 8 == 0:
                #adjust_learning_rate(self.optimizer1, 0.8)
                #adjust_learning_rate(self.optimizer2, 0.8)

            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            val_loss2 = self.writer.data['val2']['loss'] / self.writer.data['val2']['count']
            recent_val_dsc = self.writer.data['val']['dsc'] / self.writer.data['val']['count']
            recent_val_dsc2 = self.writer.data['val2']['dsc'] / self.writer.data['val2']['count']
            if self.args.save is not None and (self.save_frequency):
            #if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model1.save_checkpoint(self.args.save,
                                            epoch, recent_val_dsc,
                                            optimizer=self.optimizer1,
                                            model_num="model1")
                self.model2.save_checkpoint(self.args.save,
                                            epoch, recent_val_dsc2,
                                            optimizer=self.optimizer1,
                                            model_num="model2")

            # Check if there was an improvement
            is_best = recent_val_dsc > self.best_dsc
            self.best_dsc = max(recent_val_dsc, self.best_dsc)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('train2')
            self.writer.reset('val')
            self.writer.reset('val2')



    def train_epoch(self, epoch):
        self.model1.train()
        self.model2.train()
        # batch_idx = 0
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader_weak):

            consistency_weight = self.get_current_consistency_weight(epoch)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            input_tensor = torch.unsqueeze(input_tensor, 1)
            target = torch.unsqueeze(target, 1)
            output1, output1_aux1, output1_aux2, output1_aux3 = self.model1(input_tensor,mode = "weak")
            output2, output2_aux1, output2_aux2, output2_aux3 = self.model2(input_tensor,mode = "weak")

            #jocor
            n,ch,h,w=target.size()
            remember_rate = 1 - self.rate_schedule[epoch]
            num_remember = int(remember_rate * (n * w * h))
            loss_jo=0
            per_ch_score1=[]
            per_ch_score2=[]
            co_lambda=0.1
            for i in range(ch):
                output1_vec = torch.flatten(output1[:,i])
                output2_vec = torch.flatten(output2[:,i])
                target_vec = torch.flatten(target[:,i])
                loss_pick_1,_ = self.criterion(output1_vec, target_vec)
                loss_pick_2,_ = self.criterion(output2_vec, target_vec)
                loss_pick = ((loss_pick_1 + loss_pick_2) * (1 - co_lambda)
                             + ((output1_vec - output2_vec) ** 2) * co_lambda * consistency_weight)
                idx_sorted = torch.argsort(loss_pick)
                loss_tmp = torch.mean(loss_pick[idx_sorted[:num_remember]])
                per_ch_score1_tmp = compute_per_channel_dsc(torch.nn.Sigmoid()(output1_vec[idx_sorted[:num_remember]]),
                                                        target_vec[idx_sorted[:num_remember]],mode="train")
                per_ch_score2_tmp = compute_per_channel_dsc(torch.nn.Sigmoid()(output2_vec[idx_sorted[:num_remember]]),
                                                        target_vec[idx_sorted[:num_remember]],mode="train")
                loss_jo=loss_jo+loss_tmp

                per_ch_score1.append(per_ch_score1_tmp.detach().cpu().numpy())
                per_ch_score2.append(per_ch_score2_tmp.detach().cpu().numpy())
            loss_jo=loss_jo/ch


            consist_rate  = min( consistency_weight*self.consist_factor,1)
            num_consist = int(consist_rate * (n*h*w))
            loss_consist_1=0
            loss_consist_2=0
            for i in range(self.num_aug):
                input_aug,flipCode = flip(input_tensor)
                output1_aug,_ = flip(output1,flipCode)
                output2_aug,_ = flip(output2,flipCode)

                if epoch > 9:
                    input_aug = gasuss_noise(input_aug, self.aug_mean, self.aug_std)
                    input_aug,disps,rotates,zooms = deform(input_aug)
                    output1_aug,_,_,_ = deform(output1_aug,disps,rotates,zooms)
                    output2_aug,_,_,_ = deform(output2_aug,disps,rotates,zooms)
                    input_aug = input_aug.to(device)
                    output1_aug = output1_aug.to(device)
                    output2_aug = output2_aug.to(device)

                #mutual aug
                aug_output1,_,_,_ = self.model1(input_aug,mode = "strong")
                aug_output2,_,_,_ = self.model2(input_aug,mode = "strong")

                certaintymap1 = (aug_output1.detach()-output1_aug.detach())**2
                certaintymap2 = (aug_output2.detach()-output2_aug.detach())**2

                for j in range(ch):
                    output1_aug_vec=output1_aug[:,j].flatten()
                    output2_aug_vec=output2_aug[:,j].flatten()
                    aug_output1_vec=aug_output1[:,j].flatten()
                    aug_output2_vec=aug_output2[:,j].flatten()
                    certaintymap1_vec=certaintymap1[:,j].flatten()
                    certaintymap2_vec=certaintymap2[:,j].flatten()

                    idxcert1_sorted = torch.argsort(certaintymap1_vec)[:num_consist]
                    idxcert2_sorted = torch.argsort(certaintymap2_vec)[:num_consist]

                    loss_tmp1 = self.kl_loss_compute(aug_output1_vec[idxcert2_sorted], output2_aug_vec[idxcert2_sorted])
                    loss_tmp2 = self.kl_loss_compute(aug_output2_vec[idxcert1_sorted], output1_aug_vec[idxcert1_sorted])

                    loss_consist_1 += loss_tmp1
                    loss_consist_2 += loss_tmp2

            #loss_last = loss
            loss_last_1 =  loss_jo + (loss_consist_1 +loss_consist_2)/2/self.num_aug/ch * consistency_weight #* 0.0009

            self.optimizer1.zero_grad()
            loss_last_1.backward()
            self.optimizer1.step()

            self.iter_num += 1
            self.writer.update_scores(batch_idx, loss_pick_1.mean().item(),per_ch_score1, 'train',
                                      epoch * self.len_epoch + batch_idx)
            self.writer.update_scores(batch_idx, loss_pick_2.mean().item(),per_ch_score2, 'train2',
                                      epoch * self.len_epoch + batch_idx)
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')
                self.writer.display_terminal(partial_epoch, epoch, 'train2')
            # batch_idx +=1
            # del loss_1, loss_2
        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)
        self.writer.display_terminal(self.len_epoch, epoch, mode='train2', summary=True)


    def validate_epoch(self, epoch):
        self.model1.eval()
        self.model2.eval()


        for batch_idx, (input_tensor, target) in enumerate(self.valid_data_loader):
            with torch.no_grad():
                # input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input_tensor = input_tensor.to(device)
                target = target.to(device)
                input_tensor = torch.unsqueeze(input_tensor, 1)
                target = torch.unsqueeze(target, 1)
                output1,_,_,_ = self.model1(input_tensor)
                output2,_,_,_ = self.model2(input_tensor)
                _,loss1 = self.criterion(output1, target)
                _,loss2 = self.criterion(output2, target)
                output1 = nn.Sigmoid()(output1)
                output2 = nn.Sigmoid()(output2)
                dsc = (compute_per_channel_dsc(output1, target)).detach().cpu().numpy()
                dsc2 = (compute_per_channel_dsc(output2, target)).detach().cpu().numpy()


                self.writer.update_scores(batch_idx, loss1.item(), dsc, 'val',
                                          epoch * self.len_epoch + batch_idx)
                self.writer.update_scores(batch_idx, loss2.item(), dsc2, 'val2',
                                          epoch * self.len_epoch + batch_idx)
        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val2', summary=True)


    def kl_loss_compute(self,pred, soft_targets, reduce=True):
        pred_sig = pred.sigmoid()
        tgt_sig = soft_targets.sigmoid()
        kl = F.kl_div(torch.log(pred_sig), tgt_sig, reduce=False)+F.kl_div(torch.log(1-pred_sig), 1-tgt_sig, reduce=False)
        return kl.mean()

    def mse_compute(self,pred, soft_targets, reduce=True):
        mse = (pred-soft_targets.detach())**2
        return mse.mean()


    def get_current_consistency_weight(self,epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)

def adjust_learning_rate(optimizer, epoch,alpha_plan,beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

