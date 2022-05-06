#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch

from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model,  criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = 100
#        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.best_dsc= 0
        self.epochs_since_improvement = 0

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):

            if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 8 == 0:
                adjust_learning_rate(self.optimizer, 0.8)

            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            recent_val_dsc = self.writer.data['val']['dsc'] / self.writer.data['val']['count']
            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)

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
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx,(input_tensor,target)in enumerate(self.train_data_loader):
#        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)
            target = target.to(device)

            input_tensor = torch.unsqueeze(input_tensor, 1)
            target = torch.unsqueeze(target, 1)
            self.optimizer.zero_grad()

#            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
#            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            _, per_ch_score, loss_dice = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx,( input_tensor, target) in enumerate(self.valid_data_loader):
            with torch.no_grad():
                #input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input_tensor = input_tensor.to(device)
                target = target.to(device)
                input_tensor = torch.unsqueeze(input_tensor, 1)
                target = torch.unsqueeze(target, 1)

                output = self.model(input_tensor)
                _, per_ch_score, loss = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))