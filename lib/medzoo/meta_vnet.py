#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    x = x.type(torch.float32)
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaConv2d, self).__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', ignore.weight)

        if ignore.bias is not None:
            self.register_buffer('bias', ignore.bias)
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        x1=x.shape
        x2=self.weight.shape
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class ConvVNetOne(MetaModule):
    def __init__(self):
        super(ConvVNetOne, self).__init__()
        self.conv3x3 = MetaConv2d(1, 32, 3, 1, 1)
        self.conv5x5 = MetaConv2d(1, 32, 5, 1, 2)
        self.out3x3 = MetaConv2d(32, 1, 3, 1, 1)
        self.out5x5 = MetaConv2d(32, 1, 5, 1, 2)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu3_2 = nn.ReLU()
        self.relu5_2 = nn.ReLU()
        self.outc = MetaConv2d(3, 1, 3, 1, 1)
        return

    def forward(self, x):
        x3 = self.out3x3(self.relu3(self.conv3x3(x)))
        x5 = self.out5x5(self.relu5(self.conv5x5(x)))
        x = torch.cat([x,x3,x5], dim=1)
        x = self.outc(x)
        return torch.sigmoid(x)


