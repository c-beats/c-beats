import pickle
import random
import math
import os
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import *
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


def seed_everything(seed = 402):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore


class CBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'

    def __init__(self,
                 device=torch.device('cuda'),
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=(1,1),
                 forecast_length=5,
                 backcast_length=20,
                 thetas_dim=(4, 7),
                 share_weights_in_stack=False,
                 kernels=(3,3,3),
                 channels=(1,32,32,32),
                 middle_layer_dim=64,
                 seed=402,
                 nb_harmonics=None):
        super(CBeatsNet, self).__init__()
        self.seed = seed
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        self.kernels = kernels
        self.channels = channels

        self.middle_layer_dim = middle_layer_dim
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.parameters=[]
        self.thetas_dim = thetas_dim
        self.device = device
        print('| C-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None

        seed_everything(self.seed)



    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack[stack_id]):
            block_init = CBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                #print(self.channels[block_id:block_id+2])
                block = block_init(self.kernels[block_id], self.channels[block_id:block_id+2],
                                   self.thetas_dim[stack_id], self.device, self.backcast_length,
                                   self.forecast_length, self.seed, self.nb_harmonics)
            self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == CBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == CBeatsNet.TREND_BLOCK:
            return TrendBlock


    def forward(self, backcast, return_stack=False):
        h = squeeze_last_dim(backcast).unsqueeze(1)
        backcast = torch.tensor(h).squeeze(1)
        forecast = torch.zeros(size=(backcast.size(0), self.forecast_length,))  # maybe batch size here.
        stack_res = []
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                h, b, f = self.stacks[stack_id][block_id](h)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
            stack_res.append(forecast)
            h = backcast.unsqueeze(1)

        if return_stack == True:
          return backcast, forecast, stack_res
        return backcast, forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = np.abs(np.flip(ls[:backcast_length]))
    f_ls = ls[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, kernels, channels, thetas_dim,
                 device, backcast_length=10, forecast_length=5,
                 share_thetas=False, seed=402, nb_harmonics=None):
        super(Block, self).__init__()
        self.seed = seed
        seed_everything(self.seed)

        self.kernels = kernels
        self.channels = channels
        self.thetas_dim = thetas_dim

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        self.device = device
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)
            #self.theta_f_fc = self.theta_b_fc = Sequential(nn.Linear(backcast_length*channels[1], backcast_length*channels[1]//2, bias=False),
            #                                               BatchNorm1d(backcast_length*channels[1]//2),
            #                                               ReLU(inplace=True),
                                                                #nn.Linear(backcast_length*channels[1]//2, thetas_dim, bias=False)).cuda()
        else:
            self.theta_b_fc = nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)
            #self.theta_f_fc nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)
            #self.theta_b_fc = Sequential(nn.Linear(backcast_length*channels[1], backcast_length*channels[1]//2, bias=False),
        #                                 BatchNorm1d(backcast_length*channels[1]//2),
        #                                 ReLU(inplace=True),
        #                                 nn.Linear(backcast_length*channels[1]//2, thetas_dim, bias=False)).cuda()
        #
        #    self.theta_f_fc = Sequential(nn.Linear(backcast_length*channels[1], backcast_length*channels[1]//2, bias=False),
        #                                 BatchNorm1d(backcast_length*channels[1]//2),
        #                                 ReLU(inplace=True),
        #                                 nn.Linear(backcast_length*channels[1]//2, thetas_dim, bias=False)).cuda()

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'




class SeasonalityBlock(Block):
    def __init__(self, kernels,  channels, thetas_dim,
                 device, backcast_length=10, forecast_length=5,
                 seed=402, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(kernels,channels, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True, seed=402)
        else:
            super(SeasonalityBlock, self).__init__(kernels, channels, thetas_dim, device, backcast_length,
                                                   forecast_length, share_thetas=True, seed=402)
        seed_everything(self.seed)
        self.season_conv = Sequential(Conv1d(channels[0], channels[1], kernel_size=kernels,padding=1),
                                       BatchNorm1d(channels[1]),
                                       ReLU(inplace=True)).cuda()


    def forward(self, x):
        h = self.season_conv(x.to(self.device))
        h_flatten = h.flatten(1)
        theta_back_pred=self.theta_b_fc(h_flatten)
        theta_fore_pred=self.theta_f_fc(h_flatten)
        backcast = seasonality_model(theta_back_pred,self.backcast_linspace, self.device)
        forecast = seasonality_model(theta_fore_pred,self.forecast_linspace, self.device)
        return h,backcast,forecast


class TrendBlock(Block):
    def __init__(self, kernels,channels, thetas_dim,
                 device, backcast_length=10, forecast_length=5,
                 seed=402, nb_harmonics=None):
        super(TrendBlock, self).__init__(kernels, channels, thetas_dim, device, backcast_length,
                                          forecast_length, share_thetas=True, seed=402)
        seed_everything(self.seed)
        self.trend_conv = Sequential(Conv1d(channels[0], channels[1], kernel_size=kernels,padding=1),
                                       BatchNorm1d(channels[1]),
                                       ReLU(inplace=True)).cuda()

    def forward(self,x):
        h = self.trend_conv(x.to(self.device))
        h_flatten = h.flatten(1)
        theta_back_pred=self.theta_b_fc(h_flatten)
        theta_fore_pred=self.theta_f_fc(h_flatten)
        backcast = trend_model(theta_back_pred,self.backcast_linspace, self.device)
        forecast = trend_model(theta_fore_pred,self.forecast_linspace, self.device)
        return h,backcast,forecast















'''
class TrendBlock(Block):
    def __init__(self, kernels,middle_layer_dim,channels, thetas_dim, device, backcast_length=10, forecast_length=5, seed=402, nb_harmonics=None):
        super(TrendBlock, self).__init__(kernels,middle_layer_dim, channels, thetas_dim, device, backcast_length,
                                          forecast_length, share_thetas=True, seed=402)
        seed_everything(self.seed)
        self.trend_conv1 = Sequential(Conv1d(1, channels[0], kernel_size=kernels[0],padding=1),
                                       BatchNorm1d(channels[0]),
                                       ReLU(inplace=True))
        self.trend_conv2 = Sequential(Conv1d(channels[0], channels[1], kernel_size=kernels[1],padding=1),
                                       BatchNorm1d(channels[1]),
                                       ReLU(inplace=True))
        self.trend_conv3 = Sequential(Conv1d(channels[1], channels[2], kernel_size=kernels[2],padding=1),
                                       BatchNorm1d(channels[2]),
                                       ReLU(inplace=True))

        self.trend_f_fc1 = self.trend_b_fc1 = Linear(640,thetas_dim[0])
        self.trend_f_fc2 = self.trend_b_fc2 = Linear(640,thetas_dim[0])
        self.trend_f_fc3 = self.trend_b_fc3 = Linear(640,thetas_dim[0])

    def forward(self,x):
        h1 = self.season_conv1(x).flatten(1)
        h2 = self.season_conv2(h1).flatten(1)
        h3 = self.season_conv3(h2).flatten(1)

        theta_back_pred1=self.trend_b_fc1(h3)
        theta_back_pred2=self.trend_b_fc2(h2)
        theta_back_pred3=self.trend_b_fc3(h1)

        theta_fore_pred1=self.trend_f_fc1(h3)
        theta_fore_pred2=self.trend_f_fc2(h2)
        theta_fore_pred3=self.trend_f_fc3(h1)


        backcast1 = trend_model(theta_back_pred1,self.backcast_linspace, self.device)
        backcast2 = trend_model(theta_back_pred2,self.backcast_linspace, self.device)
        backcast3 = trend_model(theta_back_pred3,self.backcast_linspace, self.device)
        forecast1 = trend_model(theta_fore_pred1,self.forecast_linspace, self.device)
        forecast2 = trend_model(theta_fore_pred2,self.forecast_linspace, self.device)
        forecast3 = trend_model(theta_fore_pred3,self.forecast_linspace, self.device)

        return backcast1,backcast2,backcast3,forecast1,forecast2,forecast3
'''
