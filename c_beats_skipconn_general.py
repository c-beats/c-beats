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


class CBeatsNet(nn.Module):

    def __init__(self,

                 # 1) input
                cast_lengths=(20,5), # back & fore

                 # 2) architecture (stack 차원)
                stack_types=('trend','seasonality'),
                stack_sizes=(2,2),

                # 3) architecture (block 차원)
                num_layers_CNN=(1,1),
                num_layers_FC=(1,1),
                thetas_dim=(3,7), # season & trend
                kernels=(3,3),
                channels=(32,32),
                padding_mode = 'zeros',

                # 4) etc
                device=torch.device('cuda'),
                seed=402):

        super(CBeatsNet, self).__init__()

        assert(len(stack_types)==len(stack_sizes)==len(num_layers_CNN)==len(num_layers_FC)),' ( # types of stack != # block in stack )'
        assert(len(kernels)==len(channels)),' ( # kernel != # channels ) '
        assert padding_mode in ['zeros', 'reflect', 'replicate']

        # 1) input size
        self.backcast_length = cast_lengths[0]
        self.forecast_length = cast_lengths[1]


        # 2) architecture (stack)
        self.stack_types = stack_types
        self.stack_sizes = stack_sizes

        # 3) architecture (block)
        self.num_layers_CNN=num_layers_CNN
        self.num_layers_FC=num_layers_FC
        self.thetas_dim = thetas_dim
        self.kernels = kernels
        self.channels = (1,*channels)
        self.padding_mode = padding_mode

        # 4) etc
        self.device = device
        self.seed = seed

        # 5) create stacks
        self.stacks = []
        self.parameters=[]
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)


        self.to(self.device)
        seed_everything(self.seed)


    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        for block_id in range(self.stack_sizes[stack_id]):
            if stack_type=='seasonality':
                block_init= SeasonalityBlock
            elif stack_type=='trend':
                block_init= TrendBlock
            block = block_init(self.backcast_length,self.forecast_length,
                               self.kernels[block_id], self.channels[block_id:block_id+2],self.padding_mode,
                               self.thetas_dim[stack_id],
                               self.num_layers_CNN[stack_id],self.num_layers_FC[stack_id],
                               self.device,  self.seed)
            self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks

    def forward(self, backcast, return_stack=False):
        h_prev = backcast.unsqueeze(1)
        backcast = torch.tensor(h_prev).squeeze(1)
        forecast = torch.zeros((backcast.size(0), self.forecast_length,))

        stack_result = []
        for stack_id in range(len(self.stacks)):
            backcast_cumsum=0
            forecast_cumsum=0
            for block_id in range(len(self.stacks[stack_id])):
                h, b, f = self.stacks[stack_id][block_id](h_prev)
                backcast_cumsum += b
                forecast_cumsum += f
                h_prev = h_prev + h
            backcast = backcast.to(self.device) - backcast_cumsum
            forecast = forecast.to(self.device) + forecast_cumsum
            stack_result.append(forecast)
            h_prev = backcast.unsqueeze(1)

        if return_stack == True:
          return backcast, forecast, stack_result
        return backcast, forecast

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore



def seasonality_model(thetas, t, device):
    p = thetas.size(-1)
    assert p <= thetas.size(1), 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size(-1)
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = np.abs(np.flip(ls[:backcast_length]))
    f_ls = ls[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, backcast_length, forecast_length,
                 kernels, channels,padding_mode,thetas_dim,num_layers_CNN,num_layers_FC,
                 device,seed=402):
        super(Block, self).__init__()
        self.seed = seed
        seed_everything(self.seed)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.kernels = kernels
        self.channels = channels
        self.padding_mode = padding_mode
        self.thetas_dim = thetas_dim
        self.num_layers_CNN=num_layers_CNN
        self.num_layers_FC=num_layers_FC
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)

        self.theta_f_fc = self.theta_b_fc = nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)




def layers_CNN(in_, out_, kernels):
    return Sequential(Conv1d(in_, out_, kernels,padding='same'),
                      BatchNorm1d(out_),
                      ReLU(inplace=True))

def layers_FC(in_, out_):
    return Sequential(nn.Linear(in_, out_, bias=False),
                      BatchNorm1d(out_),
                      ReLU(inplace=True))



class SeasonalityBlock(Block):
    def __init__(self, backcast_length, forecast_length,
                 kernels, channels,padding_mode,
                 thetas_dim,num_layers_CNN,num_layers_FC,
                 device,seed=402):
        super(SeasonalityBlock, self).__init__(backcast_length,forecast_length,
                                                kernels, channels, padding_mode,
                                                thetas_dim,num_layers_CNN,num_layers_FC,
                                                device, seed=402)
        seed_everything(self.seed)


        cnn_layers = [Sequential(Conv1d(channels[0], channels[1], kernels,padding=1,padding_mode=padding_mode),
                                       BatchNorm1d(channels[1]),
                                       ReLU(inplace=True))]
        cnn_layers = cnn_layers + [layers_CNN(channels[1], channels[1],kernels)  for _ in range(num_layers_CNN-1)]
        self.season_conv = nn.Sequential(*cnn_layers).cuda()

        #fc_layers = [layers_FC(backcast_length*channels[1],backcast_length*channels[1]) for _ in range(num_layers_FC-1)]
        #fc_layers = fc_layers + [nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)]
        #self.theta_f_fc = self.theta_b_fc =  nn.Sequential(*fc_layers).cuda()



    def forward(self, x):
        h = self.season_conv(x.to(self.device))
        h_flatten = h.flatten(1)
        theta_back_pred=self.theta_b_fc(h_flatten)
        theta_fore_pred=self.theta_f_fc(h_flatten)
        backcast = seasonality_model(theta_back_pred,self.backcast_linspace, self.device)
        forecast = seasonality_model(theta_fore_pred,self.forecast_linspace, self.device)
        return h,backcast,forecast


class TrendBlock(Block):
    def __init__(self, backcast_length,forecast_length,
                 kernels,channels,padding_mode,
                 thetas_dim,num_layers_CNN,num_layers_FC,
                 device, seed=402):
        super(TrendBlock, self).__init__(backcast_length,forecast_length,
                                            kernels, channels,padding_mode,
                                            thetas_dim,num_layers_CNN, num_layers_FC,
                                            device,seed=402)
        seed_everything(self.seed)

        cnn_layers = [Sequential(Conv1d(channels[0], channels[1], kernels,padding=1,padding_mode=padding_mode),
                                       BatchNorm1d(channels[1]),
                                       ReLU(inplace=True))]
        cnn_layers = cnn_layers + [layers_CNN(channels[1], channels[1], kernels)  for _ in range(num_layers_CNN-1)]
        self.trend_conv = nn.Sequential(*cnn_layers).cuda()

        #fc_layers = [layers_FC(backcast_length*channels[1],backcast_length*channels[1]) for _ in range(num_layers_FC-1)]
        #fc_layers = fc_layers + [nn.Linear(backcast_length*channels[1], thetas_dim, bias=False)]
        #self.theta_f_fc = self.theta_b_fc =  nn.Sequential(*fc_layers).cuda()



    def forward(self,x):
        h = self.trend_conv(x.to(self.device))
        h_flatten = h.flatten(1)
        theta_back_pred=self.theta_b_fc(h_flatten)
        theta_fore_pred=self.theta_f_fc(h_flatten)
        backcast = trend_model(theta_back_pred,self.backcast_linspace, self.device)
        forecast = trend_model(theta_fore_pred,self.forecast_linspace, self.device)
        return h,backcast,forecast
