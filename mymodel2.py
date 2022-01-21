from cbeats_utils import * 
from typing import Union
from torch import nn
from torch.nn import *

# ---------------------------------------------------------------------------- #
# NN MODULES

def layers_FC(num_layers, in_, mid_, out_, dropout = 0.2):
    layers_ = []
    if num_layers==1:
        mid_=out_
    layers_.append(Linear(in_, mid_))
    for i in range(num_layers-2):
        layers_.append(BatchNorm1d(mid_))
        layers_.append(ReLU(inplace = True))
        layers_.append(Dropout(dropout))
        layers_.append(Linear(mid_, mid_))
    if num_layers>1:    
        layers_.append(BatchNorm1d(mid_))
        layers_.append(ReLU(inplace = True))
        layers_.append(Dropout(dropout))
        layers_.append(Linear(mid_, out_))
    return nn.Sequential(*layers_)
    
def layers_CNN(num_layers, in_, out_, kernels, strides, padding,padding_mode):
    layers_ = []
    layers_.append(Conv1d(in_, out_, kernels,strides, padding=padding, 
                          padding_mode=padding_mode))
    for i in range(num_layers-1):
        layers_.append(BatchNorm1d(out_))
        layers_.append(ReLU(inplace = True))
        layers_.append(Conv1d(out_, out_, kernels, strides, padding=padding, 
                              padding_mode=padding_mode))
    layers_.append(BatchNorm1d(out_))
    layers_.append(ReLU(inplace = True))
    return nn.Sequential(*layers_)
       
class ConvBlock(nn.Module):
    def __init__(self, num_layers, channels, kernels, 
                 padding, strides, padding_mode, device,  seed = 402):
        super(ConvBlock, self).__init__()
        self.num_layers = num_layers
        self.in_ = channels[0]
        self.out_ = channels[1]
        self.kernels = kernels
        self.padding = padding
        self.padding_mode = padding_mode
        self.strides = strides
        self.device = device
        self.seed = seed

        seed_everything(self.seed)
        self.conv_block = layers_CNN(num_layers, int(channels[0]), int(channels[1]), 
                                      kernels, strides, padding, padding_mode).to(device)


    def forward(self, x):
        return self.conv_block(x)


class FCBlock(nn.Module):
    def __init__(self, num_layers, input_dim, 
                 middle_dim, output_dim, dropout, device, seed = 402):

        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim # = backcast_length x channels[1]
        self.middle_dim = middle_dim # 지정
        self.output_dim = output_dim # thetas_dim[0이나 1]
        self.dropout = dropout
        self.device = device
        self.seed = seed
        seed_everything(self.seed)

        self.fc_layers = layers_FC(num_layers, input_dim, middle_dim, 
                                   output_dim, dropout).to(device)
        

    def forward(self, x):
        return self.fc_layers(x)

# ---------------------------------------------------------------------------- #
# MODEL

class CBeatsNet(nn.Module):
    def __init__(self,
                cast_lengths = (20,5), # B/F
                
                block_num = (2,2), # T/S
                thetas_dim = (3,7), # T/S
                num_layers_CNN = (1,1), # T/S
                num_layers_FC = (1,1), # T/S
                
                
                kernels_trend = (3,3),  
                kernels_season = (3,3),
                channels_trend = (32,32),
                channels_season = (32,32),
                padding = 1,
                padding_mode = 'zeros',
                strides = 1,
                
                dropout = 0.2,
                middle_dim = 64,
                weight_sharing = True,
                return_decomp = True,
                return_theta = True,
                device = torch.device('cuda'),
                seed = 402, **kwargs):

        super(CBeatsNet, self).__init__()

        assert(len(num_layers_CNN)==len(num_layers_FC)==
               len(block_num)==len(thetas_dim))
        assert(len(kernels_trend)==len(channels_trend)==block_num[0])
        assert(len(kernels_season)==len(channels_season)==block_num[1])
        assert padding_mode in ['zeros', 'reflect', 'replicate']
    
        # 1) input size
        self.backcast_length = cast_lengths[0]
        self.forecast_length = cast_lengths[1]

        # 2) architecture (overall)
        self.block_num = block_num
        self.thetas_dim = thetas_dim
        self.num_layers_CNN = num_layers_CNN
        self.num_layers_FC = num_layers_FC
                
        # 3) architecture of CNN
        self.kernels_trend = kernels_trend
        self.kernels_season = kernels_season
        self.channels_trend = (1,*channels_trend)
        self.channels_season = (1,*channels_season)
        self.padding = padding
        self.padding_mode = padding_mode
        self.strides = strides

        # 4) architecture of FC
        self.dropout = dropout
        self.middle_dim = middle_dim
        self.weight_sharing = weight_sharing
        
        
        # 5) etc
        self.return_decomp = return_decomp
        self.return_theta = return_theta
        self.device = device
        self.seed = seed

        # 5) stack
        self.trend_stack = TrendStack(self.backcast_length, 
                                      self.forecast_length,
                                      thetas_dim[0],
                                      kernels_trend,(1,*channels_trend),
                                      padding, padding_mode, strides,
                                      middle_dim,dropout,
                                      block_num[0], num_layers_CNN[0], 
                                      num_layers_FC[0],
                                      weight_sharing,
                                      device, seed)
        self.season_stack = SeasonStack(self.backcast_length, 
                                        self.forecast_length,
                                        thetas_dim[1],
                                        kernels_season,(1,*channels_season),
                                        padding, padding_mode, strides,
                                        middle_dim,dropout,
                                        block_num[1], num_layers_CNN[1], 
                                        num_layers_FC[1],
                                        weight_sharing,
                                        device, seed)
        self.parameters = []
        self.parameters.extend(self.trend_stack.parameters())
        self.parameters.extend(self.season_stack.parameters())
        self.parameters = nn.ParameterList(self.parameters)
        
        self.to(self.device)
        seed_everything(self.seed)


    def forward(self, x):
        res_decomp = []

        ## Trend
        #b_TREND, f_TREND, theta_back_TREND, theta_fore_TREND = self.trend_stack(x.to(self.device))
        
        out = self.trend_stack(x.to(self.device))
        b_TREND, f_TREND, theta_back_TREND, theta_fore_TREND = out
        res_decomp.append(f_TREND)
        
        ## Seasonality
        b_SEASON, f_SEASON, theta_back_SEASON, theta_fore_SEASON = self.season_stack(b_TREND)
        #b_SEASON, f_SEASON, theta_back_SEASON, theta_fore_SEASON = out
        res_decomp.append(f_SEASON)

        forecast = f_TREND + f_SEASON
        backcast = b_TREND + b_SEASON
        
        theta_list = [theta_back_TREND, theta_fore_TREND,
                      theta_back_SEASON, theta_fore_SEASON]
        
        if self.return_decomp & self.return_theta:
            return backcast, forecast, res_decomp, theta_list
        if self.return_decomp:
            return backcast, forecast, res_decomp
        if self.return_theta:
            return backcast, forecast, theta_list
        return backcast, forecast


# ---------------------------------------------------------------------------- #
# STACK

class Stack(nn.Module):
    def __init__(self, backcast_length, forecast_length,
                 thetas_dim,
                 kernels, channels,
                 padding, padding_mode, strides,
                 middle_dim, dropout,
                 block_num, num_layers_CNN, num_layers_FC,
                 weight_sharing,
                 device, seed = 402):
        super(Stack, self).__init__()

        #-----------------------------------------------#
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.thetas_dim = thetas_dim
        #-----------------------------------------------#
        self.kernels = kernels
        self.channels = channels
        self.padding = padding
        self.padding_mode = padding_mode
        self.strides = strides
        #-----------------------------------------------#
        self.middle_dim = middle_dim
        self.dropout=dropout
        self.block_num = block_num
        self.num_layers_CNN = num_layers_CNN
        self.num_layers_FC = num_layers_FC
        self.weight_sharing = weight_sharing
        #-----------------------------------------------#
        self.device = device
        self.seed = seed
        seed_everything(self.seed)
        #-----------------------------------------------#
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, 
                                                                      forecast_length)



class TrendStack(Stack):
    def __init__(self, backcast_length, forecast_length,
                 thetas_dim,
                 kernels, channels,
                 padding, padding_mode, strides,
                 middle_dim, dropout,
                 block_num, num_layers_CNN, num_layers_FC,
                 weight_sharing,
                 device, seed = 402):
        super(TrendStack, self).__init__(backcast_length, forecast_length,
                                         thetas_dim,
                                         kernels, channels,
                                         padding, padding_mode, strides,
                                         middle_dim, dropout,
                                         block_num, num_layers_CNN, 
                                         num_layers_FC,
                                         weight_sharing,
                                         device, seed = 402)

        seed_everything(self.seed)
        self.CONV_blocks = [ConvBlock(num_layers = num_layers_CNN, 
                                      channels = channels[i:i+2], 
                                      kernels = kernels[i], 
                                      padding = padding,
                                      strides = strides, 
                                      padding_mode = padding_mode,
                                      device = device, 
                                      seed = seed) 
                            for i in range(block_num)]

        self.FC_blocks_back = [FCBlock(num_layers = num_layers_FC, 
                                           input_dim = backcast_length * channels[i+1],
                                           middle_dim = middle_dim,
                                           output_dim = thetas_dim,
                                           dropout = dropout,
                                           device = device, 
                                           seed = seed) 
                                   for i in range(block_num)]
        
        if self.weight_sharing:
            self.FC_blocks_fore = self.FC_blocks_back
        else:
            self.FC_blocks_fore = [FCBlock(num_layers = num_layers_FC, 
                                           input_dim = backcast_length * channels[i+1],
                                           middle_dim = middle_dim,
                                           output_dim = thetas_dim,
                                           dropout = dropout,
                                           device = device, seed = seed) 
                                   for i in range(block_num)]
           
        self.parameters = []
        for conv_block in self.CONV_blocks:
            self.parameters.extend(conv_block.parameters())
        for conv_block in self.FC_blocks_back:
            self.parameters.extend(conv_block.parameters())
        for conv_block in self.FC_blocks_fore:
            self.parameters.extend(conv_block.parameters())
        self.parameters = nn.ParameterList(self.parameters)
        
    def forward(self, backcast):
        h_prev = backcast.unsqueeze(1)
        backcast = torch.tensor(h_prev).squeeze(1)
        forecast = torch.zeros((backcast.size(0), self.forecast_length,))
        backcast_cumsum = 0
        forecast_cumsum = 0

        theta_back_lst = []
        theta_fore_lst = []
    
        for block_idx in range(self.block_num):
            h = self.CONV_blocks[block_idx](h_prev)
            h_flatten = h.flatten(1)
            theta_back_pred = self.FC_blocks_back[block_idx](h_flatten)
            theta_fore_pred = self.FC_blocks_fore[block_idx](h_flatten)

            theta_back_lst.append(theta_back_pred)
            theta_fore_lst.append(theta_fore_pred)
            
            back_pred = trend_model(theta_back_pred, self.backcast_linspace, 
                                    self.device)
            fore_pred = trend_model(theta_fore_pred, self.forecast_linspace, 
                                    self.device)
            backcast_cumsum += back_pred
            forecast_cumsum += fore_pred
            h_prev = h_prev + h
        backcast = backcast.to(self.device) - backcast_cumsum
        forecast = forecast.to(self.device) + forecast_cumsum

        return backcast, forecast, theta_back_lst, theta_fore_lst


class SeasonStack(Stack):
    def __init__(self, backcast_length, forecast_length,
                 thetas_dim,
                 kernels, channels,
                 padding, padding_mode, strides,
                 middle_dim, dropout,
                 block_num, num_layers_CNN, num_layers_FC,
                 weight_sharing,
                 device, seed = 402):
        super(SeasonStack, self).__init__(backcast_length, forecast_length,
                                          thetas_dim,
                                          kernels, channels,
                                          padding, padding_mode, strides,
                                          middle_dim, dropout,
                                          block_num, num_layers_CNN, 
                                          num_layers_FC,
                                          weight_sharing,
                                          device, seed = 402)

        seed_everything(self.seed)
        self.CONV_blocks = [ConvBlock(num_layers = num_layers_CNN, 
                                      channels=channels[i:i+2], 
                                      kernels = kernels[i], 
                                      padding = padding, 
                                      strides = strides, 
                                      padding_mode = padding_mode, 
                                      device = device, 
                                      seed = seed) 
                            for i in range(block_num)]

        self.FC_blocks_back = [FCBlock(num_layers = num_layers_FC, 
                                           input_dim = backcast_length * channels[i+1],
                                           middle_dim = middle_dim,
                                           output_dim = thetas_dim,
                                           dropout = dropout,
                                           device = device, seed = seed) 
                                   for i in range(block_num)]
        
        if self.weight_sharing:
            self.FC_blocks_fore = self.FC_blocks_back
        else:
            self.FC_blocks_fore = [FCBlock(num_layers = num_layers_FC, 
                                           input_dim = backcast_length * channels[i+1],
                                           middle_dim = middle_dim,
                                           output_dim = thetas_dim,
                                           dropout = dropout,
                                           device = device, seed = seed) 
                                   for i in range(block_num)]
           
        self.parameters = []
        for conv_block in self.CONV_blocks:
            self.parameters.extend(conv_block.parameters())
        for conv_block in self.FC_blocks_back:
            self.parameters.extend(conv_block.parameters())
        self.parameters = nn.ParameterList(self.parameters)
        
    def forward(self, backcast):
        h_prev = backcast.unsqueeze(1)
        backcast = torch.tensor(h_prev).squeeze(1)
        forecast = torch.zeros((backcast.size(0), self.forecast_length,))
        backcast_cumsum = 0
        forecast_cumsum = 0
        
        theta_back_lst = []
        theta_fore_lst = []
        
        for block_idx in range(self.block_num):
            h = self.CONV_blocks[block_idx](h_prev)
            h_flatten = h.flatten(1)
            theta_back_pred = self.FC_blocks_back[block_idx](h_flatten)
            theta_fore_pred = self.FC_blocks_fore[block_idx](h_flatten)
            
            theta_back_lst.append(theta_back_pred)
            theta_fore_lst.append(theta_fore_pred)
            
            back_pred = seasonality_model(theta_back_pred, 
                                          self.backcast_linspace, self.device)
            fore_pred = seasonality_model(theta_fore_pred, 
                                          self.forecast_linspace, self.device)
            backcast_cumsum += back_pred
            forecast_cumsum += fore_pred
            h_prev = h_prev + h
        backcast = backcast.to(self.device) - backcast_cumsum
        forecast = forecast.to(self.device) + forecast_cumsum
        return backcast, forecast, theta_back_lst, theta_fore_lst
        
