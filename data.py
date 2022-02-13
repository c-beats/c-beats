import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
 
# ---------------------------------------------------------------------------- #
# SPLIT DATA

def train_val_test_split(lst, train_ratio):

    train_len = int(len(lst)*train_ratio)
    val_len = (len(lst)-train_len)//2
    test_len = len(lst)-train_len-val_len
    
    train_dat = lst[:train_len]
    val_dat = lst[train_len:train_len+val_len]
    test_dat = lst[-test_len:]
    return train_dat, val_dat, test_dat


def x_y_split(lst, back_len, forward_len, stride):
    X = []
    y = []
    for i in np.arange(0, len(lst)-back_len, stride):
        if len(lst[(i+back_len):(i+back_len+forward_len)])==forward_len:
            X.append(np.asarray(lst[i:(i+back_len)]))
            y.append(np.asarray(lst[(i+back_len):(i+back_len+forward_len)]))
    return X,y


def train_val_test_Xy_split(data, train_ratio, backcast_length, 
                            forecast_length, stride):
    train_dat, val_dat, test_dat = train_val_test_split(data, train_ratio)
    train_X, train_y = x_y_split(train_dat, backcast_length, forecast_length,stride)
    val_X, val_y = x_y_split(val_dat, backcast_length, forecast_length,stride)
    test_X, test_y = x_y_split(test_dat, backcast_length, forecast_length,stride)
    return ( (train_X , train_y), 
             (val_X , val_y), 
             (test_X , test_y))





# ---------------------------------------------------------------------------- #
# SCALING

def scaling(lst,train_ratio,type='minmax'):

    assert type in ['None', 'minmax','z']
    train_len = int(len(lst)*train_ratio)
    train_dat = lst[:train_len]

    if type=='minmax':
        train_max = np.max(train_dat)
        train_min = np.min(train_dat)

        return (lst - train_min )/ (train_max - train_min)
    elif type == 'z':
        train_mean=np.mean(train_dat)
        train_std=np.std(train_dat)
        return (lst-train_mean)/train_std

    else:
        return lst
    
# ---------------------------------------------------------------------------- #
# DATASET

class TotalDataset(Dataset):
    def __init__(self, X , y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return torch.tensor(X).float(), torch.tensor(y).float()


# ---------------------------------------------------------------------------- #
# LOADER

def initialize_loader(data, train_ratio = 0.6, forecast_length = 5, 
                      backcast_length = 20, global_backcast = 40, stride = 5, 
                      scale_type = 'minmax',batch_size = 16,  
                      check = False, **kwargs):
    
    # Do some scaling
    data_scaled = scaling(data, train_ratio, type=scale_type)


            
    # Split data
    train_dat, val_dat, test_dat =  train_val_test_Xy_split(data_scaled, 
                                                    train_ratio, backcast_length, 
                                                    forecast_length, stride)

    total_dat, _, _ = train_val_test_Xy_split(data_scaled, 
                                              1.0, backcast_length, 
                                              forecast_length, stride)
    # unpack tuple
    train_X, train_y = train_dat
    val_X, val_y = val_dat
    test_X , test_y = test_dat

    total_X, total_y = total_dat

    # initialize seed & loader
    #np.random.seed(seed_num) shuffle 안 하므로;ㅣ두

    train_dataset = TotalDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size, drop_last=True,
                              shuffle=True)

    val_dataset = TotalDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    test_dataset = TotalDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    total_dataset = TotalDataset(total_X, total_y)
    total_loader = DataLoader(total_dataset, batch_size, shuffle=False)

    # CHECK
    if check == True:
        batch1 = next(iter(train_loader))
        #batch2 = next(iter(val_loader))
        #batch3 = next(iter(test_loader))
        #batch4 = next(iter(total_loader))

        print('Tr X data Shape:', batch1[0].shape)
        print('Tr y data Shape:', batch1[1].shape)
        #print('Val X data Shape:', batch2[0].shape)
        #print('Val y data Shape:', batch2[1].shape)
        #print('Test X data Shape:', batch3[0].shape)
        #print('Test y data Shape:', batch3[1].shape)
        #print('Tot X data Shape:', batch4[0].shape)
        #print('Tot y data Shape:', batch4[1].shape)

    return {'tr': train_loader, 'val': val_loader, 
            'test': test_loader, 'tot': total_loader}

