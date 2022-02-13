from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import PIL
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch 
import io

# ---------------------------------------------------------------------------- #
# UTILS

def trend_theta(backcast, theta_dim, window = None):
    def ma(data, start, window):
        return(np.mean(data[start : start + window]))
    t = np.arange(len(backcast))
    if window:
        n = len(backcast)
        MA_list = [ma(backcast, i, window) for i in range(n-window)]
        y = np.concatenate((np.array(backcast[:window]), np.array(MA_list)))
    else:
        y = np.array(backcast)
    theta_list = np.polyfit(t, y, theta_dim-1) # …3차,2차,1차,상수
    theta_list = theta_list[::-1]  # 상수,1차,2차,3차 …
    return theta_list



def tb_writer(prefix, tb_dir):
    cur_time = time.strftime("%m%d_%H%M%S", time.localtime()) 
    prefix = prefix + cur_time
    
    return SummaryWriter(tb_dir + prefix)


def generate_plot(model, loader, view_len = 50):
    gt = []
    pred = []
    trend = []
    season = []
    model.eval()
    with torch.no_grad():
        
        for x_total, y_total in loader:
            _, forecast, stack_res_tr, _ = model(x_total.to(model.device))

            pred.append(forecast.detach().cpu().numpy())
            gt.append(y_total.numpy())           
            trend_res = stack_res_tr[0].detach().cpu().numpy()
            season_res = stack_res_tr[1].detach().cpu().numpy() - trend_res
            trend.append(trend_res)
            season.append(season_res)
    
    
    gt = np.vstack(gt).flatten()
    pred = np.vstack(pred).flatten()
    trend = np.vstack(trend).flatten()
    season = np.vstack(season).flatten()
    
    
    beg, mid, end = 100, len(gt)//2, len(gt) - view_len

    f, axs = plt.subplots(3,1, figsize = (25, 15))

    for idx, ts_id in enumerate([beg, mid, end]):
        plt.subplot(3,1,idx + 1)
        #plt.figure(figsize = (20, 5))
        plt.plot(gt[ts_id:ts_id + view_len ], label = 'gt')
        plt.plot(pred[ts_id:ts_id + view_len], label = 'pred')
        plt.plot( trend[ts_id:ts_id + view_len], label = 'trend')
        plt.plot( season[ts_id:ts_id + view_len], label = 'season' )
        plt.title(f'TSID:{ts_id}~{ts_id + view_len}')
        plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()
    return buf