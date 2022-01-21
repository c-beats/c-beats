import time
import os
from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from data import * 
from experiment_module import * 

# ---------------------------------------------------------------------------- #
# HYPERPARAMETER TUNING

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', **kwargs):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



# ---------------------------------------------------------------------------- #
# TRAIN & EVAL

def get_optimizer(lr, parameters):
    return optim.Adam(lr = lr, params = parameters )


class Trainer:
    def __init__(self, model, config):
        ''''''
        self.config = EasyDict(config)
        self.loaders = initialize_loader(data = np.loadtxt(config['data_dir']), 
                                          **config)
        self.model = model
        self.optimizer = get_optimizer(self.config.lr, self.model.parameters() )
        self.loss = self.get_loss(config)

        cur_time = time.strftime("%m%d_%H%M%S", time.localtime()) 
        self.prefix = config['prefix'] + cur_time
        self.make_dir()

    def get_loss(self, config):
        # 추후에 다른 objective 도 추가...
        if config['loss_type'] == 'mse':
            return torch.nn.MSELoss()

    def make_dir(self):
        ''''''
        if not os.path.exists('./model_weights'):
            os.mkdir('./model_weights')
        path_ = './model_weights/' + self.prefix
        os.mkdir(path_)
        self.save_path = path_ + './checkpoint.pt'


    def train_one_epoch(self):
        ''''''
        tr_loss_lst = []
        self.model.train()
        
        beg, med, end = 1, len(self.loaders['tr']) // 2, len(self.loaders['tr'])
        back_diff_lst, fore_diff_lst, back_diff_avg_lst, fore_diff_avg_lst = [], [], [], []

        for batch_idx, (x_tr, y_tr) in enumerate(self.loaders['tr']):
            
            x_tr_, y_tr_ = x_tr.to(self.config.device), y_tr.to(self.config.device)
            
            self.optimizer.zero_grad()

            if self.config.return_decomp & self.config.return_decomp:
                backcast, forecast, res_decomp, theta_list = self.model(x_tr_)

                if batch_idx in [beg, med, end -1]:
                    theta_pred_b = np.array([ block[0][0].detach().cpu().numpy() for block 
                                      in theta_list[:self.config.num_layers_FC[0]]]).sum(axis = 0)
                    theta_pred_f = np.array([ block[1][0].detach().cpu().numpy() for block 
                                      in theta_list[:self.config.num_layers_FC[0]]]).sum(axis = 0)

                    theta_true_b_ma3 = trend_theta( x_tr[0].numpy(), 
                                                    self.config.thetas_dim[0], window = 3)
                    theta_true_f_ma3 = trend_theta( y_tr[0].numpy(), 
                                                    self.config.thetas_dim[0], window = 3)

                    back_diff_ma3 = abs(theta_pred_b - theta_true_b_ma3)
                    fore_diff_ma3 = abs(theta_pred_f - theta_true_f_ma3)
                    avg_diff_b = np.mean(back_diff_ma3)
                    avg_diff_f = np.mean(fore_diff_ma3)

                    back_diff_lst.append(back_diff_ma3)
                    fore_diff_lst.append(fore_diff_ma3)
                    back_diff_avg_lst.append(avg_diff_b)
                    fore_diff_avg_lst.append(avg_diff_f)

            ## 모델에서 걍 다 리턴하게 하고 else 부분 날려도 됨 
            else:
                if self.config.return_decomp:
                    backcast, forecast, res_decomp = self.model(x_tr_)
                elif self.config.return_theta:
                    backcast, forecast, theta_list = self.model(x_tr_)
                else:
                    backcast, forecast = self.model(x_tr_)

            loss_back = self.loss(backcast.view_as(x_tr), 
                             torch.zeros_like(x_tr).to(self.config.device))
            loss_fore = self.loss(forecast.view_as(y_tr), y_tr_)
            total_loss = self.config.alpha*loss_fore + (1-self.config.alpha)*loss_back
            tr_loss_lst.append(total_loss.item())

            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            total_loss.backward()
            self.optimizer.step()

        #print(total_loss)
        return np.mean(tr_loss_lst), back_diff_lst, fore_diff_lst, back_diff_avg_lst, fore_diff_avg_lst

    def evaluate(self, mode = 'validate'):
        ''''''
        val_loss_lst = []
        self.model.eval()
        if mode == 'validate':
            loader = self.loaders['val']
            beg, med, end = 1, len(loader) // 2, len(loader)
        else:
            loader = self.loaders['test']
            beg, med, end = 1, len(loader) // 2, len(loader)

        fore_diff_lst, fore_diff_avg_lst = [], []

        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(loader):
                x_val_ = x_val.to(self.config.device)
                y_val_ = y_val.to(self.config.device)

                if self.config.return_decomp & self.config.return_decomp:
                    _, forecast, res_decomp, theta_list = self.model(x_val_)

                    theta_pred_f = np.array([ block[1][0].detach().cpu().numpy() 
                                               for block 
                                               in theta_list[:self.config.num_layers_FC[0]]]).sum(axis = 0)

                    theta_true_f_ma3 = trend_theta( y_val[0].numpy(), 
                                                    self.config.thetas_dim[0], window = 3)

                    fore_diff_ma3 = abs(theta_pred_f - theta_true_f_ma3)
                    avg_diff_f = np.mean(fore_diff_ma3)

                    fore_diff_lst.append(fore_diff_ma3)
                    fore_diff_avg_lst.append(avg_diff_f)


                ## 모델에서 걍 다 리턴하게 하고 else 부분 날려도 됨 
                else:
                    if self.config.return_decomp:
                        _, forecast, res_decomp = self.model(x_val_)
                    elif self.config.return_theta:
                        _, forecast, theta_list = self.model(x_val_)
                    else:
                        _, forecast = self.model(x_val_)

                val_loss = self.loss(forecast.view_as(y_val), y_val_)
                val_loss_lst.append(val_loss.item())

        return np.mean(val_loss_lst), fore_diff_lst, fore_diff_avg_lst

    def run(self):
        ''''''
        early_stopping = EarlyStopping(path = self.save_path, **self.config)
        timer = time.time()
        writer = tb_writer(self.config.prefix, self.config.tb_dir)        

        for e in range(self.config.epochs):
            tr_loss, back_diff_lst, fore_diff_lst, \
                  back_diff_avg_lst, fore_diff_avg_lst = self.train_one_epoch()
            val_loss, fore_diff_lst_v, fore_diff_avg_lst_v = self.evaluate('validate')
            test_loss, fore_diff_lst_t, fore_diff_avg_lst_t = self.evaluate('test')

            if e % 10 == 0:
                plot_buf = generate_plot(self.model, self.loaders['tot'])
                img = PIL.Image.open(plot_buf)
                img = ToTensor()(img)
                writer.add_image(f'prediction', img, e)

            writer.add_scalar('[1]Loss/1.train', tr_loss, global_step = e)
            writer.add_scalar('[1]Loss/2.val', val_loss, global_step = e)
            writer.add_scalar('[1]Loss/3.test', test_loss, global_step = e)

            writer.add_scalar('[2](Train)Backcast/1.beg(avg)', back_diff_avg_lst[0], global_step = e)
            writer.add_scalar('[2](Train)Backcast/1.beg(theta0)', back_diff_lst[0][0], global_step = e)
            writer.add_scalar('[2](Train)Backcast/1.beg(theta1)', back_diff_lst[0][1], global_step = e)
            writer.add_scalar('[2](Train)Backcast/1.beg(theta2)', back_diff_lst[0][2], global_step = e)
            writer.add_scalar('[2](Train)Backcast/2.mid(avg)', back_diff_avg_lst[1], global_step = e)
            writer.add_scalar('[2](Train)Backcast/2.mid(theta0)', back_diff_lst[1][0], global_step = e)
            writer.add_scalar('[2](Train)Backcast/2.mid(theta1)', back_diff_lst[1][1], global_step = e)
            writer.add_scalar('[2](Train)Backcast/2.mid(theta2)', back_diff_lst[1][2], global_step = e)
            writer.add_scalar('[2](Train)Backcast/3.end(avg)', back_diff_avg_lst[2], global_step = e)
            writer.add_scalar('[2](Train)Backcast/3.end(theta0)', back_diff_lst[2][0], global_step = e)
            writer.add_scalar('[2](Train)Backcast/3.end(theta1)', back_diff_lst[2][1], global_step = e)
            writer.add_scalar('[2](Train)Backcast/3.end(theta2)', back_diff_lst[2][2], global_step = e)

            writer.add_scalar('[3](Train)Forecast/1.beg(avg)', fore_diff_avg_lst[0], global_step = e)
            writer.add_scalar('[3](Train)Forecast/1.beg(theta0)', fore_diff_lst[0][0], global_step = e)
            writer.add_scalar('[3](Train)Forecast/1.beg(theta1)', fore_diff_lst[0][1], global_step = e)
            writer.add_scalar('[3](Train)Forecast/1.beg(theta2)', fore_diff_lst[0][2], global_step = e)
            writer.add_scalar('[3](Train)Forecast/2.mid(avg)', fore_diff_avg_lst[1], global_step = e)
            writer.add_scalar('[3](Train)Forecast/2.mid(theta0)', fore_diff_lst[1][0], global_step = e)
            writer.add_scalar('[3](Train)Forecast/2.mid(theta1)', fore_diff_lst[1][1], global_step = e)
            writer.add_scalar('[3](Train)Forecast/2.mid(theta2)', fore_diff_lst[1][2], global_step = e)
            writer.add_scalar('[3](Train)Forecast/3.end(avg)', fore_diff_avg_lst[2], global_step = e)
            writer.add_scalar('[3](Train)Forecast/3.end(theta0)', fore_diff_lst[2][0], global_step = e)
            writer.add_scalar('[3](Train)Forecast/3.end(theta1)', fore_diff_lst[2][1], global_step = e)
            writer.add_scalar('[3](Train)Forecast/3.end(theta2)', fore_diff_lst[2][2], global_step = e)

            writer.add_scalar('[4](Valid)Forecast/1.beg(avg)', fore_diff_avg_lst_v[0], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/1.beg(theta0)', fore_diff_lst_v[0][0], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/1.beg(theta1)', fore_diff_lst_v[0][1], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/1.beg(theta2)', fore_diff_lst_v[0][2], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/2.mid(avg)', fore_diff_avg_lst_v[1], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/2.mid(theta0)', fore_diff_lst_v[1][0], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/2.mid(theta1)', fore_diff_lst_v[1][1], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/2.mid(theta2)', fore_diff_lst_v[1][2], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/3.end(avg)', fore_diff_avg_lst_v[2], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/3.end(theta0)', fore_diff_lst_v[2][0], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/3.end(theta1)', fore_diff_lst_v[2][1], global_step = e)
            writer.add_scalar('[4](Valid)Forecast/3.end(theta2)', fore_diff_lst_v[2][2], global_step = e)

            writer.add_scalar('[5](Test)Forecast/1.beg(avg)', fore_diff_avg_lst_t[0], global_step = e)
            writer.add_scalar('[5](Test)Forecast/1.beg(theta0)', fore_diff_lst_t[0][0], global_step = e)
            writer.add_scalar('[5](Test)Forecast/1.beg(theta1)', fore_diff_lst_t[0][1], global_step = e)
            writer.add_scalar('[5](Test)Forecast/1.beg(theta2)', fore_diff_lst_t[0][2], global_step = e)
            writer.add_scalar('[5](Test)Forecast/2.mid(avg)', fore_diff_avg_lst_t[1], global_step = e)
            writer.add_scalar('[5](Test)Forecast/2.mid(theta0)', fore_diff_lst_t[1][0], global_step = e)
            writer.add_scalar('[5](Test)Forecast/2.mid(theta1)', fore_diff_lst_t[1][1], global_step = e)
            writer.add_scalar('[5](Test)Forecast/2.mid(theta2)', fore_diff_lst_t[1][2], global_step = e)
            writer.add_scalar('[5](Test)Forecast/3.end(avg)', fore_diff_avg_lst_t[2], global_step = e)
            writer.add_scalar('[5](Test)Forecast/3.end(theta0)', fore_diff_lst_t[2][0], global_step = e)
            writer.add_scalar('[5](Test)Forecast/3.end(theta1)', fore_diff_lst_t[2][1], global_step = e)
            writer.add_scalar('[5](Test)Forecast/3.end(theta2)', fore_diff_lst_t[2][2], global_step = e)


            writer.flush()

            print(f'{e+1}/{self.config.epochs} tr_loss:{tr_loss:.5f}, val_loss:{val_loss:.5f}, test_loss:{test_loss:.5f}')
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch{e+1}')
                break
        print(f'Time taken: {time.time() - timer}')
        return self.model



# ---------------------------------------------------------------------------- #

