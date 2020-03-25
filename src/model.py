import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from math import ceil

class build_model(nn.Module):
    def __init__(self, hyperparameter):
        super(build_model, self).__init__()       
        
        input_size = hyperparameter['input_size']
        output_size = hyperparameter['output_size']
        code_size = hyperparameter['code_size']
        AE_lr = hyperparameter['AE_lr']
        DNN_lr = hyperparameter['DNN_lr']
        drop_rate = hyperparameter['drop_rate']
        
        self.epoch = hyperparameter['epoch']
        self.n_repeats = hyperparameter['n_repeats']
        self.n_splits = hyperparameter['n_splits']
        self.save_path = hyperparameter['save_path']
        self.patience = hyperparameter['patience']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder1 = self.build_encoder(input_size, code_size, drop_rate)
        self.decoder1 = self.build_decoder(input_size, code_size)
        AE_params1 = list(self.encoder1.parameters()) + list(self.decoder1.parameters())
        self.AE_opt1 = torch.optim.RMSprop(AE_params1, lr=AE_lr)
        
        self.encoder2 = self.build_encoder(input_size, code_size, drop_rate)
        self.decoder2 = self.build_decoder(input_size, code_size)
        AE_params2 = list(self.encoder2.parameters()) + list(self.decoder2.parameters())
        self.AE_opt2 = torch.optim.RMSprop(AE_params2, lr=AE_lr)
        
        self.encoder3 = self.build_encoder(input_size, code_size, drop_rate)
        self.decoder3 = self.build_decoder(input_size, code_size)
        AE_params3 = list(self.encoder3.parameters()) + list(self.decoder3.parameters())
        self.AE_opt3 = torch.optim.RMSprop(AE_params3, lr=AE_lr)
        
        self.AE_criterion = nn.MSELoss()        
        
        self.DNN = self.build_DNN(code_size*3, output_size, drop_rate)
        DNN_params = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.encoder3.parameters()) + list(self.DNN.parameters())        
        self.DNN_opt = torch.optim.Adam(DNN_params, lr=DNN_lr)
        
        self.DNN_criterion = nn.BCEWithLogitsLoss()
        
    
    def forward(self, x1, x2, x3):
        x1 = self.encoder1(x1)
        x1_de = self.decoder1(x1)
        
        x2 = self.encoder2(x2)
        x2_de = self.decoder2(x2)
        
        x3 = self.encoder3(x3)
        x3_de = self.decoder3(x3)
        
        x_dnn = torch.cat((x1, x2, x3), 1)
        pred = self.DNN(x_dnn)
        
        return x1_de, x2_de, x3_de, pred
    
    def fit(self, dataloader, repeat):        
        n = ceil(len(dataloader.dataset)/dataloader.batch_size)
        loss_per_epoch = [] # ['DNN', 'SSP', 'TSP', 'GSP']        
        previous = -1
        for i in range(1, self.epoch+1):
            dnn, ae1, ae2, ae3, j = 0, 0, 0, 0, 1
            
            for x1, x2, x3, y in dataloader:
                x1, x2, x3, y = x1.to(self.device), x2.to(self.device), x3.to(self.device), y.to(self.device)
                o1, o2, o3, pred = self(x1, x2, x3)
                
                self.AE_opt1.zero_grad()
                AE_loss1 = self.AE_criterion(o1, x1) 
                AE_loss1.backward(retain_graph=True)
                self.AE_opt1.step()
                
                self.AE_opt2.zero_grad()
                AE_loss2 = self.AE_criterion(o2, x2)
                AE_loss2.backward(retain_graph=True)
                self.AE_opt2.step()
                
                self.AE_opt3.zero_grad()
                AE_loss3 = self.AE_criterion(o3, x3)
                AE_loss3.backward(retain_graph=True)
                self.AE_opt3.step()
                
                self.DNN_opt.zero_grad()
                DNN_loss = self.DNN_criterion(pred, y)
                DNN_loss.backward()
                self.DNN_opt.step()
                
                tmp_loss = list(map(lambda x: round(float(x), 6), [DNN_loss, AE_loss1, AE_loss2, AE_loss3]))
                dnn += tmp_loss[0]
                ae1 += tmp_loss[1]
                ae2 += tmp_loss[2]
                ae3 += tmp_loss[3]
                
                if j % 50 == 0:
                    print(f'Repeat {repeat+1}/{self.n_repeats*self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss:  DNN {dnn/j:.6f}  SSP {ae1/j:.6f}  TSP {ae2/j:.6f}  GSP {ae3/j:.6f}')
                    print()
                    
                j += 1
                
            j -= 1
            print(f'Repeat {repeat+1}/{self.n_repeats*self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss:  DNN {dnn/j:.6f}  SSP {ae1/j:.6f}  TSP {ae2/j:.6f}  GSP {ae3/j:.6f}')
            print()
            loss_per_epoch.append(list(map(lambda x: x/(j), [dnn, ae1, ae2, ae3])))            
            
            
            if len(loss_per_epoch) > self.patience:
                sum_loss = np.sum(np.array(loss_per_epoch), 1)
                current = np.argmin(sum_loss)
                
                if previous != current:
                    previous = current
                    self.save_model(repeat, loss_per_epoch)
                    
                elif (previous == current) and (previous + self.patience == len(loss_per_epoch)):
                    print('===================Early Stopping===================')
                    break
                                            
    
    def build_encoder(self, input_size, code_size, drop_rate):        
        encoder = nn.Sequential(            
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),            
            nn.Linear(1000, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True))
        
        return encoder
    
    def build_decoder(self, input_size, code_size):
        decoder = nn.Sequential(
            nn.Linear(code_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),            
            nn.Linear(1000, input_size),            
            nn.Sigmoid())
        
        return decoder
    
    def build_DNN(self, input_size, output_size, drop_rate):
        DNN = nn.Sequential(            
            nn.Linear(input_size, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),            
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(2000, output_size))
        
        return DNN
    
    
    def save_model(self, repeat, loss_per_epoch):
        self.path = self.save_path + str(repeat) + '/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        torch.save(self.state_dict(), self.path+'model_checkpoint')
        pd.to_pickle(loss_per_epoch, self.path+'loss_per_epoch.pkl')
    
    def load_model(self, path):
        weights = torch.load(path, map_location=self.device)
        self.load_state_dict(weights)        