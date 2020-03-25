import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import json

from model import build_model
from utils import index_data, convert_tensor

SS_mat = pd.read_pickle('../data/structural_similarity_matrix.pkl')
TS_mat = pd.read_pickle('../data/target_similarity_matrix.pkl')
GS_mat = pd.read_pickle('../data/GO_similarity_matrix.pkl')

mlb, _, idx2label, drugPair2effectIdx = index_data()
pd.to_pickle(mlb, '../data/mlb.pkl')
pd.to_pickle(idx2label, '../data/idx2label.pkl')
    
x_idx = []
y_idx = []
for k, v in drugPair2effectIdx.items():
    x_idx.append(k)
    y_idx.append(v)
x_idx, y_idx = np.array(x_idx), np.array(y_idx)

with open('../data/hyperparameter.json') as fp:
    hparam = json.load(fp)

kf = RepeatedStratifiedKFold(n_splits=hparam['n_splits'], n_repeats=hparam['n_repeats'], random_state=2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, (train_idx, test_idx) in enumerate(kf.split(x_idx, y_idx)):    
    x_train = x_idx[train_idx]
    y_train = y_idx[train_idx]    
    
    SS, TS, GS, y = convert_tensor(x_train, y_train, SS_mat, TS_mat, GS_mat, mlb, idx2label)
    dataset = torch.utils.data.TensorDataset(SS, TS, GS, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = build_model(hparam)
    model.to(device)
    model.fit(dataloader, i)
        
    x_test = x_idx[test_idx]
    y_test = y_idx[test_idx]
    pd.to_pickle([x_test, y_test], model.path+'test_data.pkl')
    del x_test, y_test
    