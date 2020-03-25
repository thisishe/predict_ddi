import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score, precision_score, accuracy_score

def index_data():    
    drugPair2effect = pd.read_pickle('../data/drugPair2effect_idx.pkl')
    y_all = list(drugPair2effect.values())
    
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(y_all)
    
    labels = sorted(list(set(y_all)))
    
    label2idx = {}
    for i, j in enumerate(labels):
        label2idx[j] = i
        
    drugPair2effectIdx = {}
    for k, v in drugPair2effect.items():
        drugPair2effectIdx[k] = label2idx[v]
        
    idx2label = np.zeros(len(label2idx), dtype='O')
    for k, v in label2idx.items():
        idx2label[v] = np.array(k)
    
    return mlb, label2idx, idx2label, drugPair2effectIdx


def convert_tensor(x_idx, y_idx, SS_mat, TS_mat, GS_mat, mlb, idx2label):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SS = torch.tensor(SS_mat[x_idx].reshape(len(x_idx), len(SS_mat)*2)).float()
    TS = torch.tensor(TS_mat[x_idx].reshape(len(x_idx), len(TS_mat)*2)).float()
    GS = torch.tensor(GS_mat[x_idx].reshape(len(x_idx), len(GS_mat)*2)).float()
    y = torch.tensor(mlb.transform(idx2label[y_idx])).float()
    
    return SS, TS, GS, y

def evaluate_model(answer, prediction):
    accuracy = accuracy_score(answer, prediction)
    macro_recall = recall_score(answer, prediction, average='macro')
    macro_precision = precision_score(answer, prediction, average='macro')
    micro_recall = recall_score(answer, prediction, average='micro')
    micro_precision = precision_score(answer, prediction, average='micro')
    
    return accuracy, macro_recall, macro_precision, micro_recall, micro_precision
