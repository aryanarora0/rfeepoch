import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from model import MLP
from training import train_epoch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

def custom_rfe(x, y, num_features, batch_size = 32):
    sel_features = list(range(x.shape[1]))
    elim_features = [] 

    while len(sel_features) > num_features:
        x_reduced = x[:, sel_features]
        train_dataset = TensorDataset(x_reduced, y)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        
        model = MLP(len(sel_features))
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        loss_log = []
        train_epoch(model, train_loader, criterion, optimizer, loss_log)
        
        #feature importance using absolute weights of the first layer, sum over output neurons
        with torch.no_grad():
            weights = model.layer1.weight.data.abs()
            feature_importance = torch.sum(weights, dim=0)
        
        min_idx = torch.argmin(feature_importance).item()
        elim_feature = sel_features.pop(min_idx)
        elim_features.append(elim_feature)
        print(f'Eliminated feature {elim_feature}. Importance: {feature_importance[min_idx].item()}\n')
    
    return sel_features, elim_features
