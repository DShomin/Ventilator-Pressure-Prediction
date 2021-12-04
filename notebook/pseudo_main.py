#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# In this notebook, I'm trying to integrate the public notebook for Ventillator Pressure Competition written in Pytorch to Fastai. The reason is to leverage high level API of fastai to avoid repetitive pattern ( for example fititing with a scheduler learning rate, adding some callback  like ReduceLROnPlateau )
# 


import numpy as np # linear algebra
import pandas as pd
from torch._C import device # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from fastai.torch_core import default_device
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.progress import ProgressCallback
from fastai.optimizer import OptimWrapper
from torch import optim
from fastai.losses import MSELossFlat, L1LossFlat, LabelSmoothingCrossEntropyFlat, CrossEntropyLossFlat
from fastai.metrics import accuracy_multi, AccumMetric
from fastai.callback.schedule import Learner
from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import IndexSplitter
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import KFold
from tqdm import tqdm
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import argparse
import os
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class VentilatorDataset(Dataset):
    def __init__(self, data, target, idx):
        self.data = torch.from_numpy(data).float()
        if target is not None:
            self.targets = torch.from_numpy(target).float() * torch.from_numpy(idx)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'targets'): return self.data[idx], self.targets[idx]
        else: return self.data[idx]

class VentilatorCEDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float()
        if target is not None:
            self.t_dict = self.target2dict(target)
            mapped_target = np.vectorize(self.t_dict.get)(target)
            self.targets = torch.from_numpy(mapped_target).long()
            
    def target2dict(self, target):
        uni = np.unique(target)
        target_dict = dict()
        for i, val in enumerate(uni):
            target_dict[val] = i
        return target_dict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'targets'): return self.data[idx], self.targets[idx]
        else: return self.data[idx]

class PressDataset(Dataset):
    def __init__(self, x, y, uout_zero_flag=None, num_cutout=0, cutout_ratio=0.3, cutout_size=5):
        self.input = torch.from_numpy(x).float()
        self.label = torch.from_numpy(y).float()
        self.cutout_ratio = cutout_ratio
        self.cutout_size = cutout_size
        self.nun_cutout = num_cutout
        # self.uout_zero_flag = uout_zero_flag

    def __len__(self):
        return len(self.input)

    def erasing_data(self, x, cutout_size=5):
        f_ = random.randint(0, len(x))
        t_ = f_ + cutout_size
        x[f_:t_, :] = 0
        return x

    def __getitem__(self, index):
        
        data = dict()
        x = self.input[index]
        if random.random() < self.cutout_ratio:
            for _ in range(random.randint(1, self.nun_cutout)):
                x = self.erasing_data(x, self.cutout_size)
        data['input'] = x
        # data['zero_uout_flag'] = self.uout_zero_flag[index]
        data['label'] = self.label[index]
        return data['input'], data['label']



class RNNModel(nn.Module):
    def __init__(self, input_size=72):
        hidden = [400, 300, 200, 100]
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden[0],
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden[0], hidden[1],
                             batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * hidden[1], hidden[2],
                             batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2 * hidden[2], hidden[3],
                             batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden[3], 50)
        self.selu = nn.SELU()
        self.fc2 = nn.Linear(50, 1)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)

        return x



# train = np.load('../FE/train_f.npy')
train = np.load('../FE/train_v2.npy')
train_idx = np.load('../FE/uout_train.npy')

pseudo = np.load('../FE/pseudo.npy')
pseudo_y = np.load('../FE/pseudo_y.npy').reshape(-1, 80)
pseudo_idx = np.load('../FE/u_out_index.npy').astype(bool).reshape(-1, 80)                                                                                                                       

targets = np.load('../FE/y_train.npy')
# test = np.load('../FE/test_f.npy')
test = np.load('../FE/test_v2.npy')




batch_size = 512
submission = pd.read_csv('../data/sample_submission.csv')
test_dataset = VentilatorDataset(test, None, None)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


########################## Experimenting with one fold


## device
# 명령행에서 받을 키워드 인자를 설정합니다.
parser = argparse.ArgumentParser("")
parser.add_argument("--c_fold", type=str, default='')
parser.add_argument("--device", type=str, default='')
args = parser.parse_args()

# default_device()
torch.cuda.set_device(f'cuda:{args.device}')
c_fold= int(args.c_fold)

kf = KFold(n_splits=10, shuffle=True, random_state=32)

# train_index=list(range(int(0.8*len(train)))) ## Change to have reasonable train/valid dataset
# valid_index=list(range(int(0.2*len(train)), len(train)))

for i, (train_index, valid_index) in enumerate(kf.split(train)):
    if i == c_fold:
        train_input, valid_input = train[train_index], train[valid_index]
        train_idx, valid_idx = train_idx[train_index], train_idx[valid_index]
        train_targets, valid_targets = targets[train_index], targets[valid_index]

train_input = np.concatenate((train_input, pseudo))
train_targets = np.concatenate((train_targets, pseudo_y))
train_idx = np.concatenate((train_idx, pseudo_idx))

train_dataset = VentilatorDataset(train_input, train_targets, train_idx)
valid_dataset = VentilatorDataset(valid_input, valid_targets, valid_idx)


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)

dls = DataLoaders(train_loader, valid_loader)
model = RNNModel() # RNNModel()


learn = Learner(dls, model, loss_func=L1LossFlat()) # CrossEntropyLossFlat()

df_test = pd.read_csv('../data/test.csv')


learn.fit_one_cycle(235, lr_max=3e-3, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10))
# 235
checkpoint_filepath = f'fold_pse_{c_fold}_10.hdf5'
torch.save(model.state_dict(), checkpoint_filepath)

# preds_fold.append(preds)
# df_test[['id', 'pressure']].to_csv('submission.csv', index=False)



preds = []
with torch.no_grad():
    for data in tqdm(test_loader):
        pred = model(data.to('cuda'))
        pred = pred.squeeze(-1).flatten()
        preds.extend(pred.detach().cpu().numpy())

preds = np.array(preds)
df_test['pressure'] = preds


df_test[['id', 'pressure']].to_csv(f'submission_pse_fold_{c_fold}_10.csv', index=False)



########################################################################## Uncomment code below KFold Prediction



# kf = KFold(n_splits=5, shuffle=True)
# preds_fold = []
        
# for fold, (train_index, valid_index) in enumerate(kf.split(idx)):
#     preds = []
#     model = RNNModel().to('cuda')
#     print("FOLD:", fold)
#     print(train_index)
#     print(valid_index)

#     train_input, valid_input = train[train_index], train[valid_index]
#     train_targets, valid_targets = targets[train_index], targets[valid_index]

#     train_dataset = VentilatorDataset(train_input, train_targets)
#     valid_dataset = VentilatorDataset(valid_input, valid_targets)
    
#     train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)
    
#     dls = DataLoaders(train_loader, valid_loader)
#     learn = Learner(dls, model, loss_func=MSELossFlat())
#     learn.fit_one_cycle(1, lr_max=2e-3)
    
#     with torch.no_grad():
#         for data in test_loader:
#             pred = model(data.to('cuda')).squeeze(-1).flatten()
#             preds.extend(pred.detach().cpu().numpy())
#     preds_fold.append(preds)


# preds_fold = np.array(preds_fold)
# df_test['pressure'] = np.median(preds_fold, axis=0)
# df_test[['id', 'pressure']].to_csv('submission.csv', index=False)

