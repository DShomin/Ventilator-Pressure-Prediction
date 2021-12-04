from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import random
import torch

class DataRetriever_LSTM(Dataset):
    def __init__(self, df, scaler, breath_id_list, train_flag):
        self.df = df
        self.scaler = scaler
        self.breath_id_list = breath_id_list
        self.train_flag = train_flag
            
    def __len__(self):
        return len(self.breath_id_list)
    
    def __getitem__(self, index):
        breath_id = self.breath_id_list[index]

        if self.train_flag:
            formatted_data = self.df[self.df['breath_id']==breath_id].sort_values(by = ['time_step'], ascending = True)[['breath_id','R', 'C', 'time_step', 'u_in', 'u_out', 'pressure']].reset_index(drop = True)
        else:
            formatted_data = self.df[self.df['breath_id']==breath_id].sort_values(by = ['time_step'], ascending = True)[['breath_id','R', 'C', 'time_step', 'u_in', 'u_out']].reset_index(drop = True)
            formatted_data['pressure'] = 0
            
        # Scaling
        formatted_data = pd.DataFrame(self.scaler.transform(formatted_data[['R', 'C', 'time_step', 'u_in', 'u_out', 'pressure']])).reset_index(drop = True)
        formatted_data.columns = ['R', 'C', 'time_step', 'u_in', 'u_out', 'pressure']
        
        X = torch.tensor(np.stack([formatted_data['time_step'], formatted_data['R'], formatted_data['C'], formatted_data['u_in'], formatted_data['u_out']], axis = 1)).float()
        
        if (self.train_flag):
            return {"X": X, "y": torch.tensor(formatted_data['pressure']).float()}
        else:
            return {"X": X, "id": breath_id}

class VentilatorCEDataset(Dataset):
    def __init__(self, data, target, uout_zero_flag):
        self.data = torch.from_numpy(data).float()
        self.uout_zero_flag = uout_zero_flag
        if target is not None:
            self.targets = torch.from_numpy(target).float()

            t_dict = self.target2dict(target)
            mapped_target = np.vectorize(t_dict.get)(target)
            self.cv_targets = torch.from_numpy(mapped_target).float()

    def target2dict(self, target):
        uni = np.unique(target)
        target_dict = dict()
        for i, val in enumerate(uni):
            target_dict[val] = i
        return target_dict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = dict()
        data['input'] = self.data[idx]
        data['zero_uout_flag'] = self.uout_zero_flag[idx]
        if hasattr(self, 'targets'): 
            data['label'] = self.targets[idx]
            data['e_label'] = self.cv_targets[idx]

        return data

def cat2label(y):
    return y * 0.07030214545121005 - 1.895744294564641

class PressDataset(Dataset):
    def __init__(self, x, y, uout_zero_flag, num_cutout=0, cutout_ratio=0.3, cutout_size=5):
        self.input = x
        self.label = y
        self.cutout_ratio = cutout_ratio
        self.cutout_size = cutout_size
        self.nun_cutout = num_cutout
        self.uout_zero_flag = uout_zero_flag

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
        data['zero_uout_flag'] = self.uout_zero_flag[index]
        data['label'] = self.label[index]
        return data


class PressDataset2(Dataset):
    def __init__(self, x, x_u_zero, y):
        self.input = x
        self.x_u_zero = 1 - x_u_zero
        self.label = y

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        
        data = dict()
        x = self.input[index]
        data['input'] = x
        data['zero_label'] = self.x_u_zero * self.label
        data['label'] = self.label[index]
        return data

def get_min_max_scl(train_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df[['R', 'C', 'time_step', 'u_in', 'u_out', 'pressure']])
    return scaler

def get_train_valid_b_id(train_df, c_fold):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    C_FOLD = 0
    for i, (train_idx, valid_idx) in enumerate(kf.split(X=train_df.breath_id.unique())):

        if i == C_FOLD:
            train_b_id = train_df.breath_id.unique()[train_idx]
            df_train = train_df[train_df['breath_id'].isin(train_b_id)].reset_index(drop = True),

            valid_b_id = train_df.breath_id.unique()[valid_idx]
            df_valid = train_df[train_df['breath_id'].isin(valid_b_id)].reset_index(drop = True),

    return train_b_id, df_train, valid_b_id, df_valid

if __name__ == '__main__':

    x_trian = np.load('../FE/train_f.npy')#[:sample_idx]
    train_zero_flag = np.load('../FE/uout_train.npy')#[:sample_idx]
    y_train = np.load('../FE/y_train.npy')#[:sample_idx]

    # split fold
    C_FOLD = 0
    # skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf = KFold(n_splits=5, random_state=42, shuffle=True)

    for i, (train_idx, valid_idx) in enumerate(skf.split(X=x_trian)):

        TRAIN_X = x_trian[train_idx]
        TRAIN_UOUT = train_zero_flag[train_idx]
        TRAIN_Y = y_train[train_idx]


        VALID_X = x_trian[valid_idx]
        VALID_UOUT = train_zero_flag[valid_idx]
        VALID_Y = y_train[valid_idx]

    train_dataset = VentilatorCEDataset(TRAIN_X, TRAIN_Y, TRAIN_UOUT)
    target_sample = train_dataset[0]['cv_label']
    origin_target_sample = train_dataset[0]['label']

    print(origin_target_sample)
    target_sample = cat2label(target_sample)
    print(target_sample)
    # train_b_id, df_train, valid_b_id, df_valid = get_train_valid_b_id(train_df, c_fold=0)
    

    # train_data_retriever = DataRetriever_LSTM(
    #     df_train,
    #     scaler,
    #     train_b_id.tolist(),
    #     train_flag = True)

    # valid_data_retriever = DataRetriever_LSTM(
    #     df_valid,
    #     scaler,
    #     valid_b_id.tolist(),
    #     train_flag = True)
    # print(train_data_retriever[0])
    # x_train = np.load('../FE/x_train.npy')
    # y_train = np.load('../FE/y_train.npy')

    # dataset = PressDataset(x_train, y_train)

    # print(dataset[0]['input'].shape)
    # print(dataset[0]['label'].shape)
