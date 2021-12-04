from VFE import add_features
import pandas as pd
import numpy as np
from pickle import load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch
import pytorch_lightning as pl

from model import LSTM_2, LSTM, LSTM_ATTN

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        learning_rate: float = 2e-3,
    ):
        super().__init__()

        self.backbone = LSTM_2()

    def forward(self, x):
        x, _ = self.backbone.lstm1(x)
        x, _ = self.backbone.lstm2(x)
        x, _ = self.backbone.lstm3(x)
        x, _ = self.backbone.lstm4(x)
        x = self.backbone.fc1(x)
        x = self.backbone.selu(x)
        x = self.backbone.fc2(x)
        x = self.backbone.outact(x) # add
        return x

class PressTestDataset(Dataset):
    def __init__(self, x):
        self.input = x

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        
        x = self.input[index]
        return x


def inference(ckpt_file, model, dataloader):
    model = model.load_from_checkpoint(f'./model/{ckpt_file}')
    model = model.eval().cuda()

    output_list = list()
    with torch.no_grad():
        for x in tqdm(dataloader):
            output = model(x.cuda().float())
            output_list.append(output)

    test_preds = torch.cat(output_list).cpu().numpy()

    return test_preds

if __name__ == '__main__':

    # test set
    test_ori = pd.read_csv('../data/test.csv')
    test = add_features(test_ori)
    test.drop(['id', 'breath_id'], axis=1, inplace=True)

    RS = load(open('../FE/RS.pkl', 'rb'))
    test = RS.transform(test)
    test = test.reshape(-1, 80, test.shape[-1])

    test_dataset = PressTestDataset(test)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=8)


    model = LitClassifier()
    ckpt_file = 'epoch=99-val_mae=0.1883_fold_0.ckpt'
    model = model.load_from_checkpoint(f'./model/{ckpt_file}')
    model = model.eval().cuda()

    output_list = list()
    with torch.no_grad():
        for x in tqdm(test_dataloader):
            output = model(x.cuda().float())
            output_list.append(output)

    test_preds = torch.cat(output_list).cpu().numpy()

    pressure = np.load('../FE/y_train.npy')
    P_MIN = np.min(pressure)
    P_MAX = np.max(pressure)
    P_STEP = pressure[0][1] - pressure[0][0]
    print('Min pressure: {}'.format(P_MIN))
    print('Max pressure: {}'.format(P_MAX))
    print('Pressure step: {}'.format(P_STEP))
    print('Unique values:  {}'.format(np.unique(pressure).shape[0]))
    
    submission = pd.read_csv('../data/sample_submission.csv')

    # For ensemble
    # submission["pressure"] = np.median(np.vstack(test_preds),axis=0)

    submission["pressure"] = np.vstack(test_preds)[:, 0]
    submission["pressure"] = np.round((submission.pressure - P_MIN)/P_STEP) * P_STEP + P_MIN
    submission.pressure = np.clip(submission.pressure, P_MIN, P_MAX)
    submission.to_csv('submission.csv', index=False)