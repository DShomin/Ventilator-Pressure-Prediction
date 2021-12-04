import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader
import random
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint#, StochasticWeightAveraging
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from typing import Optional

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold

from model import LSTM_2, LSTM_ATTN, WaveNet, LSTM
from data import PressDataset, VentilatorCEDataset, cat2label

import optuna
from optuna.integration import PyTorchLightningPruningCallback

BASE_DIR = '../data/'
TRAIN_PATH = os.path.join(BASE_DIR, 'train_dataset')
TEST_PATH = os.path.join(BASE_DIR, 'test_dataset')

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        n_layer=2,
        loss_fn='mae',
        opt='adamw',
        backbone: Optional[LSTM] = None,
        learning_rate: float = 2e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])

        if backbone is None:
            backbone = LSTM(n_layer=n_layer)
        #     # backbone = WaveNet(in_dim=50)

        self.backbone = backbone
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            self.criterion = nn.HuberLoss()
        elif loss_fn == 'l1smooth':
            self.criterion = nn.SmoothL1Loss()

        self.opt = opt


    def forward(self, x):
        x = self.backbone.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        
        x, z_l, l = batch['input'], batch['zero_label'], batch['label']

        output = self.backbone(x)[:, :, 0]

        loss_1 = self.criterion(output, z_l)
        loss_2 = self.criterion(output, l)
        loss = loss_1 + loss_2
        try:
            pred = output.detach().cpu()
            self.log("Train E Loss", loss, on_step= True,prog_bar=True, logger=True)

        
        except:
            pass

        return {"loss": loss, 
        "e_predictions": output.detach().cpu(), 
        }

    def training_epoch_end(self, outputs):

        e_preds = []
        e_labels = []
        r_preds = []
        r_labels = []
        flags = []
        
        for output in outputs:
            
            e_preds += output['e_predictions']
            e_labels += output['e_labels']
            r_preds += output['r_predictions']
            r_labels += output['r_labels']
            flags += output['flag']

        e_labels = torch.stack(e_labels).detach().cpu()
        e_preds = torch.stack(e_preds).detach().cpu()

        r_labels = torch.stack(r_labels).detach().cpu()
        r_preds = torch.stack(r_preds).detach().cpu()
        flags = torch.stack(flags).detach().cpu()

        e_trn_mae = mean_absolute_error(e_labels, e_preds)
        e_trn_zero_mae = mean_absolute_error(e_labels[flags], e_preds[flags])

        r_trn_mae = mean_absolute_error(r_labels, r_preds)
        r_trn_zero_mae = mean_absolute_error(r_labels[flags], r_preds[flags])
        
        self.log("E trn_mae", e_trn_mae, prog_bar=True, logger=True)
        self.log('E trn_zero_mae', e_trn_zero_mae, prog_bar=True, logger=True)

        self.log("R trn_mae", r_trn_mae, prog_bar=True, logger=True)
        self.log('R trn_zero_mae', r_trn_zero_mae, prog_bar=True, logger=True)
        

    def validation_step(self, batch, batch_idx):

        x, r_y, e_y, flag = batch['input'], batch['label'], batch['e_label'], batch['zero_uout_flag']
        x = x
        labels = e_y

        output = self.backbone(x)[:, :, 0]
        loss = self.criterion(output, labels)

        r_pred = cat2label(output.detach().cpu())
        
        self.log('val_loss', loss, on_step= True, prog_bar=True, logger=True)
        return {"loss": loss, 
        "e_predictions": output.detach().cpu(), 
        "e_labels": labels.detach().cpu(), 
        "r_predictions": r_pred, 
        "r_labels":r_y,
        'flag':flag}

    def validation_epoch_end(self, outputs):
        
        e_preds = []
        e_labels = []
        r_preds = []
        r_labels = []
        flags = []
        
        for output in outputs:
            
            e_preds += output['e_predictions']
            e_labels += output['e_labels']
            r_preds += output['r_predictions']
            r_labels += output['r_labels']
            flags += output['flag']

        e_labels = torch.stack(e_labels).detach().cpu()
        e_preds = torch.stack(e_preds).detach().cpu()

        r_labels = torch.stack(r_labels).detach().cpu()
        r_preds = torch.stack(r_preds).detach().cpu()
        flags = torch.stack(flags).detach().cpu()

        e_val_mae = mean_absolute_error(e_labels, e_preds)
        e_val_zero_mae = mean_absolute_error(e_labels[flags], e_preds[flags])

        r_val_mae = mean_absolute_error(r_labels, r_preds)
        r_val_zero_mae = mean_absolute_error(r_labels[flags], r_preds[flags])
        
        self.log("E val_mae", e_val_mae, prog_bar=True, logger=True)
        self.log('E val zero_mae', e_val_zero_mae, prog_bar=True, logger=True)

        self.log("R val mae", r_val_mae, prog_bar=True, logger=True)
        self.log('R val zero mae', r_val_zero_mae, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        out = self.backbone(batch)
        return out

    def configure_optimizers(self):
        if self.opt == 'adamw':
            optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.hparams.learning_rate)
        elif self.opt == 'adam':
            optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.hparams.learning_rate)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.backbone.parameters(), lr=self.hparams.learning_rate)
        elif self.opt == 'rms':
            optimizer = torch.optim.RMSprop(self.backbone.parameters(), lr=self.hparams.learning_rate)
        
        # scheduler_cosie = CosineAnnealingLR(optimizer, T_max= 10, eta_min=1e-6, last_epoch=-1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)
        # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosie)
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='R val zero mae') # , lr_scheduler=scheduler_warmup lr_scheduler=scheduler[optimizer], [scheduler]

class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        TRAIN_X,
        TRAIN_UOUT,
        TRAIN_Y,
        VALID_X,
        VALID_UOUT,
        VALID_Y,
        cutout_size,
        num_cutout,
        batch_size: int = 1024,
    ):
        super().__init__()
        if cutout_size == 0:
            cutout_ratio=0.0
        else:
            cutout_ratio=0.3
        # trn_dataset = PressDataset(TRAIN_X, TRAIN_Y, TRAIN_UOUT, num_cutout=num_cutout, cutout_ratio=cutout_ratio , cutout_size=cutout_size) 
        trn_dataset = VentilatorCEDataset(TRAIN_X, TRAIN_Y, TRAIN_UOUT)
        # val_dataset = PressDataset(VALID_X, VALID_Y, VALID_UOUT, num_cutout=0, cutout_ratio=0.0, cutout_size=0) 
        val_dataset = VentilatorCEDataset(VALID_X, VALID_Y, VALID_UOUT)
        
        self.train_dset = trn_dataset
        self.valid_dset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.valid_dset, batch_size=self.batch_size, shuffle=False, num_workers=6) 

def cli_main() -> float:
    
    loss = 'mse' # l1smooth
    n_layers = 3

    # Data augemntation
    cutout_size = 5
    num_cutout = 1
    
    # model
    model = LSTM_2()

    classifier =  LitClassifier(n_layer=n_layers, loss_fn=loss, backbone=model)

    logger = WandbLogger(name=f'LSTM_ec_regre_mse_fold{C_FOLD}', project='kaggle pressure')
    mc = ModelCheckpoint('model', monitor='R val zero mae', mode='min', filename='{epoch}-{val_mae:.4f}_' + f'fold_{C_FOLD}')
    # swa = StochasticWeightAveraging(swa_epoch_start=2, annealing_epochs=2)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=300,
        # accelerator='ddp_spawn',
        # stochastic_weight_avg=True,
        callbacks=[
            mc, 
        ],
        logger=logger
        )
    mydatamodule = MyDataModule(TRAIN_X, TRAIN_UOUT, TRAIN_Y, VALID_X, VALID_UOUT, VALID_Y, num_cutout, cutout_size)
    trainer.fit(classifier, datamodule=mydatamodule)
    
    return trainer.checkpoint_callback.best_model_score# trainer.callback_metrics["val_mae"].item()

def optuna_cli_main(trial: optuna.trial.Trial) -> float:

    # Optimizer
    # opt = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rms', 'adamw'])
    opt = 'adam'

    # loss = trial.suggest_categorical('loss_fn', ['mae', 'mse', 'huber'])
    loss = 'huber'
    # n_layers = 3

    # Data augemntation
    # cutout_size = trial.suggest_int('cutout_size', 0, 10)
    cutout_size = 5
    num_cutout = trial.suggest_int('num_cutout', 1, 5)
    
    # model
    # model_name = trial.suggest_categorical('model_name', ['LSTM', 'LSTM_ATTN'])
    # n_layers = trial.suggest_int('num_lstm_layer', 2, 4)


    n_layers = 2
    # if model_name == 'LSTM':
    #     model = LSTM(n_layer=n_layers)
    # elif model_name == 'LSTM_ATTN':
    #     model = LSTM_ATTN()
    model = LSTM_2()

    classifier =  LitClassifier(n_layer=n_layers, loss_fn=loss, backbone=model, opt=opt)

    mc = ModelCheckpoint('model', monitor='val_mae', mode='min', filename='{epoch}-{val_mae:.4f}_' + f'fold_{C_FOLD}')
    # swa = StochasticWeightAveraging(swa_epoch_start=2, annealing_epochs=2)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        # accelerator='ddp_spawn',
        # stochastic_weight_avg=True,
        callbacks=[mc, 
        PyTorchLightningPruningCallback(trial, monitor="val_mae")
        ],
        )
    mydatamodule = MyDataModule(TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, num_cutout, cutout_size)
    trainer.fit(classifier, datamodule=mydatamodule)
    
    return trainer.checkpoint_callback.best_model_score# trainer.callback_metrics["val_mae"].item()

def get_dummy(df):
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)
    return df

if __name__ == '__main__':
    sample_idx = 1000
    # x_trian = np.load('../FE/x_train.npy')#[:sample_idx]
    x_trian = np.load('../FE/train_f.npy')#[:sample_idx]
    train_zero_flag = np.load('../FE/uout_train.npy')#[:sample_idx]
    y_train = np.load('../FE/y_train.npy')#[:sample_idx]
    
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    train = get_dummy(train)
    test = get_dummy(test)

    x_train = train.drop(['id', 'breath_id', 'pressure'], axis=1)
    x_u_zero = train['u_out']
    y_train = train['pressure']
    
    # split fold
    C_FOLD = 0
    # skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf = KFold(n_splits=5, random_state=42, shuffle=True)

    for i, (train_idx, valid_idx) in enumerate(skf.split(X=x_train)):

        if i == C_FOLD:
            TRAIN_X = x_train[train_idx]
            TRAIN_UOUT = train_zero_flag[train_idx]
            TRAIN_Y = y_train[train_idx]


            VALID_X = x_train[valid_idx]
            VALID_UOUT = train_zero_flag[valid_idx]
            VALID_Y = y_train[valid_idx]

    seed_everything()
    cli_main()

    # study = optuna.create_study(storage='sqlite:///db.sqlite3', study_name='num cutout')
    # study.optimize(optuna_cli_main, n_trials=10)

    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: {}".format(trial.value))

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))