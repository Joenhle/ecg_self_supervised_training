import os
import argparse
import torch
import torch
import pytz
import torch.optim as optim
import torch.nn as nn
import signal

from dataset import PretrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from util.schedule import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import PreTrainConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)
print(device)

def init_logger(exp_id):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建文件处理程序
    root = f'/root/Joenhle/ecg_self_supervised_training/ckpt/pre_train/win1200_200wtrain_20wval/resnet_autoencoder/{exp_id}'
    os.makedirs(root, True)
    path = os.path.join(root, f"{exp_id}_exp.log")
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 添加文件处理程序到logger
    logger.addHandler(file_handler)
    return logger

def signal_handler(sig, frame):
    pass
signal.signal(signal.SIGTERM, signal_handler)

def dataloader_signal_handle(worker_id):
    def signal_handler(sig, frame):
        pass
    signal.signal(signal.SIGTERM, signal_handler)

def pre_train(model_name:str):
    # make model
    if model_name == 'mae':
        from config import MaePreTrainConfig
        import model.mae as model_mae_1d    
        model = model_mae_1d.mae_prefer_custom(winsize=MaePreTrainConfig.winsize, patch_size=MaePreTrainConfig.patch_size)
        model = model.to(device)
    elif model_name == 'resnet_autoencoder':
        from config import ResnetAEPreTrainConfig
        from model.resnet18_autoencoder import AutoEncoder
        model = AutoEncoder()
        model = model.to(device)

    # make data
    train_dataset = PretrainDataset(PreTrainConfig.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=PreTrainConfig.batch_size, shuffle=True, num_workers=6, worker_init_fn=dataloader_signal_handle)
    val_dataset = PretrainDataset(PreTrainConfig.val_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=PreTrainConfig.batch_size, shuffle=True, num_workers=6, worker_init_fn=dataloader_signal_handle)
    train_dataset.name = 'win1200_200wtrain_20wval'


    exp_id = f'{datetime.now().astimezone(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d%H%M")}'
    logger = init_logger(exp_id)

    # make ckpt_dir
    ckpt_dir = 'ckpt/pre_train/{}/{}/{}'.format(train_dataset.name, model.name, exp_id)
    writer = SummaryWriter(log_dir=ckpt_dir, flush_secs=2)
    early_stopping = EarlyStopping(patience=30, save_dir=ckpt_dir)
    
    # ready train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.8, min_lr=1e-8)
    
    for epoch in range(PreTrainConfig.epoch_num):
        # train
        model.train()
        all_loss = 0
        prog_iter = tqdm(train_dataloader, desc="training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.forward_loss(batch)
            loss.backward()
            optimizer.step()
            all_loss += loss
        logger.info(f'epoch={epoch} train_loss={all_loss}')
        writer.add_scalar('loss/train', all_loss, epoch)
        
        # validate
        model.eval()
        prog_iter = tqdm(val_dataloader, desc="validating", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter):
                batch = batch.to(device)
                loss = model.forward_loss(batch)
                all_loss += loss
        logger.info(f'epoch={epoch} val_loss={all_loss}')
        writer.add_scalar('loss/val', all_loss, epoch)
        scheduler.step(all_loss)
        early_stopping.check(all_loss, model)
        if early_stopping.early_stop:
            return

if __name__ == '__main__':
    # model_name = 'mae'
    model_name = 'resnet_autoencoder'
    pre_train(model_name)
