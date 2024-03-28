import os
import torch
import torch
import pytz
import uuid
import torch.optim as optim
import torch.nn as nn
import signal
import model.mae as model_mae_1d    
from dataset import PretrainDataset, PreTrainContrastDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from util.schedule import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model.resnet18_autoencoder import AutoEncoder
from model.simclr import SimCLR
from model.backbone import resnet1d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)
print(device)

def init_logger(model_name, exp_id):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建文件处理程序
    root = f'/root/Joenhle/ecg_self_supervised_training/ckpt/pre_train/win1200_200wtrain_20wval/{model_name}/{exp_id}'
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

def pre_train(model_name:str, backbone_name:str = None):
    # make model
    if model_name == 'mae':
        from config import MaePreTrainConfig as PreTrainConfig
        model = model_mae_1d.mae_prefer_custom(winsize=PreTrainConfig.winsize, patch_size=PreTrainConfig.patch_size)
    elif model_name == 'resnet_autoencoder':
        from config import ResnetAEPreTrainConfig as PreTrainConfig
        model = AutoEncoder()
    elif model_name == 'simclr':
        from config import SimCLRPreTrainConfig as PreTrainConfig
        backbone = None
        if backbone_name == 'resnet':
            backbone = resnet1d.get_resnet_v1()
            model = SimCLR(
                backbone=backbone,
                contrast_batch_size=PreTrainConfig.batch_size,
                n_view=PreTrainConfig.n_view,
                temperature=PreTrainConfig.temperature,
                signal_length=1200,
                mle_outshape=1024
            )
        elif backbone_name == 'vit':
            pass
    model = model.to(device)
    # make data
    if model_name == 'simclr':
        train_dataset = PreTrainContrastDataset(PreTrainConfig.train_data_path, n_view=PreTrainConfig.n_view)
        val_dataset = PreTrainContrastDataset(PreTrainConfig.val_data_path, n_view=PreTrainConfig.n_view)
    else:
        train_dataset = PretrainDataset(PreTrainConfig.train_data_path)
        val_dataset = PretrainDataset(PreTrainConfig.val_data_path)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=PreTrainConfig.batch_size, shuffle=True, num_workers=4, worker_init_fn=dataloader_signal_handle)
    val_dataloader = DataLoader(val_dataset, batch_size=PreTrainConfig.batch_size, shuffle=True, num_workers=4, worker_init_fn=dataloader_signal_handle)
    
    
    train_dataset.name = 'win1200_200wtrain_20wval'
    exp_id = f'{datetime.now().astimezone(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d%H%M")}'
    logger = init_logger(model_name, exp_id)

    # make ckpt_dir
    ckpt_dir = 'ckpt/pre_train/{}/{}/{}'.format(train_dataset.name, model.name, exp_id)
    writer = SummaryWriter(log_dir=ckpt_dir, flush_secs=2)
    early_stopping = EarlyStopping(patience=30, save_dir=ckpt_dir)
    
    # ready train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.8, min_lr=1e-8)

    step = 0
    all_loss = 0
    for epoch in range(PreTrainConfig.epoch_num):
        # train
        prog_iter = tqdm(train_dataloader, desc="training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            model.train()
            if model_name == 'simclr':
                batch = torch.cat(batch, dim=0)
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.forward_loss(batch)
            loss.backward()
            optimizer.step()
            all_loss += loss

            step += 1
            if step % PreTrainConfig.val_every_n_steps == 0:
                logger.info(f'batch_iter={step // PreTrainConfig.val_every_n_steps} train_loss={all_loss}')
                writer.add_scalar('loss/train', all_loss, step // PreTrainConfig.val_every_n_steps)
                all_loss = 0
                # validate
                model.eval()
                prog_iter = tqdm(val_dataloader, desc="validating", leave=False)
                avg_top1, avg_top5 = 0, 0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter):
                        if model_name == 'simclr':
                            batch = torch.cat(batch, dim=0)
                            batch = batch.to(device)
                            logits, labels, loss = model(batch)
                            top1, top5 = model.accuracy(logits, labels, topk=(1, 5))
                            avg_top1 += top1[0]
                            avg_top5 += top5[0]
                        else:
                            batch = batch.to(device)
                            loss = model.forward_loss(batch)
                        all_loss += loss
                if model_name == 'simclr':
                    logger.info(f'batch_iter={step // PreTrainConfig.val_every_n_steps} val_loss={all_loss}, top1={avg_top1 / (batch_idx + 1)}%, top5={avg_top5 / (batch_idx + 1)}%')
                else:
                    logger.info(f'batch_iter={step // PreTrainConfig.val_every_n_steps} val_loss={all_loss}')
                writer.add_scalar('loss/val', all_loss, step // PreTrainConfig.val_every_n_steps)
                scheduler.step(all_loss)
                early_stopping.check(all_loss, model)
                if early_stopping.early_stop:
                    return
                all_loss = 0

if __name__ == '__main__':
    backbone = None
    # model_name = 'mae'
    # model_name = 'resnet_autoencoder'
    model_name, backbone = 'simclr', 'resnet'
    pre_train(model_name=model_name, backbone_name=backbone)
