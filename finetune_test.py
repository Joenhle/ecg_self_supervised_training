import torch
import pytz
import signal
import torch.optim as optim
from tqdm import tqdm
from dataset import MitDataset
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report 
from model.resnet18_autoencoder import AutoEncoder
from model.classifier import MlpHeadV1, Classifier
from util.schedule import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)
print(device)

def signal_handler(sig, frame):
    pass
signal.signal(signal.SIGTERM, signal_handler)

def dataloader_signal_handle(worker_id):
    def signal_handler(sig, frame):
        pass
    signal.signal(signal.SIGTERM, signal_handler)

def infer(model, data_loader, task, output_metrics = False, writer = None, iter = None):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    prog_iter = tqdm(data_loader, task, leave=False)
    y_true_list = []
    y_pred_list = []
    all_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            input_y = input_y.reshape(-1)
            pred = model(input_x)
            y_true_list.extend(input_y.tolist())
            y_pred_list.extend(torch.max(pred, dim=1)[1].tolist())
            loss = loss_func(pred, input_y)
            all_loss += loss

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='micro')
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='macro')
    if output_metrics:
        arr_metrics = precision_recall_fscore_support(y_true_list, y_pred_list)
        writer.add_scalar('A/precision', arr_metrics[0][0], iter)
        writer.add_scalar('A/recall', arr_metrics[1][0], iter)
        writer.add_scalar('A/f1', arr_metrics[2][0], iter)
        
        writer.add_scalar('V/precision', arr_metrics[0][1], iter)
        writer.add_scalar('V/recall', arr_metrics[1][1], iter)
        writer.add_scalar('V/f1', arr_metrics[2][1], iter)
        
        writer.add_scalar('N/precision', arr_metrics[0][2], iter)
        writer.add_scalar('N/recall', arr_metrics[1][2], iter)
        writer.add_scalar('N/f1', arr_metrics[2][2], iter)
        
        writer.add_scalar('micro-f1', micro_f1, iter)
        writer.add_scalar('macro/precision', macro_p, iter)
        writer.add_scalar('macro/recall', macro_r, iter)
        writer.add_scalar('macro/f1', macro_f1, iter)
        
    print(classification_report(y_true_list, y_pred_list))    
    return all_loss, micro_f1, macro_p, macro_r, macro_f1

def finetune(model_name, backbone, classifier_head_name):
    # make classifier_model (pretrain_model(backbone) + classifier_head)
    pre_train_model, classifier_head = None, None
    if model_name == 'resnet_autoencoder':
        from config import ResnetAEFineTuneConfig as FineTuneConfig
        pre_train_model = AutoEncoder()
    elif model_name == 'mae':
        from config import MaeFineTuneConfig as FineTuneConfig
        pass
    elif model_name == 'simclr':
        from config import SimCLRFineTuneConfig as FineTuneConfig
        pass

    if classifier_head_name == 'mlp_v1':
        classifier_head = MlpHeadV1(pretrain_out_dim=pre_train_model.out_dim, class_n=FineTuneConfig.class_n)
    
    model = Classifier(pre_train_model=pre_train_model, classifier_head=classifier_head)
    checkpoint = torch.load(FineTuneConfig.ckpt_path, map_location='cpu')
    model.pre_train_model.load_state_dict(checkpoint, strict=True)
    
    if FineTuneConfig.pretrain_model_freeze:
        for _, p in model.pre_train_model.named_parameters():
            p.requires_grad = False
    
    model = model.to(device)

    # make data
    train_dataset = MitDataset(FineTuneConfig.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=FineTuneConfig.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False, worker_init_fn=dataloader_signal_handle)
    # TODO 后面改成直接把运行时刻的pretrain或者finetune_test文件+config文件直接拷贝到ckpt目录里形成快照
    val_dataset = MitDataset(FineTuneConfig.val_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=FineTuneConfig.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False, worker_init_fn=dataloader_signal_handle)


    # make ckpt_dir
    ckpt_dir = 'ckpt/classifier/{}/{}/{}/{}'.format(FineTuneConfig.train_data_name, FineTuneConfig.val_data_name, model.name, datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d%H%M"))
    writer = SummaryWriter(log_dir=ckpt_dir, flush_secs=2)
    
    # ready train
    early_stopping = EarlyStopping(patience=30, save_dir=ckpt_dir)
    batch_iter = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.8, min_lr=1e-8)
    
    for epoch in range(FineTuneConfig.epoch_num):
        # train
        all_loss = 0
        prog_iter = tqdm(train_dataloader, desc="training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            model.train()
            input_x, input_y = tuple(t.to(device) for t in batch)
            input_y = input_y.reshape(-1)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            
        # validate
        print(all_loss)
        writer.add_scalar('loss/train', all_loss, batch_iter)
        all_loss, micro_f1, macro_p, macro_r, macro_f1 = infer(model, val_dataloader, "validating", True, writer, batch_iter)
        writer.add_scalar('loss/val', all_loss, batch_iter)
        batch_iter += 1
        scheduler.step(macro_f1)
        early_stopping(all_loss, macro_f1, model)
        all_loss = 0
        if early_stopping.early_stop:
            return

def test(model_name, backbone):
    pass


if __name__ == '__main__':

    backbone = None
    model_name = "resnet_autoencoder"
    classifier_head_name = "mlp_v1"
    finetune(model_name=model_name, backbone=backbone, classifier_head_name=classifier_head_name)
