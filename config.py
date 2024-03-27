class PreTrainConfig:
    data_standardization = True
    batch_size = 512
    epoch_num = 1000
    train_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/train_100w.txt'
    val_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/val_10w.txt'
    # 每隔多少step验证一次，训练集太大，一个epoch太长
    val_every_n_steps = 200


class ResnetAEPreTrainConfig(PreTrainConfig):
    val_every_n_steps = 200
    pass

class MaePreTrainConfig(PreTrainConfig):
    winsize = 1200
    patch_size = 48

class SimCLRPreTrainConfig(PreTrainConfig):
    batch_size = 400 # simclr的batch size建议大一点
    n_view = 2
    temperature = 0.07


class FineTuneConfig:
    train_data_name = "train_1173"
    val_data_name = "val_588"
    train_data_path = "/root/mit_ecg_dataset/index/train_1173.txt"
    val_data_path = "/root/mit_ecg_dataset/index/val_588.txt"
    
    class_n = 3
    epoch_num = 100
    batch_size = 8
    pretrain_model_freeze = True

class ResnetAEFineTuneConfig(FineTuneConfig):
    ckpt_path = "/root/Joenhle/ecg_self_supervised_training/ckpt/pre_train/win1200_200wtrain_20wval/resnet_autoencoder/202403252311/min_val_loss=62.15719985961914.pth"

class MaeFineTuneConfig(FineTuneConfig):
    ckpt_path = ""
    pass

class SimCLRFineTuneConfig(FineTuneConfig):
    ckpt_path = ""
    pass


class TestConfig:
    pass