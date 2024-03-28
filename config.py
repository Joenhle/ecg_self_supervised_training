class EmptyConfig:
    pass

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
    val_every_n_steps = 20


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
    ckpt_path = "/root/Joenhle/ecg_self_supervised_training/ckpt/pre_train/win1200_200wtrain_20wval/mae/202403262235/min_val_loss=87.66512298583984.pth"
    winsize = 1200
    patch_size = 48
    pass

class SimCLRFineTuneConfig(FineTuneConfig):
    ckpt_path = ""
    pass

class TestConfig:
    test_data_name = "test_4113"
    test_data_path = "/root/mit_ecg_dataset/index/test_4113.txt"
    class_n = 3
    epoch_num = 100
    batch_size = 8
    pass

class ResnetAETestConfig(TestConfig):
    ckpt_path = "/root/Joenhle/ecg_self_supervised_training/ckpt/classifier/train_1173/val_588/resnet_autoencoder+mlp_v1/202403281212/max_f1=0.4693944962076419.pth"

class MaeTestConfig(TestConfig):
    ckpt_path = "/root/Joenhle/ecg_self_supervised_training/ckpt/classifier/train_1173/val_588/mae+mlp_v1/202403281507/max_f1=0.6616821585638027.pth"
    winsize = 1200
    patch_size = 48
    pass

class SimCLRTestConfig(TestConfig):
    ckpt_path = ""
    pass