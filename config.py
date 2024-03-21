class PreTrainConfig:
    data_standardization = True
    batch_size = 512
    epoch_num = 1000
    train_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/train_100w.txt'
    val_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/val_10w.txt'
    # 每隔多少step验证一次，训练集太大，一个epoch太长
    val_every_n_steps = 10



class MaePreTrainConfig(PreTrainConfig):
    winsize = 1200
    batch_size = 48

class ResnetAEPreTrainConfig(PreTrainConfig):
    pass


class SimCLRPreTrainConfig(PreTrainConfig):
    batch_size = 400 # simclr的batch size建议大一点
    n_view = 2
    temperature = 0.07