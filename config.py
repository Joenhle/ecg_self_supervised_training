class PreTrainConfig:
    data_standardization = True
    batch_size = 512
    epoch_num = 1000
    train_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/train_200w.txt'
    val_data_path = '/root/physionet.org/data/p1_people1000_segmentall_sample_data/win1200/val_20w.txt'

class MaePreTrainConfig(PreTrainConfig):
    winsize = 1200
    patch_size = 48

class ResnetAEPreTrainConfig(PreTrainConfig):
    pass

