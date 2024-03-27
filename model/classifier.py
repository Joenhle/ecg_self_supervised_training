import torch
import torch
from torch import nn

class ClassifierHead(nn.Module):
    pass

class MlpHeadV1(ClassifierHead):
    name = "mlp_v1"
    def __init__(self, pretrain_out_dim, class_n):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(pretrain_out_dim, affine=False, eps=1e-6)
        self.relu = torch.nn.ReLU()
        self.fc = nn.Linear(pretrain_out_dim, class_n)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):

    def __init__(self, pre_train_model, classifier_head):
        super().__init__()
        self.pre_train_model = pre_train_model
        self.classifier_head = classifier_head
    
    @property
    def name(self):
        return f'{self.pre_train_model.name}+{self.classifier_head.name}'
    
    def forward(self, x):
        embedding = self.pre_train_model.forward_feature(x)
        out = self.classifier_head(embedding)
        return out
        