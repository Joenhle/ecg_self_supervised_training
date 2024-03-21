import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.resnet1d import ResNet1D
from typing import Union


class SimCLR(nn.Module):

    name = 'simclr'

    def __init__(self, backbone: Union[ResNet1D], contrast_batch_size, n_view, temperature, signal_length = 1200, mle_outshape = 1024, device = 'cuda'):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        backbone_outshape = backbone.in_out_shape_map[signal_length]
        self.mle = nn.Sequential(nn.Linear(backbone_outshape, backbone_outshape), nn.ReLU(), nn.Linear(backbone_outshape, mle_outshape))
        self.contrast_batch_size = contrast_batch_size
        self.n_view = n_view
        self.device = torch.device(device)
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def forward_features(self, x):
        x = self.backbone.forward_features(x)
        x = self.mle(x)
        return x
        
    def forward(self, x):
        features = self.forward_features(x)
        labels = torch.cat([torch.arange(self.contrast_batch_size) for i in range(self.n_view)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        
        return logits, labels, loss
    
    def forward_loss(self, features):
        logits, labels, loss = self.forward(features)
        return loss

    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res