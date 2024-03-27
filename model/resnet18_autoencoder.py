import torch
import torch.nn as nn
import numpy as np
from typing import Type, List
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3Transposed(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0):
    return nn.ConvTranspose1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding = output_padding, # output_padding is neccessary to invert conv2d with stride > 1
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1Transposed(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0):
    return nn.ConvTranspose1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding = output_padding)

class BasicBlockEnc(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, block: Type[BasicBlockEnc], layers: List[int]):
        super().__init__()
    
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[BasicBlockEnc], planes: int, blocks: int, stride: int = 1,):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class BasicBlockDec(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, output_padding: int = 0, upsample = None):
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3Transposed(planes, inplanes, stride, output_padding=output_padding)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Transposed(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, block, layers: List[int]) -> None:
        super().__init__()
        self.inplanes = 64 # change from 2048 to 64. It should be the shape of the output image chanel.

        self.de_conv1 = nn.ConvTranspose1d(self.inplanes, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.Upsample(scale_factor=2) # NOTE: invert max pooling

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1 ,output_padding = 0, last_block_dim=64)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlockDec], planes: int, blocks: int, stride: int = 2,output_padding: int = 1, last_block_dim: int = 0):
        upsample = None
        layers = []
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes)
            )

        if last_block_dim == 0:
            last_block_dim = self.inplanes//2

        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                conv1x1Transposed(planes, last_block_dim, stride, output_padding),
                nn.BatchNorm1d(last_block_dim),
            )

        layers.append(block(last_block_dim, planes, stride, output_padding, upsample))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.unpool(x)
        x = self.de_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


    def forward(self, x):
        return self._forward_impl(x)

class AutoEncoder(nn.Module):

    name = 'resnet_autoencoder'
    out_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2]) 
        self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])
    
    def forward_loss(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return F.mse_loss(x, y)
    
    def forward_feature(self, x):
        return self.encoder(x).mean(dim=2)
        

if __name__ == '__main__':
    autoencoder = AutoEncoder()

    x = np.random.rand(1200)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    loss = autoencoder.forward_loss(x)
    print(loss)