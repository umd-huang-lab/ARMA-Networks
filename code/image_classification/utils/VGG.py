import math
import torch.nn as nn
import torch.nn.functional as F
from .ARMA_Layer import ARMA2d
from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, arma=True, dataset="CIFAR10", rf_init=0, w_kernel_size=3, a_kernel_size=3, 
                                                                           batch_norm=False,  dropout=True):
        super(VGG, self).__init__()
        
        num_classes = {   "MNIST":  10,
                        "CIFAR10":  10,
                       "CIFAR100": 100,
                       "ImageNet":1000}[dataset]

        self.features = self._make_layers(cfg[vgg_name], batch_norm, arma, rf_init, w_kernel_size, a_kernel_size)
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
                nn.LogSoftmax(dim=-1)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



    def _make_layers(self, cfg, batch_norm, arma, init, w_ksz, a_ksz):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if arma:
                    conv2d = ARMA2d(in_channels, x, w_kernel_size = w_ksz, w_padding = w_ksz//2, 
                                                    a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2)  
                else:
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=w_ksz, padding=w_ksz//2)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

