import math
import torch.nn as nn
import torch.nn.functional as F
from .ARMA_Layer import ARMA2d

class AlexNet(nn.Module):
    def __init__(self, arma, num_classes, init, w_ksz, a_ksz):
        super(AlexNet,  self).__init__()
        if arma:
            self.features = nn.Sequential(
                ARMA2d(3, 64, w_kernel_size=w_ksz, w_stride=2, w_padding=w_ksz//2, 
                              a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

                ARMA2d(64, 192, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
                                a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

                ARMA2d(192, 384, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
                                 a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
                nn.ReLU(inplace=True),

                ARMA2d(384, 256, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
                                 a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
                nn.ReLU(inplace=True),
                
                ARMA2d(256, 256, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
                                 a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
                )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=w_ksz, stride=2, padding=w_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(64, 192, kernel_size=w_ksz, padding=w_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(192, 384, kernel_size=w_ksz, padding=w_ksz//2),
                nn.ReLU(inplace=True),

                nn.Conv2d(384, 256, kernel_size=w_ksz, padding=w_ksz//2),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=w_ksz, padding=w_ksz//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x



def AlexNet_(arma=True, dataset="CIFAR10", rf_init=0, w_kernel_size=3, a_kernel_size=3):
    num_classes = {       "MNIST":  10,
                        "CIFAR10":  10,
                       "CIFAR100": 100,
                       "ImageNet":1000}[dataset]
                       
    return AlexNet(arma, num_classes, rf_init, w_kernel_size, a_kernel_size)
