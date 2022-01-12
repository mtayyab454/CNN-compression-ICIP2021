'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNetL(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetL, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64*2, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64*2, 192*2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192*2, 384*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384*2, 256*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256*2, 256*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256*2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnetL(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNetL(**kwargs)
    return model

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model

import torch
from collections import OrderedDict


def alexnetL_100(pretrained):

    model = alexnetL(num_classes=100)
    return model

def alexnet_100(pretrained):

    model = alexnet(num_classes=100)
    if pretrained:
        state_dict = load_state_dict_from_url('https://www.dropbox.com/s/y8swkxl6or25k63/alexnet_cifar100.pth.tar?dl=1')['state_dict']
        new_dict = OrderedDict()

        for f in state_dict.keys():
            new_dict.update({f[7:]: state_dict[f]})

        model.load_state_dict(new_dict)

    return model

def alexnet_10(pretrained):
    model = alexnet(num_classes=10)
    if pretrained:
        state_dict = load_state_dict_from_url('https://www.dropbox.com/s/vqmop30kutj99om/alexnet_cifar10.pth.tar?dl=1')['state_dict']
        new_dict = OrderedDict()

        for f in state_dict.keys():
            new_dict.update({f.replace('.module', ''):state_dict[f]})

        model.load_state_dict(new_dict)

    return model

# m = alexnet_100(True)
# print(m)
