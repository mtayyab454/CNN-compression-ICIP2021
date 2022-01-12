'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from collections import OrderedDict
from torch.hub import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, l=512):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(l, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


x = 1
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'D2': [40, 40, 'M', 80, 80, 'M', 160, 160, 160, 'M', 160, 160, 160, 'M', 320, 320, 320, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

def vgg16_D1(pretrained):
    model = VGG(make_layers(cfg['D1']), num_classes=100, l=256)

    return model

def vgg16_D2(pretrained):
    model = VGG(make_layers(cfg['D2']), num_classes=100, l=320)

    return model

def vgg16_100(pretrained):
    model = vgg16(num_classes=100)

    if pretrained:
        checkpoint = load_state_dict_from_url('https://www.dropbox.com/s/qyae0rktj7qv1tt/vgg16_cifar100.pth?dl=1')
        model.load_state_dict(checkpoint)

    return model

def vgg16_10(pretrained):
    model = vgg16(num_classes=10)

    if pretrained:
        checkpoint = load_state_dict_from_url('https://www.dropbox.com/s/onbz3a8p1gipaz2/vgg16_cifar10.pth?dl=1')
        model.load_state_dict(checkpoint)

    return model

def vgg16_svhn(pretrained):
    model = vgg16(num_classes=10)

    if pretrained:
        checkpoint = load_state_dict_from_url('https://www.dropbox.com/s/su0pzhl8et9f46r/vgg16_svhn.pth?dl=1')
        model.load_state_dict(checkpoint)

    return model

def vgg16_10to100(pretrained): # VGG16 for CIFAR100 initilized with CIFAR10 weights
    model = vgg16_100(pretrained=False)

    if pretrained:
        temp = vgg16_10(True)
        state_dict = temp.state_dict()
        new_dict = OrderedDict()

        for f in state_dict.keys():
            if f[:8] == 'features':
                new_dict.update({f: state_dict[f]})

        model.load_state_dict(new_dict, strict=False)

    return model

def vgg16_100to10(pretrained): # VGG16 for CIFAR10 initilized with CIFAR100 weights
    model = vgg16_10(pretrained=False)

    if pretrained:
        temp = vgg16_100(True)
        state_dict = temp.state_dict()
        new_dict = OrderedDict()

        for f in state_dict.keys():
            if f[:8] == 'features':
                new_dict.update({f: state_dict[f]})

        model.load_state_dict(new_dict, strict=False)

    return model

# model = vgg16_10to100(True)
# print(model(torch.randn(4, 3, 32, 32)))