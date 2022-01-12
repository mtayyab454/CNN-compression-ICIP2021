import os
import torch
from cifar.models.vgg import vgg16_100, vgg16_D1, vgg16_D2, vgg16_10, vgg16_svhn
from cifar.models.densenet import densel190k40_100
from cifar.models.alexnet import alexnet_100
from cifar.models.resnet import resnet110_100, resnet56_10

def get_cifar_models(model_name, dataset_name, pretrained=True):

    if dataset_name in ['CIFAR100', 'cifar100']:
        model = get_cifar100_models(model_name, pretrained)
    elif dataset_name in ['CIFAR10', 'cifar10']:
        model = get_cifar10_models(model_name, pretrained)

    return model

def get_cifar100_models(model_name, pretrained):
    model = None

    if model_name == 'alexnet':
        model = alexnet_100(pretrained)
    elif model_name == 'vgg16':
        model = vgg16_100(pretrained)
    ##### Experimental Models ######
    elif model_name == 'vgg16_D1':
        model = vgg16_D1(pretrained)
    elif model_name == 'vgg16_D2':
        model = vgg16_D2(pretrained)
    ################################
    elif model_name == 'resnet110':
        model = resnet110_100(pretrained)
    elif model_name == 'densenet190':
        model = densel190k40_100(pretrained)
    # elif model_name == 'lenet':
    #     model = lenet_cifar100(pretrained)

    return model

def get_cifar10_models(model_name, pretrained):
    model = None

    if model_name == 'vgg16':
        model = vgg16_10(pretrained)
    elif model_name == 'resnet56':
        model = resnet56_10(pretrained)

    return model

# get_cifar_models('resnet56', 'cifar10')
