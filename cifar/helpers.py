import torch
import torch.nn as nn

from base_code.basisModel import basisModel
from base_code.basisModel import display_stats as base_display_stats
from models.models import get_cifar_models

from collections import OrderedDict



def display_stats(basis_model, model, exp_name, input_size=[32, 32], count_relu=False):
    return base_display_stats(basis_model, model, exp_name, input_size, count_relu)