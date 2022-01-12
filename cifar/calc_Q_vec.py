from helpers import get_Q_vec, get_t_vec
from cifar.models.models import get_cifar_models
from base_code.basisModel import basisModel
from cifar.helpers import display_stats

original_filters = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
Q_vec = get_Q_vec(model_name = 'resnet56-cifar10', filter_list = original_filters, min_map = 92.7, min_Q=4)
model = get_cifar_models('resnet56', 'cifar10', pretrained=False)
basis_model = basisModel(model, use_weights=True, add_bn=True, trainable_basis=True, replace_fc=False, sparse_filters=False)
basis_model.update_channels(Q_vec)
stats = display_stats(basis_model, model, '')