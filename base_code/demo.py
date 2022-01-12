import torch
from basisModel import basisModel
import torchvision.models as models

use_weights = True
add_bn = True
trainable_basis = True
replace_fc = False
sparse_filters = False

compression_factor = 0.8

model = models.vgg16(pretrained=True)
compressed_model = basisModel(model, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters)
compressed_model.update_channels(compression_factor)

compressed_model(torch.randn(1, 3, 224, 224))
