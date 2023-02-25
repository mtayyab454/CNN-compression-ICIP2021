import torch
from basis_layer import BasisConv2d
import torchvision.models as models
import torch.nn as nn

use_weights = True
add_bn = True
trainable_basis = True
replace_fc = False
sparse_filters = False

model = models.vgg16(pretrained=True)
# replace each Conv2d layer in mycnn with BasisConv2d
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        weight = module.weight.data.clone()
        bias = module.bias.data.clone() if module.bias is not None else None
        in_channels = module.in_channels
        basis_channels = min(module.out_channels, module.weight.numel() // module.weight.size(0))
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        setattr(model, name, BasisConv2d(weight, bias, add_bn, in_channels, basis_channels, out_channels, kernel_size))

model(torch.randn(1, 3, 224, 224))
