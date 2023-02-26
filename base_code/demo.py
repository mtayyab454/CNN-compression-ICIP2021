import torch
import torch.nn as nn

import torchvision.models as models

from basis_layer import replace_conv2d_with_basisconv2d

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    # Replace all Conv2d layers in model with BasisConv2d
    replace_conv2d_with_basisconv2d(model)

    # Test the model
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)