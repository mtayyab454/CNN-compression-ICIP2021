import torch
import torch.nn as nn

import torchvision.models as models

from basis_layer import replace_conv2d_with_basisconv2d, replace_basisconv2d_with_conv2d

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    # Replace all Conv2d layers in model with BasisConv2d
    replace_conv2d_with_basisconv2d(model)
    basis_model_output = model(input_tensor)

    # Replace all BasisConv2d layers in model with Conv2d
    replace_basisconv2d_with_conv2d(model)
    model_output = model(input_tensor)
    print( (model_output-basis_model_output).abs().sum() )
    print(torch.allclose(model_output, basis_model_output, atol=1e-2))  # True