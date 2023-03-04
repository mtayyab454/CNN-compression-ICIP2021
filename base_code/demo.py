import torch
import torch.nn as nn

import torchvision.models as models

from basis_helpers import replace_conv2d_with_basisconv2d, replace_basisconv2d_with_conv2d, trace_model, get_basis_channels_from_t, display_stats

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    # Replace all Conv2d layers in model with BasisConv2d

    num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
    _, _, basis_channels = get_basis_channels_from_t(model, [1.0]*num_conv)

    basis_model = replace_conv2d_with_basisconv2d(model, basis_channels)
    basis_model_output = basis_model(input_tensor)

    # Replace all BasisConv2d layers in model with Conv2d
    model = replace_basisconv2d_with_conv2d(basis_model)
    model_output = model(input_tensor)
    print( (model_output-basis_model_output).abs().sum() )
    print(torch.allclose(model_output, basis_model_output, atol=1e-2))  # True

    display_stats(basis_model, model, 'test', [3, 224, 224])