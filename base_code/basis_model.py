import copy

import torch
import torch.nn as nn

from .basis_layer import BasisConv2d

def display_stats(basis_model, model, exp_name, input_size):

    ms = ModelStats()
    _, _, info = ms.get_stats(basis_model, input_size)

    sum_basisconv2d_vals = lambda mylist:sum([val for val, ln in zip(mylist, info['layer_name']) if ln == 'BasisConv2d'])

    org_flops = sum_basisconv2d_vals(info['org_flops'])
    basis_flops = sum_basisconv2d_vals(info['basis_flops'])

    org_filters = sum_basisconv2d_vals(info['out_channels'])
    basis_filters = sum_basisconv2d_vals(info['basis_channels'])

    num_model_conv_param = sum_basisconv2d_vals(info['org_params'])
    num_basis_conv_param = sum_basisconv2d_vals(info['basis_params'])

    num_model_param = sum(p.numel() for p in model.parameters())
    num_basis_param = sum(p.numel() for p in basis_model.parameters())

    print_text = f"\n############################################# {exp_name} #############################################\n"
    print_text += f"\n    Model FLOPs: {org_flops / 10 ** 6:.2f}M"
    print_text += f"\n    Basis Model FLOPs: {basis_flops / 10 ** 6:.2f}M"
    print_text += f"\n    % Reduction in FLOPs: {100 - (basis_flops * 100 / org_flops):.2f} %"
    print_text += f"\n    % Speedup: {org_flops / basis_flops:.2f} %\n"

    print_text += f"\n    Model Conv Params: {num_model_conv_param / 10 ** 6:.2f}M"
    print_text += f"\n    Basis Model Conv params: {num_basis_conv_param / 10 ** 6:.2f}M"
    print_text += f"\n    % Reduction in Conv params: {100 - (num_basis_conv_param * 100 / num_model_conv_param):.2f} %\n"

    print_text += f"\n    Model Total Params: {num_model_param / 10 ** 6:.2f}M"
    print_text += f"\n    Basis Model Total params: {num_basis_param / 10 ** 6:.2f}M"
    print_text += f"\n    % Reduction in Total params: {100 - (num_basis_param * 100 / num_model_param):.2f} %\n"

    print_text += f"\n    Model Filters: {org_filters}"
    print_text += f"\n    Basis Model Filters: {basis_filters}"
    print_text += f"\n    % Reduction in Filters: {100 - (basis_filters * 100.0 / org_filters):.2f} %\n"

    print_text += "\n    Model Accuracy: "
    print_text += "\n    Basis Model Accuracy: "
    print_text += "\n    Reduction in Accuracy: \n"

    print_text += f"\n    Filters in original convs: {info['out_channels']}"
    print_text += f"\n    Filters in basis convs: {info['basis_channels']}"
    print_text += "\n\n#########################################################################################################\n"

    stats = {'print_stats': print_text, 'exp_name': exp_name, 'flops_info': info, 'org_flops': org_flops,
             'basis_flops': basis_flops, 'model_conv_param': num_model_conv_param,
             'basis_conv_param': num_basis_conv_param,
             'model_param': num_model_param, 'basis_param': num_basis_param, 'org_filters': org_filters,
             'basis_filters': basis_filters,
             'org_filters_vec': info['out_channels'],
             'basis_filters_vec': info['basis_channels']}

    print(print_text)
    return stats
def trace_model(model):
    in_channels, out_channels, basis_channels, layer_type = [], [], [], []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels.append(module.in_channels)
            out_channels.append(module.out_channels)
            basis_channels.append(min(module.out_channels, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))
            layer_type.append('conv')
        elif isinstance(module, nn.Linear):
            in_channels.append(module.in_features)
            out_channels.append(module.out_features)
            basis_channels.append(min(module.out_features, module.in_features))
            layer_type.append('linear')

    num_conv = sum(1 for lt in layer_type if lt == 'conv')
    num_linear = sum(1 for lt in layer_type if lt == 'linear')

    return num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type
def get_basis_channels_from_t(model, t):

    assert all(0 <= x <= 1 for x in t), "Values of t must be between 0 and 1"

    in_channels, out_channels, basis_channels = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):

            in_channels.append(module.in_channels)
            out_channels.append(module.out_channels)

            weight = module.weight.data.clone()
            H = weight.view(weight.shape[0], -1)
            [u, s, v_t] = torch.svd(H)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind] ** 2

            c_sum = torch.cumsum(delta, 0)
            c_sum = c_sum / c_sum[-1]
            idx = torch.nonzero(c_sum >= t.pop(0))[0].item()

            basis_channels.append(min(idx+1, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))

    return in_channels, out_channels, basis_channels
def replace_basisconv2d_with_conv2d_(module):
    """
    Recursively replaces all BasisConv2d layers in a module with Conv2d layers.

    Args:
        module (torch.nn.Module): The module whose BasisConv2d layers will be replaced.
    Returns:
        None.
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, BasisConv2d):
            # Replace the Conv2d layer with a BasisConv2d layer
            conv_layer = child_module.combine_conv_f_with_conv_w()
            setattr(module, name, conv_layer)
            # module._modules[name] = basis_layer
        else:
            # Recursively apply the function to the child module
            replace_basisconv2d_with_conv2d_(child_module)
def replace_basisconv2d_with_conv2d(basis_model):
    model = copy.deepcopy(basis_model)
    replace_basisconv2d_with_conv2d_(model)
    return model
def replace_conv2d_with_basisconv2d_(module, basis_channels_list=None, add_bn_list=None):
    """
    Recursively replaces all Conv2d layers in a module with BasisConv2d layers.

    Args:
        module (torch.nn.Module): The module whose Conv2d layers will be replaced.
        basis_channels_list (list, optional): A list of integers specifying the number
            of basis channels for each BasisConv2d layer in the model. If None, the number
            of basis channels will be set to min(out_channels, weight.numel() // weight.size(0)).
        add_bn_list (list, optional): A list of Boolean values specifying whether to add
            batch normalization to each BasisConv2d layer in the model. If None, batch
            normalization will not be added.

    Returns:
        None.
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Conv2d):
            # Replace the Conv2d layer with a BasisConv2d layer
            weight = child_module.weight.data.clone()
            bias = child_module.bias.data.clone() if child_module.bias is not None else None
            if add_bn_list is None:
                add_bn = False
            else:
                add_bn = add_bn_list.pop(0)

            in_channels = child_module.in_channels
            if basis_channels_list is None:
                basis_channels = min(child_module.out_channels, child_module.weight.numel() // child_module.weight.size(0))
            else:
                basis_channels = basis_channels_list.pop(0)

            out_channels = child_module.out_channels
            kernel_size = child_module.kernel_size
            stride = child_module.stride
            padding = child_module.padding
            dilation = child_module.dilation
            groups = child_module.groups

            basis_layer = BasisConv2d(weight, bias, add_bn, in_channels, basis_channels, out_channels, kernel_size, stride, padding, dilation, groups)
            setattr(module, name, basis_layer)
            # module._modules[name] = basis_layer
        else:
            # Recursively apply the function to the child module
            replace_conv2d_with_basisconv2d_(child_module, basis_channels_list, add_bn_list)
def replace_conv2d_with_basisconv2d(model, basis_channels_list=None, add_bn_list=None):
    basis_model = copy.deepcopy(model)
    replace_conv2d_with_basisconv2d_(basis_model, basis_channels_list, add_bn_list)
    return basis_model
class ModelStats:
    def __init__(self):
        self.reset_variables()

    def reset_variables(self):
        self.total_flops = 0
        self.total_params = 0

        self.info = {'org_flops': [], 'basis_flops': [],'org_params': [], 'basis_params': [], 'input_size': [], 'output_size': [], 'layer_name': [],
                'in_channels': [], 'out_channels': [], 'basis_channels': [], 'kernel_size': []}
    def get_stats(self, model, input_size):
        self.reset_variables()

        # Register appropriate hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, BasisConv2d):
                handles.append(module.register_forward_hook(self.basisconv2d_hook))
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(self.linear_hook))

        device = next(model.parameters()).device
        x = torch.rand(1, *input_size).to(device)
        model(x)

        for hd in handles:
            hd.remove()

        return self.total_flops, self.total_params, self.info

    def linear_hook(self, module, input, output):
        input_size = input[0].size()
        output_size = output.size()

        # Calculate the number of FLOPS for a fully connected layer
        layer_flops = input_size[0] * input_size[1] * output_size[1]
        layer_params = sum(p.numel() for p in module.parameters())

        # Add the FLOPS to the total count
        self.total_flops += layer_flops
        self.total_params += layer_params

        self.info['org_flops'].append(layer_flops)
        self.info['basis_flops'].append(None)
        self.info['org_params'].append(layer_params)
        self.info['basis_params'].append(None)
        self.info['input_size'].append(input_size)
        self.info['output_size'].append(output_size)
        self.info['layer_name'].append(module._get_name())
        self.info['in_channels'].append(None)
        self.info['out_channels'].append(None)
        self.info['basis_channels'].append(None)
        self.info['kernel_size'].append(None)

    def basisconv2d_hook(self, module, input, output):
        input_size = input[0].size()
        output_size = output.size()

        in_channels = module.conv_f.in_channels
        basis_channels = module.conv_f.out_channels
        out_channels = module.conv_w.out_channels

        # Calculate the number of FLOPS BasisConv2d layer
        filter_mul = module.conv_f.kernel_size[0] * module.conv_f.kernel_size[1] * in_channels
        filter_add = filter_mul

        org_mul_num = output_size[0] * output_size[1] * (filter_mul + filter_add) * out_channels

        basisconv_mul_num = output_size[0] * output_size[1] * ((filter_mul + filter_add)) * basis_channels
        proj_mul_num = output_size[0] * output_size[1] * (basis_channels + basis_channels) * out_channels

        layer_flops = basisconv_mul_num+proj_mul_num
        layer_params = sum(p.numel() for p in module.parameters())

        self.total_flops += layer_flops
        self.total_params += layer_params

        self.info['org_flops'].append(org_mul_num)
        self.info['basis_flops'].append(layer_flops)
        self.info['org_params'].append(module.conv_f.kernel_size[0] * module.conv_f.kernel_size[1] * in_channels * out_channels)
        self.info['basis_params'].append(layer_params)
        self.info['input_size'].append(input_size)
        self.info['output_size'].append(output_size)
        self.info['layer_name'].append(module._get_name())
        self.info['in_channels'].append(in_channels)
        self.info['out_channels'].append(out_channels)
        self.info['basis_channels'].append(basis_channels)
        self.info['kernel_size'].append(module.conv_f.kernel_size)