import torch
import torch.nn as nn
import copy
import math
from base_code.basis_layer import basisConv2d, basisLinear

def display_stats(basis_model, model, exp_name, input_size, count_relu=False):
    # if input_size is None:
    #     input_size = [32, 32]
    print_text = ''
    print_text = print_text + '\n############################################# '+ exp_name +' #############################################\n'
    # print(exp_name)
    if hasattr(model, 'features'):
        num_model_conv_param = sum(p.numel() for p in model.features.parameters())
        num_basis_conv_param = sum(p.numel() for p in basis_model.model.features.parameters())
    else:
        if hasattr(model, 'fc'):
            num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.fc.parameters())
            num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.fc.parameters())
        elif hasattr(model, 'linear'):
            num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.linear.parameters())
            num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.linear.parameters())
        elif hasattr(model, 'classifier'):
            num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.classifier.parameters())
            num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.classifier.parameters())
        else:
            print_text = print_text + 'Verify conv the parameters!!!'
            model_fc_param = 0
            for m in model.children():
                if isinstance(m, nn.Linear):
                    temp = sum(p.numel() for p in m.parameters())
                    model_fc_param += temp

            basis_fc_param = 0
            for m in basis_model.model.children():
                if isinstance(m, nn.Linear) or isinstance(m, basisLinear):
                    temp = sum(p.numel() for p in m.parameters())
                    basis_fc_param += temp

            num_model_conv_param = sum(p.numel() for p in model.parameters()) - model_fc_param
            num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - basis_fc_param

    num_model_param = sum(p.numel() for p in model.parameters())
    num_basis_param = sum(p.numel() for p in basis_model.model.parameters())

    # num_basis_conv_param = num_basis_conv_param - (basis_model.model.features[14].basis_weight.numel() + basis_model.model.features[21].basis_weight.numel() + basis_model.model.features[28].basis_weight.numel())

    org_flops, basis_flops, info = basis_model.get_flops(input_size, count_relu=False)
    print_text = print_text + '\n' + '    Model FLOPs: %.2fM' % (org_flops / 10**6)
    print_text = print_text + '\n' + '    Basis Model FLOPs: %.2fM' % (basis_flops / 10**6)
    print_text = print_text + '\n' + '    %% Reduction in FLOPs: %.2f %%' % (100 - (basis_flops*100/org_flops))
    print_text = print_text + '\n' + '    %% Speedup: %.2f %%\n' % ( org_flops/basis_flops )

    print_text = print_text + '\n' + '    Model Conv Params: %.2fM' % (num_model_conv_param / 10**6)
    print_text = print_text + '\n' + '    Basis Model Conv params: %.2fM' % (num_basis_conv_param / 10**6)
    print_text = print_text + '\n' + '    %% Reduction in Conv params: %.2f %%\n' % (100 - (num_basis_conv_param * 100 / num_model_conv_param) )

    print_text = print_text + '\n' + '    Model Total Params: %.2fM' % (num_model_param / 10**6)
    print_text = print_text + '\n' + '    Basis Model Total params: %.2fM' % (num_basis_param / 10**6)
    print_text = print_text + '\n' + '    %% Reduction in Total params: %.2f %%\n' % (100 - (num_basis_param * 100 / num_model_param) )

    org_filters, basis_filters = sum(basis_model.num_original_filters.tolist()), sum(basis_model.num_basis_filters.tolist())
    print_text = print_text + '\n' + '    Model Filters: %d' % (org_filters)
    print_text = print_text + '\n' + '    Basis Model Filters: %d' % (basis_filters)
    print_text = print_text + '\n' + '    %% Reduction in Filters: %.2f %%\n' % (100 - (basis_filters*100.0/org_filters))

    print_text = print_text + '\n' + '    Model Accuracy: '
    print_text = print_text + '\n' + '    Basis Model Accuracy: '
    print_text = print_text + '\n' + '    Reduction in Accuracy: \n'

    print_text = print_text + '\n' + '    Filter in original convs: ' + str(basis_model.num_original_filters.tolist())
    print_text = print_text + '\n' + '    Filter in original basis convs: ' + str(basis_model.num_basis_filters.tolist())
    print_text = print_text + '\n' + '\n#########################################################################################################\n'

    stats = {'print_stats':print_text, 'exp_name':exp_name, 'flops_info':info, 'org_flops': org_flops, 'basis_flops': basis_flops, 'model_conv_param': num_model_conv_param, 'basis_conv_param':num_basis_conv_param,
             'model_param':num_model_param, 'basis_param':num_basis_param, 'org_filters': org_filters, 'basis_filters': basis_filters,
             'org_filters_vec': basis_model.num_original_filters.tolist(), 'basis_filters_vec': basis_model.num_basis_filters.tolist()};
    # stats = fromarrays([exp_name, input_size, org_flops, basis_flops, num_model_conv_param, num_basis_conv_param, num_model_param, num_basis_param, org_filters, basis_filters],
    #             names=['exp_name', 'input_size', 'org_flops', 'basis_flops', 'model_conv_param', 'basis_conv_param', 'model_param', 'basis_param', 'org_filters', 'basis_filters'])
    print(print_text)
    return stats

# def display_stats(basis_model, model, exp_name, input_size, count_relu=False):
#     # if input_size is None:
#     #     input_size = [32, 32]
#     print_text = ''
#     print_text + '\n############################################# '+ exp_name +' #############################################\n'
#     # print(exp_name)
#     if hasattr(model, 'features'):
#         num_model_conv_param = sum(p.numel() for p in model.features.parameters())
#         num_basis_conv_param = sum(p.numel() for p in basis_model.model.features.parameters())
#     else:
#         if hasattr(model, 'fc'):
#             num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.fc.parameters())
#             num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.fc.parameters())
#         elif hasattr(model, 'linear'):
#             num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.linear.parameters())
#             num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.linear.parameters())
#         elif hasattr(model, 'classifier'):
#             num_model_conv_param = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.classifier.parameters())
#             num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - sum(p.numel() for p in basis_model.model.classifier.parameters())
#         else:
#             print('Verify conv the parameters!!!')
#             model_fc_param = 0
#             for m in model.children():
#                 if isinstance(m, nn.Linear):
#                     temp = sum(p.numel() for p in m.parameters())
#                     model_fc_param += temp
#
#             basis_fc_param = 0
#             for m in basis_model.model.children():
#                 if isinstance(m, nn.Linear) or isinstance(m, basisLinear):
#                     temp = sum(p.numel() for p in m.parameters())
#                     basis_fc_param += temp
#
#             num_model_conv_param = sum(p.numel() for p in model.parameters()) - model_fc_param
#             num_basis_conv_param = sum(p.numel() for p in basis_model.model.parameters()) - basis_fc_param
#
#     num_model_param = sum(p.numel() for p in model.parameters())
#     num_basis_param = sum(p.numel() for p in basis_model.model.parameters())
#
#     # num_basis_conv_param = num_basis_conv_param - (basis_model.model.features[14].basis_weight.numel() + basis_model.model.features[21].basis_weight.numel() + basis_model.model.features[28].basis_weight.numel())
#
#     org_flops, basis_flops, info = basis_model.get_flops(input_size, count_relu=False)
#     print('    Model FLOPs: %.2fM' % (org_flops / 10**6))
#     print('    Basis Model FLOPs: %.2fM' % (basis_flops / 10**6))
#     print('    %% Reduction in FLOPs: %.2f %%' % (100 - (basis_flops*100/org_flops)))
#     print('    %% Speedup: %.2f %%\n' % ( org_flops/basis_flops ))
#
#     print('    Model Conv Params: %.2fM' % (num_model_conv_param / 10**6))
#     print('    Basis Model Conv params: %.2fM' % (num_basis_conv_param / 10**6))
#     print('    %% Reduction in Conv params: %.2f %%\n' % (100 - (num_basis_conv_param * 100 / num_model_conv_param) ))
#
#     print('    Model Total Params: %.2fM' % (num_model_param / 10**6))
#     print('    Basis Model Total params: %.2fM' % (num_basis_param / 10**6))
#     print('    %% Reduction in Total params: %.2f %%\n' % (100 - (num_basis_param * 100 / num_model_param) ))
#
#     org_filters, basis_filters = sum(basis_model.num_original_filters.tolist()), sum(basis_model.num_basis_filters.tolist())
#     print('    Model Filters: %d' % (org_filters))
#     print('    Basis Model Filters: %d' % (basis_filters))
#     print('    %% Reduction in Filters: %.2f %%\n' % (100 - (basis_filters*100.0/org_filters)))
#
#     print('    Model Accuracy: ')
#     print('    Basis Model Accuracy: ')
#     print('    Reduction in Accuracy: \n')
#
#     print('    Filter in original convs: ', basis_model.num_original_filters.tolist())
#     print('    Filter in original basis convs: ', basis_model.num_basis_filters.tolist())
#     print('\n#########################################################################################################\n')
#
#     stats = {'exp_name':exp_name, 'flops_info':info, 'org_flops': org_flops, 'basis_flops': basis_flops, 'model_conv_param': num_model_conv_param, 'basis_conv_param':num_basis_conv_param,
#              'model_param':num_model_param, 'basis_param':num_basis_param, 'org_filters': org_filters, 'basis_filters': basis_filters,
#              'org_filters_vec': basis_model.num_original_filters.tolist(), 'basis_filters_vec': basis_model.num_basis_filters.tolist()};
#     # stats = fromarrays([exp_name, input_size, org_flops, basis_flops, num_model_conv_param, num_basis_conv_param, num_model_param, num_basis_param, org_filters, basis_filters],
#     #             names=['exp_name', 'input_size', 'org_flops', 'basis_flops', 'model_conv_param', 'basis_conv_param', 'model_param', 'basis_param', 'org_filters', 'basis_filters'])
#     return stats

def trace_model(model):
    in_channels = []
    out_channels = []
    basis_channels = []
    layer_type = []

    for n, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d):

            in_channels.append(m.in_channels)
            out_channels.append(m.out_channels)
            basis_channels.append( min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]) )
            layer_type.append('conv')

        elif isinstance(m, nn.Linear):

            in_channels.append(m.in_features)
            out_channels.append(m.out_features)
            basis_channels.append(min(m.out_features, m.in_features))
            layer_type.append('linear')

    return in_channels, out_channels, basis_channels, layer_type

def replace_layer(module, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters, count=0):

    for n, m in list(module.named_children()):
        if isinstance(m, nn.Conv2d):

            if replace_fc[count]:
                print('Processing conv ', count)
                basis_channels = min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
                module._modules[n] = basisConv2d(m, basis_channels, use_weights[count], add_bn[count], trainable_basis[count], sparse_filters)
                ## Verify that the reconstruction is correct
                # x = torch.randn(4, m.in_channels, 9, 9)
                # o1 = m(x)
                # o2 = module._modules[n](x)
                #
                # print((o1-o2).abs().mean())
            else:
                print('Skipping conv ', count)
            count += 1
        elif isinstance(m, nn.Linear):

            if replace_fc[count]:
                print('Processing linear ', count)
                basis_channels = min(m.out_features, m.in_features)
                module._modules[n] = basisLinear(m, basis_channels, use_weights[count], add_bn[count], trainable_basis[count], sparse_filters)
                ## Verify that the reconstruction is correct
                # x = torch.randn(4, m.in_features)
                # o1 = m(x)
                # o2 = module._modules[n](x)
                #
                # print((o1-o2).abs().mean())
            else:
                print('Skipping linear ', count)
            count += 1
        else:
            count = replace_layer(m, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters, count)

    return count

class baseModel(nn.Module):
    def __init__(self, model, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters):
        super(baseModel, self).__init__()

        self.model = copy.deepcopy(model)
        _, original_channels, basis_channels, layer_type = trace_model(self.model)
        num_layers = len(basis_channels)

        use_weights = [use_weights] * num_layers
        add_bn = [add_bn] * num_layers
        trainable_basis = [trainable_basis]*num_layers

        if not isinstance(replace_fc, list):
            if replace_fc is False:
                replace_fc = [True] * sum((x == 'conv') for x in layer_type) + [False] * sum((x == 'linear') for x in layer_type)
            elif replace_fc is True:
                replace_fc = [True] * num_layers

        replace_layer(self.model, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters)
        ## Till this point all lists () lenght is equal to num_layers

        ## Remove entries for the layers which are not replaced
        pruned_basis_channels = []
        pruned_original_channels = []
        for i, n in enumerate(replace_fc):
            if n is True:
                pruned_basis_channels.append(basis_channels[i])
                pruned_original_channels.append(original_channels[i])

        ## Save all variables
        self.register_buffer('num_original_filters', torch.IntTensor(pruned_original_channels))
        self.register_buffer('num_basis_filters', torch.IntTensor(pruned_basis_channels))
        self.register_buffer('num_layers', torch.IntTensor([len(pruned_basis_channels)]))

    def cuda(self, device=None):
        super(baseModel, self).cuda(device)
        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.cpu()
                m.cuda()

    def cpu(self):
        super(baseModel, self).cpu()

    def update_channels_coefficients(self, th_T):

        count = 0
        for m in self.model.modules():
            if isinstance(m, basisConv2d):
                c_data = m.coefficients.data.clone()
                c_data_sum = c_data.sum(1).abs()
                c_data_sum, ind = torch.sort(c_data_sum, descending=True)

                c_sum = torch.cumsum(c_data_sum, 0)
                c_sum = c_sum / c_sum[-1]

                idx = torch.nonzero(c_sum >= th_T)[0]

                filter_to_keep = ind[:idx.item()].tolist()
                self.num_basis_filters[count] = torch.IntTensor([len(filter_to_keep)])
                m.update_channels_coefficients(filter_to_keep)

                count += 1

        print('update_channels: Model compression is updated to', th_T)

    def update_channels(self, th_T):
        #th_T is either a float value {0 - 1}
        # Or a list of float values {0 - 1}
        # Or a list of Intigers

        # th_T is a float value {0 - 1}
        if not isinstance(th_T, list):
            count = 0
            for m in self.model.modules():
                if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                    c_sum = torch.cumsum(m.delta, 0)
                    c_sum = c_sum / c_sum[-1]
                    idx = torch.nonzero(c_sum >= th_T)[0]
                    self.num_basis_filters[count] = torch.IntTensor([idx[0] + 1])
                    m.update_channels(idx[0] + 1)

                    count += 1

            print('update_channels: Model compression is updated to', th_T)
        else:
            if len(th_T) == self.num_layers:

                ###############################################################################################
                # th_T is a list of float values {0 - 1}
                if all( (x >= 0.0 and x <= 1.0) for x in th_T):
                    count = 0
                    self.num_basis_filters = torch.IntTensor(th_T)
                    for m in self.model.modules():
                        if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                            c_sum = torch.cumsum(m.delta, 0)
                            c_sum = c_sum / c_sum[-1]
                            idx = torch.nonzero(c_sum >= th_T[count])[0]
                            self.num_basis_filters[count] = torch.IntTensor([idx[0] + 1])
                            m.update_channels(idx[0] + 1)
                            count += 1

                    print('update_channels: Model compression is updated to', th_T)
                ###############################################################################################
                # th_T is list of Intigers
                elif all(isinstance(x, int) for x in th_T):
                    count = 0
                    num_basis_filters_local = []
                    for m in self.model.modules():
                        if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                            nbf = m.update_channels(th_T[count])
                            num_basis_filters_local.append(nbf)
                            count += 1
                    self.num_basis_filters = torch.IntTensor(num_basis_filters_local)
                    print('update_channels: Model compression is updated to', th_T)
                else:
                    raise ValueError('update_channels: Cannot mix Floats {0.0 to 1.0 } and Ints')
            else:
                raise ValueError('update_channels: len(th_T) is not equal to num_layers')

    # def load_state_dict(self, state_dict, strict=True):
    #
    #     # count = 0
    #     # for m in self.model.modules():
    #     #     if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
    #     #         m.update_channels(state_dict['num_basis_filters'][count].item())
    #     #         # m.load_state_dict(state_dict, strict)
    #     #         count += 1
    #     self.update_channels(state_dict['num_basis_filters'].cpu().tolist())
    #     super(baseModel, self).load_state_dict(state_dict, strict)
    #     print('load_state_dict: Model compression is updated')

    def randomize_filters(self):

        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.randomize_filters()

    def basis_parameters(self):
        basis_param = []
        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                basis_param.extend(m.parameters())

        return basis_param

    def get_flops(self, input_size, count_relu):

        info = {'org_flops': [], 'basis_flops': [], 'input_size': [], 'output_size': [], 'layer_name': [],
                'in_channels': [], 'out_channels': [], 'basis_channels': [], 'kernel_size': []}

        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.flops_flag = True

        device = next(self.parameters()).device
        in_ch = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, basisConv2d):
                in_ch = m.in_channels
                break
        x = torch.rand(1, in_ch, *input_size).to(device)

        train_mode = self.training
        self.eval()

        self.model(x)
        self.train(train_mode)

        org_flops = 0
        basis_flops = 0
        for n, m in self.model.named_modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.flops_flag = False

                numel_out = (m.flops[3][0] * m.flops[3][1] * m.out_channels) if isinstance(m, basisConv2d) else m.out_features
                mul_factor = 0
                # Batchnorm flops can be ignored because batchnorm can be combined with preceding conv layer
                # if m.add_bn: mul_factor += 1
                if count_relu: mul_factor = 1

                c_org_flops = m.flops[0] + (mul_factor * numel_out)
                c_basis_flops = m.flops[1] + (mul_factor * numel_out)

                org_flops += c_org_flops
                basis_flops += c_basis_flops

                info['org_flops'].append(c_org_flops)
                info['basis_flops'].append(c_basis_flops)
                info['input_size'].append(m.flops[2])
                info['output_size'].append(m.flops[3])
                info['layer_name'].append('model.'+n)

                info['in_channels'].append(m.in_channels if isinstance(m, basisConv2d) else m.in_features)
                info['out_channels'].append(m.out_channels if isinstance(m, basisConv2d) else m.out_features)
                info['basis_channels'].append(m.basis_channels.item() if isinstance(m, basisConv2d) else m.basis_features.item())
                info['kernel_size'].append(m.kernel_size if isinstance(m, basisConv2d) else [])

        return org_flops, basis_flops, info

    def forward(self, x, augment=None):
        if augment == None:
            x = self.model(x)
        else:
            x = self.model(x, augment)
        return x

class basisModel(baseModel):

    def __init__(self, model, use_weights, add_bn, trainable_basis, replace_fc = False, sparse_filters = False):
        super(basisModel, self).__init__(model, use_weights, add_bn, trainable_basis, replace_fc, sparse_filters)


# import torchvision.models as models
# m = models.vgg16(pretrained=True)
# em = basisModel(m, True, False, False)
# em.update_channels(1.0)
#
# x = torch.randn(1, 3, 224, 224)
# o1 = m(x)
# o2 = em(x)
# print((o1-o2).abs().sum())
#
# org_flops, basis_flops, info = em.get_flops([224, 224], False)
# print(org_flops/basis_flops)
