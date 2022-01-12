import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import math

def comp_dct3_basis(M, N, O, P):
    mat = torch.zeros([P, M, N, O])
    count = 0
    for m in range(M):
        for n in range(N):
            for o in range(O):
                mat[count, :, :, :] = get_basis(m, n, o, M, N, O)
                count = count + 1

                if count == P:
                    return mat


    return mat

def get_basis(m, n, o, M, N, O):
    mat = torch.zeros([M, N, O])

    for p in range(M):
        for q in range(N):
            for r in range(O):
                if m == 0:
                    Ap = 1/math.sqrt(M)
                else:
                    Ap = math.sqrt(2/M)

                if n == 0:
                    Aq = 1/math.sqrt(N)
                else:
                    Aq = math.sqrt(2/N)

                if o == 0:
                    Ar = 1/math.sqrt(O)
                else:
                    Ar = math.sqrt(2/O)

                temp = math.cos( math.pi*(2*p + 1)*m/(2*M) ) * math.cos( math.pi*(2*q + 1)*n/(2*N) ) * math.cos( math.pi*(2*r + 1)*o/(2*O) )
                mat[p, q, r] = Ap * Aq * Ar * temp

    return mat

def get_rip(n, m, th):

    mat = torch.FloatTensor(n, m)
    mat[:, 0] = torch.randn(n)

    for i in range(1, m):

        done = False
        while not done:
            x = torch.FloatTensor(np.random.randn(n))
            x = x/x.norm()
            v = mat.t().matmul(x)
            vmax = v.abs().max()

            if vmax is None:
                print('wtf!')

            if vmax < th:
                mat[:, i] = x
                # print(i)
                done = True

    return mat

def calc_eig(X_org, sparse_filters):

    if not sparse_filters:
        X = X_org.view(X_org.shape[0], -1)
        [u, s, v] = torch.svd(X)
        _, ind = torch.sort(s, descending=True)
        delta = s[ind]**2
        phi = v[:, ind]

    else:
        X = X_org.view(X_org.shape[0], -1)
        X = torch.mm(torch.t(X), X)
        [u, s, v] = torch.svd(X, some=False)
        _, ind = torch.sort(s, descending=True)
        delta = s[ind]
        phi = v[:, ind]

    # X = X_org.view(X_org.shape[0], -1)
    # X = torch.mm(torch.t(X), X)
    # delta, phi = torch.eig(X, eigenvectors=True)
    # _, ind = torch.sort(delta[:, 0], descending=True)
    #
    # delta = delta[ind, 0]
    # phi = phi[:, ind]

    return phi, delta

# https://stackoverflow.com/questions/53875821/scipy-generate-nxn-discrete-cosine-matrix
class basisConv2d(nn.Module):
    def __init__(self, org_conv, basis_channels, use_weights=True, add_bn = True, trainable_basis=True, sparse_filters=False):
        super(basisConv2d, self).__init__()

        conv = copy.deepcopy(org_conv)
        conv.cpu()

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.add_bn = add_bn
        # self.basis_bias = None

        self.flops_flag = False
        self.flops = [0, 0, 0, 0] # [org_flops, basis_flops, input_size, output_size]

        if self.add_bn == True:
            self.bn = nn.BatchNorm2d(basis_channels)

        self.register_buffer('basis_channels', torch.IntTensor([basis_channels]))
        self.coefficients = nn.Parameter(torch.Tensor(self.basis_channels.item(), self.out_channels)).type_as(conv.weight.data)

        ######## Should I make bias a parameter too? ########
        self.register_parameter('basis_bias', None)
        if conv.bias is not None:
            self.register_parameter('bias', nn.Parameter(torch.Tensor(self.out_channels).type_as(conv.weight.data) ))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('org_weight', torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size).type_as(conv.weight.data) )
        if trainable_basis:
            self.register_parameter('basis_weight',  nn.Parameter(torch.Tensor(self.basis_channels.item(), self.in_channels // self.groups, *self.kernel_size).type_as(conv.weight.data)))
        else:
            self.register_buffer('basis_weight', torch.Tensor(self.basis_channels.item(), self.in_channels // self.groups, *self.kernel_size).type_as(conv.weight.data) )

        self.register_buffer('phi', torch.Tensor(self.kernel_size[0] * self.kernel_size[1] * self.in_channels, self.kernel_size[0] * self.kernel_size[1] * self.in_channels).type_as(conv.weight.data) )
        self.register_buffer('delta', torch.Tensor(self.kernel_size[0] * self.kernel_size[1] * self.in_channels).type_as(conv.weight.data) )

        if use_weights:

            if conv.bias is not None:
                self.bias.data = conv.bias.data

            self.org_weight = conv.weight.data
            #################### Calculate Basis (Eigen) #######################
            self.phi, self.delta = calc_eig(self.org_weight, sparse_filters)
            ############################################################

            self.update_channels(self.basis_channels.item())

    def cuda(self, device=None):
        self.coefficients.data = self.coefficients.data.cuda()
        self.basis_weight.data = self.basis_weight.data.cuda()
        if self.add_bn == True:
            self.bn.cuda()
        if self.bias is not None:
            self.bias.data = self.bias.data.cuda()
        if self.basis_bias is not None:
            self.basis_bias.data = self.basis_bias.data.cuda()

    def update_channels_coefficients(self, basis_channels): # use it carefully !!!
        self.basis_channels = torch.IntTensor([len(basis_channels)])

        basis_shape = self.basis_weight.shape
        new_basis_weight = torch.zeros([len(basis_channels), basis_shape[1], basis_shape[2], basis_shape[3]]).type_as(self.basis_weight.data)
        new_coefficients = torch.zeros([len(basis_channels), self.coefficients.shape[1]]).type_as(self.coefficients.data)
        new_basis_bias = torch.zeros([len(basis_channels)]).type_as(self.basis_bias.data)

        count = 0
        for bc in basis_channels:
            new_basis_weight[count, :] = self.basis_weight.data[bc, :]
            new_coefficients[count, :] = self.coefficients.data[bc, :]
            new_basis_bias[count] = self.basis_bias.data[bc]
            count += 1

        self.basis_weight.data = new_basis_weight
        self.coefficients.data = new_coefficients
        self.basis_bias.data = new_basis_bias

    def update_channels(self, basis_channels): # use it carefully !!!

        if basis_channels > self.phi.shape[0]:
            basis_channels = self.phi.shape[0]

        self.basis_channels = torch.IntTensor([basis_channels])

        ef = self.phi[:, 0:self.basis_channels.item()].t()
        self.coefficients = torch.nn.Parameter(torch.mm(ef, self.org_weight.view(self.out_channels, -1).t()).type_as(self.coefficients.data) )
        self.coefficients.data = self.coefficients.data.t().unsqueeze(2).unsqueeze(3)
        self.basis_weight.data = ef.view(self.basis_channels.item(), *self.org_weight.shape[1:]).type_as(self.basis_weight.data)
        if self.add_bn is True:
            device = self.basis_weight.data.device
            self.bn = nn.BatchNorm2d(self.basis_channels.item()).to(device)

        return basis_channels

    def randomize_filters(self):

        self.coefficients.data = torch.randn(self.coefficients.shape).type_as(self.coefficients)

    def calc_flops(self, input_size, output_size):
        filter_mul = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        filter_add = filter_mul
        org_mul_num = output_size[0] * output_size[1] * (filter_mul + filter_add) * self.out_channels
        bconv_mul_num = output_size[0] * output_size[1] * ((filter_mul + filter_add)) * self.basis_channels.item()
        proj_mul_num = output_size[0] * output_size[1] * (self.basis_channels.item() + self.basis_channels.item()) * self.out_channels

        self.flops = [org_mul_num, bconv_mul_num+proj_mul_num, input_size, output_size]

    # def forward(self, x): # using 1by1 convolution
    #
    #     if self.flops_flag:
    #         input_size = x.shape[2:]
    #
    #     x = F.conv2d(x, self.basis_weight, None, self.stride, self.padding, self.dilation, self.groups)
    #     if self.add_bn == True:
    #         x = self.bn(x)
    #
    #     x = F.conv2d(x, self.coefficients.t().unsqueeze(2).unsqueeze(3), self.bias, 1, 0, self.dilation, self.groups)
    #
    #     if self.flops_flag:
    #         self.calc_flops(list(input_size), list(x.shape[2:]))
    #
    #     return x

    def forward(self, x, matmul_flag = False):

        if matmul_flag:
            if self.flops_flag:
                input_size = x.shape[2:]

            x = F.conv2d(x, self.basis_weight, self.basis_bias, self.stride, self.padding, self.dilation, self.groups)

            if self.add_bn == True:
                x = self.bn(x)

            x_shape = x.shape
            x = x.view(x_shape[0], self.basis_channels.item(), -1)

            x = torch.matmul(self.coefficients.t(), x)
            x = x.view(x_shape[0], self.out_channels, x_shape[2], x_shape[3])

            if self.bias is not None:
                x = x.transpose(1, 3) + self.bias
                x = x.transpose(3, 1)

            if self.flops_flag:
                self.calc_flops(list(input_size), list(x.shape[2:]))

        else:
            if self.flops_flag:
                input_size = x.shape[2:]

            x = F.conv2d(x, self.basis_weight, self.basis_bias, self.stride, self.padding, self.dilation, self.groups)

            if self.add_bn == True:
                x = self.bn(x)

            # x = F.conv2d(x, self.coefficients.t().unsqueeze(2).unsqueeze(3), self.bias, 1, 0, self.dilation, self.groups)
            x = F.conv2d(x, self.coefficients, self.bias, 1, 0, self.dilation, self.groups)

            if self.flops_flag:
                self.calc_flops(list(input_size), list(x.shape[2:]))

        return x

class basisLinear(nn.Module):
    def __init__(self, org_linear, basis_features, use_weights=True, add_bn = True, trainable_basis=True, sparse_filters=False):
        super(basisLinear, self).__init__()

        linear = copy.deepcopy(org_linear)
        linear.cpu()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.add_bn = add_bn
        self.trainable_basis = trainable_basis

        self.flops_flag = False
        self.flops = [0, 0, 0, 0] # [org_flops, basis_flops, input_size, output_size]

        if self.add_bn == True:
            self.bn = nn.BatchNorm1d(basis_features)

        self.register_buffer('basis_features', torch.IntTensor([basis_features]))
        self.coefficients = nn.Parameter(torch.Tensor(self.basis_features.item(), self.out_features)).type_as(linear.weight.data)

        ######## Should I make bias a parameter too? ########
        if linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(torch.Tensor(self.out_features).type_as(linear.weight.data) ))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('org_weight', torch.Tensor(self.out_features, self.in_features).type_as(linear.weight.data) )
        # self.register_buffer('basis_weight', torch.Tensor(self.basis_features.item(), self.in_features).type_as(linear.weight.data) )
        if trainable_basis:
            self.register_parameter('basis_weight', nn.Parameter(torch.Tensor(self.basis_features.item(), self.in_features).type_as(linear.weight.data) ))
        else:
            self.register_buffer('basis_weight', torch.Tensor(self.basis_features.item(), self.in_features).type_as(linear.weight.data) )

        self.register_buffer('phi', torch.Tensor(self.in_features, self.in_features).type_as(linear.weight.data) )
        self.register_buffer('delta', torch.Tensor(self.in_features).type_as(linear.weight.data) )

        if use_weights:

            if linear.bias is not None:
                self.bias.data = linear.bias.data

            self.org_weight = linear.weight.data
            #################### Calculate (Basis) Eigen #######################
            self.phi, self.delta = calc_eig(self.org_weight, sparse_filters)
            ############################################################

            self.update_channels(self.basis_features.item())

    def cuda(self, device=None):
        self.coefficients.data = self.coefficients.data.cuda()
        self.basis_weight.data = self.basis_weight.data.cuda()
        if self.add_bn == True:
            self.bn.cuda()
        if self.bias is not None:
            self.bias.data = self.bias.data.cuda()

    def update_channels(self, basis_features): # use it carefully !!!
        self.basis_features = torch.IntTensor([basis_features])

        ef = self.phi[:, 0:self.basis_features.item()].t()
        self.coefficients = torch.nn.Parameter(torch.mm(ef, self.org_weight.t()).type_as(self.coefficients.data) )
        self.basis_weight.data = ef.type_as(self.basis_weight.data)
        if self.add_bn is True:
            device = self.basis_weight.data.device
            self.bn = nn.BatchNorm1d(self.basis_features.item()).to(device)

    def randomize_filters(self):

        self.coefficients.data = torch.randn(self.coefficients.shape).type_as(self.coefficients)

    def calc_flops(self, input_size, output_size):
        org_mul_num = 2 * self.in_features * self.out_features
        bconv_mul_num = 2 * self.in_features * self.basis_features.item()
        proj_mul_num = 2 * self.basis_features.item() * self.out_features

        self.flops = [org_mul_num, bconv_mul_num+proj_mul_num, input_size, output_size]

    def forward(self, x):

        if self.flops_flag:
            input_size = x.shape[1]

        x = F.linear(x, self.basis_weight, None)
        if self.add_bn == True:
            x = self.bn(x)

        x = F.linear(x, self.coefficients.t(), self.bias)

        if self.flops_flag:
            self.calc_flops(input_size, x.shape[1])

        return x

# l = nn.Linear(128, 110)
# el = basisLinear(l, 110, True, False)
# el.update_channels(110)
#
# x = torch.randn(4, 128)
# o1 = l(x)
# o2 = el(x)
#
# print((o1-o2).abs().sum())

# c = nn.Conv2d(3, 16, 3)
# ec = basisConv2d(c, 16, True, False)
# ec.update_channels(16)
#
# x = torch.randn(1, 3, 8, 8)
# o1 = c(x)
# o2 = ec(x, True)
#
# print((o1-o2).abs().sum())


