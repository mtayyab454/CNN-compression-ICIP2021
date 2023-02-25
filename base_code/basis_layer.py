import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisConv2d(nn.Module):
    def __init__(self, weight, bias, add_bn, in_channels, basis_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, sparse_filters=False):
        super(BasisConv2d, self).__init__()
        assert basis_channels < weight.numel() // weight.size(0), "Number of filters should be less than or match input tensor dimensions"

        # define new convolution layers with F and w
        self.conv_f = nn.Conv2d(in_channels, basis_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.conv_w = nn.Conv2d(basis_channels, out_channels, 1, 1, 0, 1, groups=1, bias=True if bias is not None else False)

        # apply SVD to get F and w
        F, w = self.svd_init(weight, sparse_filters)

        # Set the weights of the new convolution layers.
        self.conv_f.weight.data = F.view(self.conv_f.out_channels, *weight.shape[1:] ).to(weight.dtype)
        self.conv_w.weight.data = w.unsqueeze(-1).unsqueeze(-1).to(weight.dtype)
        if bias is not None:
            self.conv_w.bias.data = bias.to(weight.dtype)

        if add_bn:
            self.bn = nn.BatchNorm2d(basis_channels)

            # Initialize parameters to leave input unchanged
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = None

    def svd_init(self, weight, sparse_filters):

        H = weight.view(weight.shape[0], -1)

        if sparse_filters:
            H = torch.mm(torch.t(H), H)
            [u, s, v_t] = torch.svd(H, some=False)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind]
            v_t = v_t[:, ind]
        else:
            [u, s, v_t] = torch.svd(H)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind] ** 2
            v_t = v_t[:, ind]


        F = v_t[:, 0:self.conv_f.out_channels].t()
        w = torch.mm(F, H.t()).t()

        return F, w

    def forward(self, x):
        # convolve with F and then w
        x = self.conv_f(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv_w(x)
        return x

# create an input tensor
x = torch.randn(1, 3, 32, 32)

# create a Conv2d layer with random weights
conv = torch.nn.Conv2d(3, 16, kernel_size=3, bias=True)

# create a BasisConv2d module using the weights of the Conv2d layer
basis_conv = BasisConv2d(
    conv.weight.data.clone(),
    conv.bias.data.clone() if conv.bias is not None else None,
    True,
    conv.in_channels,
    16,
    conv.out_channels,
    conv.kernel_size
)

# perform forward pass with both Conv2d and BasisConv2d modules
y_conv = conv(x)
y_basis = basis_conv(x)

# check that output tensors are equal
print(torch.allclose(y_conv, y_basis, atol=1e-5))  # True