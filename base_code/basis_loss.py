import torch
from .basis_layer import BasisConv2d
import torch.nn.functional as F
import torch.nn as nn

class ConvW_L1Loss(nn.Module):
    def __init__(self, skip_1by1=False):
        super(ConvW_L1Loss, self).__init__()
        self.skip_1by1 = skip_1by1

    def forward(self, basis_model):
        loss_m = 0
        # loss_v = []

        count = 0
        for name, module in list(basis_model.named_modules()):
            if isinstance(module, BasisConv2d):
                if self.skip_1by1 == False or module.conv_f.weight.shape[-1] > 1:
                    weights = module.conv_w.weight.data.clone()
                    curr_loss = weights.abs().mean()
                    loss_m += curr_loss
                    count += 1
                    # loss_v.append(curr_loss.item())

        # print(loss_v)
        # print(loss_c.item(), loss_m.item())
        loss_m = loss_m / count

        return loss_m

class ConvF_OrthoLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ConvF_OrthoLoss, self).__init__()
        self.alpha = alpha # initilize alpha

    def forward(self, basis_model):
        loss_m = 0
        # loss_v = []

        for name, module in list(basis_model.named_modules()):
            if isinstance(module, BasisConv2d):
                weights = module.conv_f.weight.data.clone().view(module.conv_f.weight.shape[0], -1).t()
                # Normalize weights
                weights = F.normalize(weights, 2, 0)
                Q = weights.shape[1]

                if Q == 1:
                    continue

                dot_prod = torch.mm(weights.t(), weights).abs()

                dp_diag = (1 - dot_prod.diag() ** 2).sum()
                dp_else = (dot_prod.triu(1) ** 2).sum()

                curr_loss = (self.alpha * dp_diag / Q) + (2 * (1 - self.alpha)) / (Q * (Q - 1)) * dp_else
                loss_m += curr_loss
                # loss_v.append(curr_loss.item())

        # print(loss_v)
        # print(loss_c.item(), loss_m.item())
        return loss_m

class BasisCombinationLoss(nn.Module):
    def __init__(self, l1_w, ortho_w, skip_1by1, alpha=0.5, ce_reduction='mean'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction=ce_reduction)
        self.l1 = ConvW_L1Loss(skip_1by1)
        self.l2 = ConvF_OrthoLoss(alpha)
        self.l1_w = l1_w
        self.ortho_w = ortho_w

    def forward(self, basis_model, outputs, targets):

        loss_ce = self.ce(outputs, targets)
        loss_l1 = torch.zeros(1, device=loss_ce.device)
        loss_l2 = torch.zeros(1, device=loss_ce.device)

        if self.l1_w > 0:
            loss_l1 = self.l1(basis_model)
        if self.ortho_w > 0:
            loss_l2 = self.l2(basis_model)

        loss = self.l1_w * loss_l1 + self.ortho_w * loss_l2 + loss_ce

        return loss, loss_ce, loss_l1, loss_l2