import torch
from base_code.basis_layer import basisConv2d
import torch.nn.functional as F
import torch.nn as nn

class basisL1Loss(nn.Module):
    def __init__(self, skip_1by1=False):
        super(basisL1Loss, self).__init__()
        self.skip_1by1 = skip_1by1

    def forward(self, basis_model):
        loss_m = 0
        # loss_v = []

        count = 0
        for n, m in list(basis_model.named_modules()):
            if isinstance(m, basisConv2d):
                if self.skip_1by1 == False or m.org_weight.shape[-1] > 1:
                    weights = m.coefficients
                    curr_loss = weights.abs().mean()
                    loss_m += curr_loss
                    count += 1
                    # loss_v.append(curr_loss.item())

        # print(loss_v)
        # print(loss_c.item(), loss_m.item())
        loss_m = loss_m / count

        return loss_m

class basisL2Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(basisL2Loss, self).__init__()
        self.alpha = alpha # initilize alpha

    def forward(self, basis_model, alpha=None):

        # can input different alpha when calling the loss function, otherwise initilized value is used.
        if alpha is None:
            alpha = self.alpha

        loss_m = 0
        # loss_v = []

        for n, m in list(basis_model.named_modules()):
            if isinstance(m, basisConv2d):
                weights = m.basis_weight.view(m.basis_weight.shape[0], -1).t()
                # Normalize weights
                weights = F.normalize(weights, 2, 0)
                Q = weights.shape[1]

                if Q == 1:
                    continue

                dot_prod = torch.mm(weights.t(), weights).abs()

                dp_diag = (1 - dot_prod.diag() ** 2).sum()
                dp_else = (dot_prod.triu(1) ** 2).sum()

                curr_loss = (alpha * dp_diag / Q) + (2 * (1 - alpha)) / (Q * (Q - 1)) * dp_else
                loss_m += curr_loss
                # loss_v.append(curr_loss.item())

        # print(loss_v)
        # print(loss_c.item(), loss_m.item())
        return loss_m

class basisCombinationLoss(nn.Module):
    def __init__(self, w1, w2, skip_1by1, alpha=0.5, ce_reduction='mean'):
        super(basisCombinationLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss(reduction=ce_reduction)
        self.l1 = basisL1Loss(skip_1by1)
        self.l2 = basisL2Loss(alpha)
        self.w1 = w1
        self.w2 = w2

    def forward(self, basis_model, outputs, targets, w1=None, w2=None):

        if w1 is None:
            w1 = self.w1
        if w2 is None:
            w2 = self.w2

        loss_ce = self.ce(outputs, targets)

        if w1 > 0:
            loss_l1 = self.l1(basis_model)
        else:
            loss_l1 = torch.FloatTensor([0]).to(loss_ce.device)

        if w2 > 0:
            loss_l2 = self.l2(basis_model)
        else:
            loss_l2 = torch.FloatTensor([0]).to(loss_ce.device)

        # print(loss_l0)
        # print(loss_l2)

        loss = w1 * loss_l1 + \
               w2 * loss_l2 + \
               loss_ce

        return loss, loss_ce, loss_l1, loss_l2