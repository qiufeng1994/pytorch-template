import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target.float())*10

def softmax_ce(output, target):
    return F.cross_entropy(output, target.long())

def sigmoid_ce(output, target):
    return F.binary_cross_entropy(output, target)

def softmax_cross_entropy_with_logits(output, target):
    loss = torch.sum(- target * F.log_softmax(output, -1), -1)
    mean_loss = loss.mean()
    return mean_loss

def bce(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
#
# def bce(output, target):
#     return torch.nn.BCEWithLogitsLoss(output, target)