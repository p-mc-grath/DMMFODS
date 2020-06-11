# Copied from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# Full credit to Allen Qin
# paper: https://arxiv.org/abs/1708.02002

import torch
import torch.nn as nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    '''
    Pixel-wise loss, down-weighing easy negative samples, e.g. for high background-foreground imbalance
    Works with binary as well as probabilistic input
    '''
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        '''
        Arguments:
            alpha: Hyperparam; refer to paper
            gamma: Hyperparam; refer to paper
            logits: boolean: True -> expecting binary input else -> values between [0,1]
            reduce: boolean: function same as reduction in torch.nn.BCEWithLogitsLoss | refer to torch.nn.functional.binary_cross_entropy_with_logits
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        '''
        Arguments:
            inputs: refer to torch.nn.functional.binary_cross_entropy_with_logits
            target: refer to torch.nn.functional.binary_cross_entropy_with_logits
        return:
            loss
        '''
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss