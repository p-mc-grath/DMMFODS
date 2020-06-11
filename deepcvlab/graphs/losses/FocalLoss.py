# Copied from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# Full credit to Allen Qin for the basic implementation of the original Focal Loss
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
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)                                                                       # -BCE_loss = log(pt)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class ClassWiseFocalLoss(FocalLoss):
    '''
    See FocalLoss
    + Allows to apply different alpha and gamma for each class
        alpha: class-class imbalance
        gamma: class-background imbalance
    '''
    def __init__(self, alpha=[1], gamma=[8, 13, 16], logits=True, reduce=False):
        '''
        See FocalLoss
        Expects gamma and alpha to be of same length
        '''
        super().__init__(alpha, gamma, logits, reduce)

    def forward(self, inputs, targets): 
        '''
        Expects targets and inputs to be structured like: batches x classes x X x Y
        
        Arguments:
            see FocalLoss
        return:
            loss
        '''
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)                                                                       # -BCE_loss = log(pt)
        
        F_loss = torch.zeros_like(pt)
        for i, (alpha, gamma) in enumerate(zip(self.alpha, self.gamma)):
            F_loss[:, i, :, :] = alpha * (1-pt[:, i, :, :])**gamma * BCE_loss[:, i, :, :]

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

