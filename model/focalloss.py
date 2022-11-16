

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2,alpha=0.25,weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = torch.zeros(2)
        self.alpha[0] += alpha
        self.alpha[1:] += (1-alpha)

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        self.alpha=self.alpha.to(target.device)
        self.weight=self.weight.to(target.device)

        alpha = self.alpha.gather(0,target.view(-1))
        weight= self.weight.gather(0,target.view(-1))
        
        logpt = torch.log(F.sigmoid(input))
        
        pt = torch.exp(logpt)
        
        
        loss = -torch.mul(torch.pow((1-pt), self.gamma), logpt)  
        loss = torch.mul(alpha*weight, loss.t())
        loss = loss.sum()
        
        return loss