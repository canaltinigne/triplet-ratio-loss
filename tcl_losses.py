"""
Original Triplet Center Loss with trainable centers.
@author: Can Altinigne

This script includes the original Triplet Center Loss.

TCL Paper:
    He et al., 2018
    - https://openaccess.thecvf.com/content_cvpr_2018/papers/He_Triplet-Center_Loss_for_CVPR_2018_paper.pdf

This script is taken from:

    - https://github.com/eriche2016/cvpr_2018_TCL.pytorch/blob/master/custom_losses.py

"""

import torch 
import torch.nn as nn 
import torch.nn.parallel
import torch.nn.functional as F 
from torch.autograd import Variable 
from torch.nn import Parameter 
import numpy as np 

# for all classes  
class TripletCenter40LossAllClass(nn.Module):
    def __init__(self, margin, class_num, dimension):
        super(TripletCenter40LossAllClass, self).__init__()
        self.margin = margin
        self.ranking_loss_center = nn.MarginRankingLoss(margin=self.margin)
        self.centers = nn.Parameter(torch.randn(class_num, dimension)) # for modelent40

    def forward(self, inputs, targets):
        n = inputs.size(0)
        m = self.centers.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(m, n).t()
        dist.addmm_(1, -2, inputs, self.centers.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # for each anchor, find the hardest positive and negative
        mask = torch.zeros(dist.size()).type(torch.cuda.BoolTensor)

        for i in range(n):
            mask[i][targets[i].item()] = 1

        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # hardest positive center
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # hardest negative center

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss_center(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        # normalize data by batch size
        return loss, prec
