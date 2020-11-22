"""
Different Loss Functions and Helper Functions
@author: Can Altinigne

This script includes the loss functions that have been used 
in the experiments. Also, there are several helper functions to 
find and initialize centers.

All loss functions get the same parameters, so that I added
parameter definitions to the first loss function only.

"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import time


def centerInit(numClass, dim):
    
    """
    Center Initialization Function.
    
    Initialize the embeddings between [-1, 1]

    Args:
        numClass: Number of classes in the dataset.
        dim: Embedding dimension.

    Returns:
        Initial center embeddings that have values between 
        [-1,1]
        
        centerInit(10,128) -> torch.Tensor(10,128)
        
    """
    
    return (torch.rand(numClass,dim)-0.5)*2


def distance(m1, m2):
    
    """
    Squared Euclidean Distance Calculation Function.

    Args:
        m1: The first tensor.
        m2: The second tensor.

    Returns:
        Euclidean Distance between two tensors.
        
        m1 = torch.from_numpy([2,3])
        m2 = torch.from_numpy([5,7])
        
        distance(m1,m2) -> 25.0
        
    """
        
    return (m1-m2)**2


def angularDistance(m1, m2):
    
    """
    Cosine Distance Calculation Function.
    
    This function is wrongly named. For real angular distance 
    calculation please check 'originalAngularTripletLossUpdated' 
    function below.

    Args:
        m1: The first tensor.
        m2: The second tensor.

    Returns:
        Cosine Distance between two tensors.
        
        m1 = torch.from_numpy([3,4])
        m2 = torch.from_numpy([6,8])
        
        distance(m1,m2) -> 0.0
        
    """
        
    return 1 - (m1*m2).sum(dim=-1)/((m1**2).sum().sqrt() * (m2**2).sum(dim=-1).sqrt() + 1e-32)


def centerUpdate(centers, c_points, numClass, adaptive=False, epoch=None, k=None, norm=False):
    
    """
    Center Update Function.
    
    Updates the centers with KMeans-like schema. It 
    also supports adaptive updating mechanism.

    Args:
        centers: Centroids from the previous epoch torch.Tensor(CLASS NUMBER, EMBEDDING SIZE)
        c_points: 
        adaptive: If adaptive updating mechanism is set to True.
        epoch: Implicit epoch parameter set in train function in trainer.py.
        k: Implicit k parameter in adaptive updating mechanism set in train.py.
        norm: Implicit normalization parameter set in train.py.
        
    """
        
    for i in range(numClass):
        if adaptive:
            rate = 0.9*np.exp(-k*epoch)
            centers[i] = rate*centers[i] + (1-rate)*c_points[i].mean(dim=0)
            
        else:
            centers[i] = c_points[i].mean(dim=0)
            
        if norm:
            centers[i] = centers[i] / (centers[i].pow(2).sum().sqrt()+1e-7)


def tripletCenterLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Original Triplet Center Loss Function.

    Args:
        output: Output embeddings: torch.Tensor(BATCH SIZE, EMBEDDING SIZE).
        target: Class label of the embeddings: torch.Tensor(BATCH SIZE, 1).
        m: Margin value in triplet loss function.
        centers: Centroids: torch.Tensor(NUMBER OF CLASSES, EMBEDDING SIZE).
        class_num: Number of classes in the dataset.
        c_points: Helper dictionary to add new embeddings to the class they belong to.
        center_distance: Lookup table for center distances: torch.Tensor(NUMBER OF CLASSES, NUMBER OF CLASSES)
        ang_distance: Same as center_distance but it has angular distance instead.

    Returns:
        Triplet Center Loss in a batch.
        
    """
        
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
                
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, centers[diffClassInd]).sum(dim=-1).min()
       
        loss = loss + F.relu(D_xc - min_D_xj + m)
        
    return loss/target.size()[0]


def tripletCenterLossVectorized(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Well vectorized Original Triplet Center Loss Function.
    
    This function is the well vectorized version of 'tripletCenterLoss' 
    function. This version is used for the time measurement experiments.
        
    """
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    n = output.size(0)
    m_ = centers.size(0)
    inputs = output

    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m_) + \
        torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(m_, n).t()
    dist.addmm_(1, -2, inputs, centers.t())

    mask = torch.zeros(dist.size()).type(torch.cuda.BoolTensor)
    #mask = torch.zeros(dist.size()).type(torch.BoolTensor)

    for i in range(n):
        mask[i][target[i].item()] = 1

    # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []

    dist_ap = torch.cat([dist[i][mask[i]].max().unsqueeze(0) for i in range(n)])
    dist_an = torch.cat([dist[i][mask[i] == 0].min().unsqueeze(0) for i in range(n)])
    
    torch.cuda.synchronize()
    ang_distance['loss'].append(time.time() - start_time)
        
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
               
    
    return F.relu(dist_ap - dist_an + m).mean()


def tripletCenterLossV2(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Our version of Original Triplet Center Loss Function
    
    This function uses look-up tables to improve the performance
    compared to the original Triplet Center Loss function.
        
    """
        
    loss = 0.

    for i, out in enumerate(output):
        classId = target[i].item()

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum().sqrt()
        
        # 2 - Random Triplet Center Loss
        #min_D_xj = distance(out, centers[diffClassInd][np.random.randint(len(diffClassInd))]).sum().sqrt()
        
        # 3 - Distance between sample and a center (the closest one to sample's center)
        min_D_xj = distance(out, center_distance[classId]).sum().sqrt()
                
        loss = loss + F.relu(D_xc - min_D_xj + m)
    
    return loss/target.size()[0]
    

def tripletCenterLossV2Vectorized(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Well vectorized version of Our Triplet Center Loss Function.
    
    This function is the well vectorized version of 'tripletCenterLossV2' 
    function. This version is used for the time measurement experiments.
        
    """
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Well Vectorized
    D_xc = distance(output, centers[target]).sum(dim=-1).sqrt()
    min_D_xj = distance(output, torch.cat([centers[center_distance[x.item()]].unsqueeze(0) for x in target])).sum(dim=-1).sqrt()
    
    torch.cuda.synchronize()
    ang_distance['loss'].append(time.time() - start_time)
    
    for i, out in enumerate(output):
        classId = target[i].item()

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
    
    return F.relu(D_xc - min_D_xj + m).mean()
    
    
def angularTripletLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Our Version of Angular Triplet Center Loss Function with Cosine Distance
    as distance metric.
    
    This loss function assumes our approach of using look-up tables.
        
    """
        
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item()

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = angularDistance(out, centers[classId])
        min_D_xj = angularDistance(out, center_distance[classId])
                
        loss = loss + F.relu(D_xc - min_D_xj + m)
    
    return loss/target.size()[0]


def angularTripletLossUpdated(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Our Version of Angular Triplet Center Loss Function with Angular Distance
    as distance metric.
    
    This loss function assumes our approach of using look-up tables.
        
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item()
        D_xc = torch.acos((out * centers[classId]).sum())
        
        hard_i = center_distance[classId]
        
        min_D_xj = torch.acos((out * centers[hard_i]).sum())      
        loss = loss + F.relu(D_xc - min_D_xj + m)
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
    return loss/target.size()[0]


def originalAngularTripletLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Original Angular Triplet Center Loss Function with Cosine Distance
    as distance metric.
    
    This loss function assumes our approach of using look-up tables.
        
    """
    
    loss = 0.
        
    for i, out in enumerate(output):
        classId = target[i].item() 
        diffClassInd = [x for x in range(class_num) if x != classId]

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = angularDistance(out, centers[classId])
        min_D_xj = angularDistance(out, centers[diffClassInd]).min()
                
        loss = loss + F.relu(D_xc - min_D_xj + m)
    
    return loss/target.size()[0]


def originalAngularTripletLossUpdated(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Original Angular Triplet Center Loss Function with Angular Distance
    as distance metric.
    
    Detailed description on how to calculate ATCL can be found in the 
    paper below.
    
    Li et al., 2019
    https://arxiv.org/pdf/1811.08622.pdf
        
    """
    
    loss = 0.
    
    g1 = dict()
    g2 = dict()
        
    for i, out in enumerate(output):
        classId = target[i].item() 
        
        D_xc = torch.acos((out * centers[classId]).sum())
        
        distances = torch.acos((out * centers).sum(dim=-1))
        distances[classId] = 1e32
        hard_i = distances.argmin()
        
        min_D_xj = torch.acos((out * centers[hard_i]).sum())
        
        l = F.relu(D_xc - min_D_xj + m)
                
        loss = loss + l
        
        a_i = D_xc
        b_i = min_D_xj
        
        if l > 0:
            if hard_i not in g1:
                g1[hard_i] = (out / (torch.sin(b_i)+1e-7)).unsqueeze(0).data
            else:
                g1[hard_i] = torch.cat((g1[hard_i], (out / (torch.sin(b_i)+1e-7)).unsqueeze(0).data), dim=0)
            
        if l > 0:
            if classId not in g2:
                g2[classId] = (out / (torch.sin(a_i)+1e-7)).unsqueeze(0).data
            else:
                g2[classId] = torch.cat((g2[classId], (out / (torch.sin(a_i)+1e-7)).unsqueeze(0).data), dim=0)
    
    return loss/target.size()[0], g1, g2


def angularOursCombined(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Loss Function which combines our implementation of 
    angular and triplet center loss functions.
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item()

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = nn.Tanh()(out).unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], nn.Tanh()(out).unsqueeze(0).data), dim=0)
        
        D_xc_ours = distance(nn.Tanh()(out), centers[classId]).sum().sqrt()
        min_D_xj_ours = distance(nn.Tanh()(out), center_distance[classId]).sum().sqrt()
        ours = F.relu(D_xc_ours - min_D_xj_ours + m) 
        
        D_xc_angular = angularDistance(out, centers[classId])
        min_D_xj_angular = angularDistance(out, ang_distance[classId])
        angular = F.relu(D_xc_angular - min_D_xj_angular + 0.7)
                
        loss = loss + ours + angular
    
    return loss/target.size()[0]


def sigmoidLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Experimental Loss Function 
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
                
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, center_distance[classId]).sum()
       
        loss = loss + (nn.Sigmoid()(D_xc) - nn.Sigmoid()(min_D_xj) + 1.)
        
    return loss/target.size()[0]


def logLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Experimental Loss Function 
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        first = -torch.log(nn.Sigmoid()((out*centers[classId]).sum()) + 1e-8) 
        second = -torch.log(1. - nn.Sigmoid()((out*center_distance[classId]).sum()) + 1e-8)
        
        loss = loss + first + second
        
    return loss/target.size()[0]


def sigmoidLossV2(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Experimental Loss Function 
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
                
        pos = (out*centers[classId]).sum()
        neg = (out*center_distance[classId]).sum()
       
        loss = loss + (-nn.Sigmoid()(pos) + nn.Sigmoid()(neg) + 1.)
        
    return loss/target.size()[0]


def logLossV2(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Experimental Loss Function 
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
                
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, center_distance[classId]).sum()
        
        first = -torch.log(nn.Sigmoid()(D_xc) + 1e-8) 
        second = -torch.log(1 - nn.Sigmoid()(min_D_xj) + 1e-8)
        
        loss = loss + first + second
        
    return loss/target.size()[0]


def logLossV3(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Redundant Experimental Loss Function 
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        pos = (out*centers[classId]).sum() / ((out**2).sum().sqrt()*(centers[classId]**2).sum().sqrt() + 1e-8)
        neg = (out*center_distance[classId]).sum() / ((out**2).sum().sqrt()*(center_distance[classId]**2).sum().sqrt() + 1e-8)
        
        first = -torch.log(torch.sigmoid(pos) + 1e-8) 
        second = -torch.log(torch.sigmoid(1 - neg) + 1e-8)
        
        loss = loss + first + second
        
    return loss/target.size()[0]


def expLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Our Version of Centroid-based Triplet Ratio Loss Function
    with Squared Euclidean distance as distance metric.
       
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, center_distance[classId]).sum()
        
        loss = loss + torch.exp(-min_D_xj / (D_xc+1e-8))
        
    return loss/target.size()[0]


class expLossLearnable(nn.Module):
    
    """
    Our Version of Centroid-based Triplet Ratio Loss Function
    with Squared Euclidean distance as distance metric.
    
    This version has trainable centers as another hyperparameter.
    Trainable centers are defined in __init__ function and named
    as self.centers. The centers are initialized as Random Gausssians.
            
    """
    
    def __init__(self, class_num, dimension):
        super(expLossLearnable, self).__init__()
        self.centers = nn.Parameter(torch.randn(class_num, dimension))

    def forward(self, output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
        
        loss = 0.
    
        for i, out in enumerate(output):
            classId = target[i].item() #0 if target[i].item() == 3 else 1
            diffClassInd = [x for x in range(class_num) if x != classId]

            D_xc = distance(out, self.centers[classId]).sum()
            min_D_xj = distance(out, self.centers[center_distance[classId]]).sum()

            loss = loss + torch.exp(-min_D_xj / (D_xc+1e-8))

        return loss/target.size()[0]


def originalTRL(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Original Centroid-based Triplet Ratio Loss Function
    with Squared Euclidean distance as distance metric.
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum().sqrt()
        min_D_xj = distance(out, center_distance[classId]).sum().sqrt()
        
        denom = torch.exp(D_xc) + torch.exp(min_D_xj) + 1e-8
        
        l = (torch.exp(D_xc)/denom).pow(2) + (1 - torch.exp(min_D_xj)/denom).pow(2)
        
        loss = loss + l
        
    return loss/target.size()[0]


def ratioLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Shallow Centroid-based Ratio Loss Function
    with Squared Euclidean distance as distance metric.
            
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, center_distance[classId]).sum()
        
        loss = loss + (D_xc / (min_D_xj + 1e-8))
        
    return loss/target.size()[0]


def variantExpLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Different Version of Our Version of Centroid-based 
    Triplet Ratio Loss Function with Squared Euclidean 
    distance as distance metric.
       
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        nom = (out**2).sum().sqrt()*(center_distance[classId]**2).sum().sqrt() - (out*center_distance[classId]).sum()
        denom = (out**2).sum().sqrt()*(centers[classId]**2).sum().sqrt() - (out*centers[classId]).sum()
        
        loss = loss + torch.exp(-nom / (denom+1e-8))
        
    return loss/target.size()[0]


def taylorLoss(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Taylor Series Approximation of Our Version of Centroid-based 
    Triplet Ratio Loss Function with Euclidean 
    distance as distance metric.
       
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        pos = distance(out, centers[classId]).sum().sqrt()
        neg = distance(out, center_distance[classId]).sum().sqrt()
        
        x = neg/(pos + 1e-8)
        
        loss = loss + (1 - x + x.pow(2)/2 - x.pow(3)/6 + x.pow(4)/24)
        
    return loss/target.size()[0]


def expLossEuclidean(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Our Version of Centroid-based Triplet Ratio Loss Function
    with Euclidean distance as distance metric.
       
    """
        
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum().sqrt()
        min_D_xj = distance(out, center_distance[classId]).sum().sqrt()
        
        loss = loss + torch.exp(-min_D_xj / (D_xc+1e-8))
        
    return loss/target.size()[0]


def taylorLossSquared(output, target, m, centers, class_num, c_points=None, center_distance=None, ang_distance=None):
    
    """
    Taylor Series Approximation of Our Version of Centroid-based 
    Triplet Ratio Loss Function with Squared Euclidean 
    distance as distance metric.
       
    """
    
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() #0 if target[i].item() == 3 else 1
        diffClassInd = [x for x in range(class_num) if x != classId]
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        pos = distance(out, centers[classId]).sum()
        neg = distance(out, center_distance[classId]).sum()
        
        x = neg/(pos + 1e-8)
        
        loss = loss + (1 - x + x.pow(2)/2 - x.pow(3)/6 + x.pow(4)/24)
        
    return loss/target.size()[0] 