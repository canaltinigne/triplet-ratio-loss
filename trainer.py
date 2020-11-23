"""
Trainer Functions
@author: Can Altinigne

This script includes the train function, which is used in train.py 
script. This function is called in each epoch.

"""

import torch
from tqdm import tqdm
import torch.nn as nn


def train(model, trainLoader, optimizer, loss_fn, t_losses, m, ep, centers, c_classes, cfg, centerDistance=None, angularDistance=None):
    
    """
    Train function which does the forward and backpropagation
    in each epoch.

    Args:
        model: ResNet model to be trained.
        trainLoader: DataLoader object for the training set.
        optimizer: Optimizer object.
        loss_fn: Selected loss function.
        t_losses: Array to add training loss values continuously.
        m: Margin value.
        ep: Current epoch.
        centers: Current centers.
        c_classes: Dictionary to keep new embeddings as values and class labels as keys. 
                   This dictionary is used to take the average of new embeddings to find
                   new centers.
        cfg: Config dictionary.
        centerDistance: Lookup tables for Euclidean distance between centers.
        angularDistance: Lookup tables for Angular distance between centers.

    Returns:
        Average Training Loss for the current epoch.
        
    """
        
    model.train()
    
    with tqdm(total=len(trainLoader), dynamic_ncols=True) as progress:
        progress.set_description(f'Epoch {ep+1}')
        t_loss = 0

        for idx, (inputs, target) in enumerate(trainLoader):

            optimizer.zero_grad()
            out = model.forward(inputs.cuda())
            
            if cfg['use_norm'] == True:
                out = out / ((out**2).sum(dim=-1).sqrt().unsqueeze(-1) + 1e-7)
            else:
                if cfg['out_func'] == 'tanh':
                    out = nn.Tanh()(out)
                elif cfg['out_func'] == 'softmax':
                    out = nn.Softmax(dim=-1)(out)
                elif cfg['out_func'] == 'sigmoid':
                    out = nn.Sigmoid()(out)
                elif cfg['out_func'] == 'relu':
                    out = nn.ReLU()(out)
                    
            if cfg['originalTCL']:
                
                cfg['centerOptim'].zero_grad()
                
                if cfg['lossType'] == 19:
                    loss = loss_fn(out, target, m, centers, cfg['class_num'], c_classes, centerDistance, angularDistance)
                elif cfg['lossType'] == 17:
                    loss, _ = loss_fn(out, target)

            else:
                if cfg['lossType'] == 4:
                    loss, g1, g2 = loss_fn(out, target, m, centers, cfg['class_num'], c_classes, centerDistance, angularDistance)  
                else:
                    loss = loss_fn(out, target, m, centers, cfg['class_num'], c_classes, centerDistance, angularDistance)
                    
                    if cfg['reg'] > 0:
                        loss = loss + cfg['reg']*out.pow(2).mean(dim=0).sum() 
                        

            loss.backward()
            
            if cfg['lossType'] == 4:
                
                alpha = 0.5

                for k in g1:
                    delta = -alpha * (g1[k].sum(dim=0) / (g1[k].size()[0]+1))
                    centers[k] = centers[k].clone() + delta

                for k in g2:
                    delta = -alpha * -(g2[k].sum(dim=0) / (g2[k].size()[0]+1))
                    centers[k] = centers[k].clone() + delta

                for k in set(g1.keys()).union(set(g2.keys())):
                    centers[k] = centers[k].clone() / (centers[k].clone().pow(2).sum().sqrt()+1e-7)
            
            if cfg['originalTCL']:
                clip_gradient(cfg['centerOptim'], 0.01)
                cfg['centerOptim'].step()

            optimizer.step()

            t_loss += loss.item()

            avg_loss = t_loss / (idx + 1)
            progress.update(1)
            progress.set_postfix(loss=avg_loss)

    avg_t_loss = t_loss / len(trainLoader)
    t_losses.append(avg_t_loss)
    
    return avg_t_loss


def clip_gradient(optimizer, grad_clip):
    
    """
    This function clips the gradients for Original Triplet Center Loss

    Args:
        optimizer: Optimizer object.
        grad_clip: Clipping limits.
        
    Taken from:
    
        - https://github.com/xlliu7/Shrec2018_TripletCenterLoss.pytorch/blob/master/misc/utils.py
  
    """
        
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)


def createCenters(model, trainLoader, numClass, dim):
    
    """
    This function creates the centroids by doing forward propagation
    with the initial network before training phase. This function is
    redundant as we create our centroids with one-hot encoded vectors
    and Random Gaussians.

    Args:
        model: Model object.
        trainLoader: DataLoader object for the training set. 
        numClass: Number of classes.
        dim: Embedding dimension.
        
    """
        
    model.eval()
    centers = torch.rand(numClass,dim).cuda()
    c_points = dict()
    
    with torch.no_grad():

        for idx, (inputs, target) in enumerate(trainLoader):
            output = model.forward(inputs.cuda())
            
            for i, out in enumerate(output):
                classId = target[i].item()
                diffClassInd = [x for x in range(centers.size()[0]) if x != classId]

                if c_points is not None:
                    if classId not in c_points:
                        c_points[classId] = out.unsqueeze(0).data
                    else:
                        c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
            
    for i in c_points:
        centers[i] = c_points[i].mean(dim=0)
            
    return centers