"""
Unsupervised Training Helper Functions
@author: Can Altinigne

This script has the helper functions and dataset classes
for unsupervised training.

"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision


def distance(m1, m2):
    
    """
    Squared Euclidean Distance Calculation Function. Loss
    functions take the sum of the output to calculate the 
    distance.

    Args:
        m1: The first tensor.
        m2: The second tensor.

    Returns:
        Euclidean Distance between two tensors as a vector.
        
        m1 = torch.from_numpy([2,3])
        m2 = torch.from_numpy([5,7])
        
        distance(m1,m2) -> torch.Tensor([9, 16])
        
    """
        
    return (m1-m2).pow(2)


class RotateNet(nn.Module):
    
    """
    Auxiliary rotation loss function to predict the angle of 
    rotated image. This loss function is implemented using 
    the following paper.
    
    Cao et al., 2019
    https://arxiv.org/abs/1911.07072

    Args:
        model: Model object.
        embDim: Embedding dimension.

    Returns:
        embedding: Output embedding.
        y_pred: Predicted rotation angle (can be 0, 90, 180 or 270; Cross Entropy loss is used).
        
    """
        
    def __init__(self, model, embDim):
        super(RotateNet, self).__init__()
        self.embeddingNet = model  
        self.linear = torch.nn.Linear(embDim, 4)

    def forward(self, x):
        embedding = self.embeddingNet(x)
        y_pred = self.linear(F.relu(embedding))
        return embedding, y_pred
    

class UnsupervisedDataset(Dataset):
    
    """
    Unsupervised Dataset Preparer

    Args:
        ds: Pandas table for the dataset, which is set implicitly in train.py.
        size: Image size of the dataset, which is set implicitly in train.py.

    Returns:
        Training dataset where there are image, positive embedding, class label
        and rotation angle for each sample in the dataset.
        
    """

    def __init__(self, ds, size):
        self.dataset = ds
        
        self.TP = torchvision.transforms.ToPILImage()
        self.rotate = torchvision.transforms.functional.rotate
        
        self.positive = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            #torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        x, y = self.dataset.__getitem__(idx)
        angle = np.random.choice([0,1,2])
        
        return {
            "query": x,
            "positive": self.positive(self.rotate(self.TP(x), (angle-1)*30)),
            "target": y,
            "rotation": angle
        }


def tripletLoss(q_out, target, centers, kmeans, center_distances, m, class_ids):
    
    """
    Original Triplet Center Loss Function in Unsupervised Settings.

    Args:
        q_out: Output embeddings: torch.Tensor(BATCH SIZE, EMBEDDING SIZE).
        target: Class label of the embeddings: torch.Tensor(BATCH SIZE, 1).
        centers: Margin value in triplet loss function.
        kmeans: Redundant parameter.
        center_distances: Look-up table that keeps the distance between centroids.
        m: Margin value.
        class_ids: Class labels that the embeddings are assigned to

    Returns:
        Triplet Center Loss in a batch.
        
    """
                   
    #pos_d = distance(query, pos).sum(dim=-1).sqrt()
    #neg_d = distance(query, neg).sum(dim=-1).sqrt()
    
    loss = 0.
    
    for i, out in enumerate(q_out):
        
        classId = class_ids[i]
        
        D_xc = (distance(out, centers[classId]).sum()+1e-8).sqrt()
        min_D_xj = (distance(out, centers[center_distances[classId]]).sum()+1e-8).sqrt()
        
        loss = loss + F.relu(D_xc - min_D_xj + m)
        
    return loss/q_out.size()[0]
        

def expTripletLoss(q_out, target, centers, kmeans, center_distances, m, class_ids):
    
    """
    Our Version of Centroid-based Triplet Ratio Loss Function in 
    Unsupervised Settings.

    Args:
        q_out: Output embeddings: torch.Tensor(BATCH SIZE, EMBEDDING SIZE).
        target: Class label of the embeddings: torch.Tensor(BATCH SIZE, 1).
        centers: Margin value in triplet loss function.
        kmeans: Redundant parameter.
        center_distances: Look-up table that keeps the distance between centroids.
        m: Margin value.
        class_ids: Class labels that the embeddings are assigned to

    Returns:
        Triplet Ratio Loss in a batch.
        
    """
    
    loss = 0.
    
    for i, out in enumerate(q_out):
        classId = class_ids[i]
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, centers[center_distances[classId]]).sum()
    
        loss = loss + torch.exp(-min_D_xj / (D_xc+1e-8))
        
    return loss/q_out.size()[0]
    
    #for query in q_out:
    
    #pos_d = distance(query, pos).sum(dim=-1)
    #neg_d = distance(query, neg).sum(dim=-1)

    #return torch.exp(-neg_d / (pos_d + 1e-8)).mean()


def ratioTripletLoss(query, pos, neg, m):
    
    """
    Shallow Centroid-based Ratio Loss Function with 
    Squared Euclidean distance as distance metric in
    Unsupervised Settings.

    Args:
        q_out: Output embeddings: torch.Tensor(BATCH SIZE, EMBEDDING SIZE).
        target: Class label of the embeddings: torch.Tensor(BATCH SIZE, 1).
        centers: Margin value in triplet loss function.
        kmeans: Redundant parameter.
        center_distances: Look-up table that keeps the distance between centroids.
        m: Margin value.
        class_ids: Class labels that the embeddings are assigned to

    Returns:
        Shallow Triplet Ratio Loss in a batch.
        
    """
    
    pos_d = distance(query, pos).sum(dim=-1)
    neg_d = distance(query, neg).sum(dim=-1)

    return (pos_d / (neg_d + 1e-8)).mean()
