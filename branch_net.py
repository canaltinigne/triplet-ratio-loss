"""
Output Layer with Multiple Branches cascaded onto ResNet structure.
(excluding the last layer of ResNet)
@author: Can Altinigne

This script includes the experimental multi-branch output layer
which includes auxilary loss functions such as rotation loss
and reconstruction loss for the unsupervised deep metric learning
applications. I mainly consider the following paper while 
implementing this script.

Cao et al., 2019
https://arxiv.org/abs/1911.07072

"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision


class UpConv(nn.Module):
    
    """
    Two Convolution layers with an activation function
    between them.

    2D Convolution -> Activation -> 2D Convolution

    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of channels in the output tensor.

    Returns:
        An output tensor after applying two 2D convolutions and ReLU 
        activation between those convolutions.
        
        DoubleConv(32,64)(torch.Tensor(1,32,48,48)) returns 
        torch.Tensor(1,64,48,48)
        
    """

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ) 

    def forward(self, x):
        return self.double_conv(x)
    
    
class Decoder(nn.Module):
    
    """
    Decoder network in order to reconstruct the input image. This
    decoder is used for auxiliary image reconstruction loss function.

    4 Upsampling Layers with 2D Convolutions

    Args:
        last_dim: Dimension of the output of ResNet network (embedding size).
        edge: Edge size of an image which is set implicitly.

    Returns:
        Reconstructed input image.
        
    """

    def __init__(self, last_dim, edge):
        super(Decoder, self).__init__()
        
        self.edge = edge//8
        self.init_net = nn.Linear(last_dim, self.edge**2)

        self.decoder = nn.ModuleList([
            UpConv(1, 256),
            UpConv(256, 128),
            UpConv(128, 64),
            UpConv(64, 3),
        ])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        
        x = F.relu(self.init_net(x)).view(-1, 1, self.edge, self.edge) 
        
        for i, l in enumerate(self.decoder):
            
            if i == 0:
                x = F.relu(self.decoder[i](x))
            
            else:
                x = self.upsample(x)
                x = self.decoder[i](x)
            
                if i != len(self.decoder)-1:
                    x = F.relu(x)
                else:
                    x = torch.sigmoid(x)
                    
        return x
    
    
class BranchNet(torch.nn.Module):
    
    """
    Multi-output module cascaded onto the output of ResNet 
    (excluding the last layer of ResNet)

        Branches:
            - Embedding 
            - Rotation Loss
            - Reconstruction Loss
            
    The image is rotated in different angles 90, 180, 270 and the 
    rotation branch predicts the rotation. Currently, this branch 
    is inactive, you need to change train.py file by adding random
    rotation in torchvision.transforms.Compose function.
    
    Check torch.rot90 function
    
        Example:

            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torch.rot90(x, 1)
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    Args:
        last_dim: Number of channels in the input tensor.
        emb_dim: Number of channels in the output tensor.
        emb_activation: Activation function applied to the embeddings.
        edge: Edge size of the images in a dataset. This paramater is set
        implicitly in autoencoder_train.py script considering the dataset 
        used.

    Returns:
        - Embeddings: torch.Tensor([BATCH SIZE, EMBEDDING SIZE])
        - Reconstructed Image: torch.Tensor([BATCH SIZE, 3, EDGE, EDGE])
        - Rotation Output: torch.Tensor([BATCH SIZE, 3])
        
    """
    
    def __init__(self, last_dim, emb_dim, emb_activation, edge):
        super(BranchNet, self).__init__()
        
        self.embedding_br = nn.Linear(last_dim, emb_dim)
        #self.rotate_br = nn.Linear(last_dim, 3)
        self.recon_br = Decoder(last_dim, edge)
        self.activation = emb_activation
        
        assert emb_activation in ['linear', 'tanh', 'sigmoid', 'norm'], "Error: Activation not found."
        

    def forward(self, x):
        
        emb_out = self.embedding_br(x)
        #rotation_out = self.rotate_br(x)
        recon_out = self.recon_br(x)
        
        if self.activation == "norm":
            emb_out = emb_out / (emb_out.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) + 1e-7)
        
        elif self.activation == 'tanh':
            emb_out = nn.Tanh()(emb_out)
        
        elif self.activation == 'sigmoid':
            emb_out = torch.sigmoid(emb_out)

        return emb_out, recon_out # rotation_out, 