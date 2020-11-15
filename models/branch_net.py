import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision


class UpConv(nn.Module):

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