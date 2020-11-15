import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ) 

    def forward(self, x):
        return self.double_conv(x)
    
    
class UNet_AE(nn.Module):

    def __init__(self, emb_size, layer_num, edge, activation):
        super().__init__()
        
        self.channels = [3, 64, 128, 256]
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduced_dim = edge // (2**layer_num)
        self.layer_num = layer_num
        
        self.encoder = nn.ModuleList([
            DoubleConv(self.channels[i], self.channels[i+1]) for i in range(layer_num)
        ])
        
        self.bottleneck_encoder = nn.Sequential(
            nn.Linear(self.channels[layer_num]*(self.reduced_dim**2), emb_size), 
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(emb_size, self.channels[layer_num]*(self.reduced_dim**2)), 
            nn.ReLU(inplace=True)
        )
               
        self.decoder = nn.ModuleList([
            DoubleConv(self.channels[i], self.channels[i-1]) for i in range(layer_num, 0, -1)
        ])
        
        
    def forward(self, x):
        
        for i, l in enumerate(self.encoder):
            x = F.relu(self.encoder[i](x))
            x = self.maxpool(x)
                        
        x = x.view(x.size()[0], -1)
        embeddings = self.bottleneck_encoder(x)
        x = self.bottleneck_decoder(embeddings)
        x = x.view(x.size()[0], self.channels[self.layer_num], self.reduced_dim, self.reduced_dim)
        
        for i, l in enumerate(self.decoder):
            x = self.upsample(x)
            x = self.decoder[i](x)
            
            if i != len(self.decoder)-1:
                x = F.relu(x)
            else:
                x = torch.sigmoid(x)
            
        return x, embeddings
    
    
class UNet_VAE(nn.Module):

    def __init__(self, emb_size, layer_num, edge, activation):
        super().__init__()
        
        self.channels = [3, 64, 128, 256]
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduced_dim = edge // (2**layer_num)
        self.layer_num = layer_num
        
        self.encoder = nn.ModuleList([
            DoubleConv(self.channels[i], self.channels[i+1]) for i in range(self.layer_num)
        ])
        
        self.mu = nn.Sequential(
            nn.Linear(self.channels[layer_num]*(self.reduced_dim**2), emb_size), 
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        
        self.var = nn.Sequential(
            nn.Linear(self.channels[layer_num]*(self.reduced_dim**2), emb_size), 
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(emb_size, self.channels[layer_num]*(self.reduced_dim**2)), 
            nn.ReLU()
        )
               
        self.decoder = nn.ModuleList([
            DoubleConv(self.channels[i], self.channels[i-1]) for i in range(self.layer_num, 0, -1)
        ])
        
        
    def forward(self, x):
        
        for i, l in enumerate(self.encoder):
            x = F.relu(self.encoder[i](x))
            x = self.maxpool(x)
                        
        x = x.view(x.size()[0], -1)
        
        z_mu = self.mu(x)
        z_var = self.var(x)
        
        std = z_var.mul(0.5).exp_()
        eps = Variable(torch.randn(std.size())).cuda() 
        x_sample = eps.mul(std).add_(z_mu)
        
        x = self.bottleneck_decoder(x_sample)
        x = x.view(-1, self.channels[self.layer_num], self.reduced_dim, self.reduced_dim)
                
        for i, l in enumerate(self.decoder):
            x = self.upsample(x)
            x = self.decoder[i](x)
            
            if i != len(self.decoder)-1:
                x = nn.ReLU()(x)
            else:
                x = torch.sigmoid(x)
            
        return x, z_mu, z_var