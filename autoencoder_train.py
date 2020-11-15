import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
import unsup_trainer as trainer
from models import *
import argparse
from datetime import datetime
from unsup_helpers import *
import os
from cub2011 import Cub2011
import PIL
from autoencoders import UNet_AE, UNet_VAE


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Deep Metric Learning')
    parser.add_argument('-e', '--epoch', type=int, required=True, help='Epoch')
    parser.add_argument('-lr', '--rate', type=float, required=True, help='Learning Rate')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch Size')
    parser.add_argument('-d', '--dim', type=int, default=1024, help='Embedding Dimension')
    parser.add_argument('-data', '--dataset', type=str, required=True, help='Choose dataset')
    parser.add_argument('-m', '--model', type=str, required=True, help='Choose model')
    parser.add_argument('-out', '--output', type=str, required=True, help='Choose embedding activation function')
    
    args = parser.parse_args()
    
    assert args.dataset in ['dogs', 'cifar10', 'cifar100', 'cub', 'cars', 'imagenet'], "Dataset not found" 
    assert args.model in ['ae', 'vae'], "Model not found"
    assert args.output in ['sigmoid', 'tanh'], "Activation not found"
    
    # -------------- DATASET PREPARATION --------------
        
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=torchvision.transforms.ToTensor())
        
    elif args.dataset == 'cub':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])

        trainset = Cub2011(root='./data', train=True, transform=transform)
        trainset.class_to_idx = range(200)
    
    elif args.dataset == 'cars':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/stanford-cars/car_data/car_data/train/', transform=transform)
        
    elif args.dataset == 'dogs':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/stanford-dogs/train/', transform=transform)
        
    elif args.dataset == 'imagenet':
        
        trainset = torchvision.datasets.ImageFolder(root='data/tiny-imagenet-200/train/',
                                                    transform=torchvision.transforms.ToTensor())
        
    # -------------- INITIALIZATIONS ----------------
    
    
    CLASS_NUM = len(trainset.class_to_idx)
    DIM = args.dim
    EPOCH = args.epoch
    LR = args.rate
    BATCH_SIZE = args.batch
    LEVEL = 2 if args.dataset == 'cifar10' else 3
    
    EDGE = {
        'cifar10': 32,
        'cub': 224,
        'cars': 224,
        'dogs': 224,
        'imagenet': 64
    }
           
    if args.model == 'ae':
        model = UNet_AE(DIM, LEVEL, EDGE[args.dataset], args.output).cuda()
    elif args.model == 'vae':
        model = UNet_VAE(DIM, LEVEL, EDGE[args.dataset], args.output).cuda()
        
    SAVE_DIR = datetime.now().strftime("%m%d%Y_%H%M%S") + '_AUTOENCODER_model_' + args.dataset + '_' + args.model + '_' + args.output + '_ep{}_lr{}_b{}_d{}/'.format(EPOCH, LR, BATCH_SIZE, DIM)
    
    os.makedirs('models/' + SAVE_DIR)
     
    lowest_error = float('inf')
    best_ep = -1
    
    t_losses = []
    m_values = []
    START_EPOCH = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ------------------ TRAINING ------------------
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
    for epoch in range(EPOCH):
        
        model.train()
    
        with tqdm(total=len(trainLoader), dynamic_ncols=True) as progress:
            progress.set_description(f'Epoch {epoch+1}')
            t_loss = 0

            for idx, (inputs, target) in enumerate(trainLoader):

                optimizer.zero_grad()
                inp = inputs.cuda()
                
                if args.model == 'ae':
                    out, _ = model.forward(inp)
                    loss = F.mse_loss(out, inp)
                elif args.model == 'vae':
                    out, z_mu, z_var = model.forward(inp)
                    
                    # reconstruction loss
                    recon_loss = F.binary_cross_entropy(out, inp, reduction='sum')

                    # kl divergence loss
                    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        
                    loss = recon_loss + kl_loss

                loss.backward()

                optimizer.step()

                t_loss += loss.item()

                avg_loss = t_loss / (idx + 1)
                progress.update(1)
                progress.set_postfix(loss=avg_loss)

        avg_t_loss = t_loss / len(trainLoader)
        t_losses.append(avg_t_loss)
    
        print('Epoch: {} | Loss: {:.4f}'.format(epoch+1, avg_t_loss))
                
        if avg_t_loss <= lowest_error:
            lowest_error = avg_t_loss
            
            state = {
                'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': t_losses,
                'm_values': m_values,
                'dim': DIM,
                'lr': LR,
                'batch_size': BATCH_SIZE
            }

            if os.path.exists('models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(best_ep)):
                os.remove('models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(best_ep))

            torch.save(state, 'models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(epoch+1))
            
            best_ep = epoch+1
    
    state = {
        'epoch': EPOCH, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': t_losses,
        'm_values': m_values,
        'dim': DIM,
        'lr': LR,
        'batch_size': BATCH_SIZE
    }

    torch.save(state, 'models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(EPOCH))

