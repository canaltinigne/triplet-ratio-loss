"""
Unsupervised Training Script for Deep Metric Learning
@author: Can Altinigne

This script is the main program that trains deep metric learning
models in unsupervised settings. Currently, K-means clustering 
algorithm is run for each epoch to assign class labels as in the 
previous unsupervised deep metric learning papers.

    - Parameters:
        - e: Number of epochs
        - lr: Learning rate
        - b: Batch size
        - d: Embedding dimension
        - m: Redundant Parameter
        - l: Loss Function, Please check if section below to see which loss function corresponds
             to which number.
        - n: Network Model, ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        - max: Redundant parameter
        - min: Redundant parameter
        - norm: In order to use L2 normalization on embeddings (for original TCL)
        - data: Dataset name, ['dogs', 'cifar10', 'cifar100', 'cub', 'cars', 'imagenet', 'flowers', 'aircraft']
        - out: Output activation on embeddings, ['linear', 'tanh', 'softmax', 'sigmoid', 'twohead', 'relu']
        - r: Give the .pth file directory to resume training from a saved model.

"""

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
from branch_net import BranchNet
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Deep Metric Learning')
    parser.add_argument('-e', '--epoch', type=int, required=True, help='Epoch')
    parser.add_argument('-lr', '--rate', type=float, required=True, help='Learning Rate')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch Size')
    parser.add_argument('-d', '--dim', type=int, required=True, help='Embedding Dimension')
    parser.add_argument('-m', '--learnm', type=float, required=True, help='If args.m==-1 learn m')
    parser.add_argument('-l', '--loss', type=int, required=True, help='1-Original Triplet Center, 2-Modified')
    parser.add_argument('-n', '--network', type=str, required=True, help='Network Model')
    parser.add_argument('-max', '--maxpool', type=int, default=0, help='Use max-pool at the end')
    parser.add_argument('-min', '--usemin', type=int, default=1, help='Use min for V2 Loss')
    parser.add_argument('-norm', '--l2norm', type=int, default=0, help='Use normalization w/ linear activation')
    parser.add_argument('-data', '--dataset', type=str, default='cifar10', help='Choose dataset')
    parser.add_argument('-out', '--output', type=str, required=True, help='Output function')
    parser.add_argument('-r', '--resume', type=str, default="", help='Resume training')
    
    args = parser.parse_args()
    
    assert args.dataset in ['dogs', 'cifar10', 'cifar100', 'cub', 'cars', 'imagenet'], "Dataset not found" 
    
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
    
    if args.dataset == 'imagenet':
        BATCH_SIZE = 32
        
    assert args.network in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'inception'], "Model not found" 
    assert args.output in ['linear', 'tanh', 'softmax', 'sigmoid', 'twohead'], "Activation function not found" 
    
    useMaxPool = True if args.maxpool == 1 else False
    useMin = True if args.usemin == 1 else False
    useNorm = True if args.l2norm == 1 else False
    
    if args.loss == 1:
        loss_fn = tripletLoss
    elif args.loss == 2:
        loss_fn = expTripletLoss
    elif args.loss == 3:
        loss_fn = ratioTripletLoss
        
    m = None
        
    if args.output == 'tanh':
        l = 2
        m = 2*l/(CLASS_NUM**(1/DIM))
        m = l*np.sqrt(DIM)/2
    elif args.output == 'sigmoid':
        l = 1
        m = 2*l/(CLASS_NUM**(1/DIM))
        m = l*np.sqrt(DIM)/2
        
    SAVE_DIR = datetime.now().strftime("%m%d%Y_%H%M%S") + '_UNSUPERVISED_model_' + args.dataset + '_' + args.network + '_' + args.output + '_ep{}_lr{}_b{}_d{}_m{}_l{}_p{}_useNorm{}_norm{}/'.format(EPOCH, LR, BATCH_SIZE, DIM, args.learnm, args.loss, args.maxpool, args.usemin, args.l2norm)
    
    os.makedirs('models/' + SAVE_DIR)
     
    lowest_error = float('inf')
    best_ep = -1
    
    t_losses = []
    m_values = []
    START_EPOCH = 0
    checkpoint = None
    
    cfg = {
        'out_func': args.output,
        'use_norm': useNorm,
        'class_num': CLASS_NUM
    }
    
    sizes = {
        'cifar10': 32,
        'cub': 224,
        'cars': 224,
        'dogs': 224,
        'imagenet': 64
    }
    
    
    # -------------- MODEL PREPARATION --------------
    
    
    if args.resume != "":
        checkpoint = torch.load(args.resume)
        DIM = checkpoint['dim']
        
    if args.dataset in ['cifar10', 'cifar100']:
        
        if args.network == 'resnet18':
        
            model = torchvision.models.resnet18(pretrained=True)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = BranchNet(model.fc.in_features, DIM, args.output, sizes[args.dataset])
            
        elif args.network == 'inception':
            
            model = torchvision.models.googlenet(pretrained=True, progress=True, transform_input=True)
            model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = BranchNet(model.fc.in_features, DIM, args.output, sizes[args.dataset])
            
        
    else:
        
        print("Cub, Cars, Tiny ImageNet and Dogs model init.")
        
        if args.network == 'resnet50':
            
            model = torchvision.models.resnet50(pretrained=True)

            if args.dataset == 'imagenet':
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            model.fc = BranchNet(model.fc.in_features, DIM, args.output, sizes[args.dataset])
        
        elif args.network == 'inception':
            
            model = torchvision.models.googlenet(pretrained=True, progress=True, transform_input=True)
            
            if args.dataset == 'imagenet':
                model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                
            model.fc = BranchNet(model.fc.in_features, DIM, args.output, sizes[args.dataset])
        


    #model = RotateNet(embModel, DIM)
    model = model.cuda()
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    # -------------- RESUME TRAINING ---------------
    
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        START_EPOCH = checkpoint['epoch']
        BATCH_SIZE = checkpoint['batch_size']
        m_values = checkpoint['m_values']
        t_losses = checkpoint['train_loss']
        m = m_values[0]
        print("Margin:", m)
        print("Model loaded.")
    

    # ------------------ TRAINING ------------------
    
    
    ts = UnsupervisedDataset(trainset, sizes[args.dataset])
    trainloader = torch.utils.data.DataLoader(ts, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    
    pts = []
    
    with torch.no_grad():
        for idx, inputs in enumerate(trainloader):
            q_images = inputs['query'].cuda()
            q_out, q_recon = model.forward(q_images)
            pts.append(q_out.data)
    
    model.train()
             
    kmeans = KMeans(n_clusters=200, random_state=0, max_iter=1000).fit(torch.cat(pts).cpu().numpy())
    centers = torch.from_numpy(np.array(kmeans.cluster_centers_)).cuda()

    
    for epoch in range(START_EPOCH, START_EPOCH+EPOCH):
        
        center_distances = dict()
        pts = []
        class_ids = []
        
        model.eval()
        
        with torch.no_grad():
            for idx, inputs in enumerate(trainloader):
                q_images = inputs['query'].cuda()
                q_out, q_recon = model.forward(q_images)
                class_ids.append(kmeans.predict(q_out.cpu().data.numpy()))
        
        model.train()
        
        for i in range(centers.size()[0]):
            distances = (centers[i] - centers).pow(2).sum(dim=-1)
            distances[i] = 1e32
            center_distances[i] = distances.argmin()
            
 
        m_values.append(m)
        
        avg_t_loss = trainer.train(
            model, trainloader, optimizer, loss_fn, 
            t_losses, m, epoch, cfg, centers, kmeans, center_distances, pts, class_ids
        )
        
        kmeans = KMeans(n_clusters=200, random_state=0, max_iter=1000).fit(torch.cat(pts).numpy())
        centers = torch.from_numpy(np.array(kmeans.cluster_centers_)).cuda()
    
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

