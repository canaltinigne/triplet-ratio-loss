import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from tqdm import tqdm
from collections import defaultdict
from glob import glob
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score as nmi
import os
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from resnet import ResNet18, ResNet50
import PIL
from cub2011 import Cub2011
from branch_net import BranchNet
from autoencoders import UNet_AE, UNet_VAE


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

classes = {
    'cifar10': 10,
    'cifar100': 100,
    'cub': 200,
    'cars': 196,
    'dogs': 120,
    'imagenet': 200
}


das = {
    'cifar10': 'CIFAR-10',
    'cub': 'CUB',
    'cars': 'CARS',
    'dogs': 'DOGS'
}


ms = [
    '06152020_143931_UNSUPERVISED_model_cub_inception_sigmoid_ep100_lr0.0001_b16_d512_m0.1_l1_p0_useNorm1_norm0',
    '06152020_150839_UNSUPERVISED_model_cars_inception_sigmoid_ep100_lr0.0001_b16_d512_m0.1_l1_p0_useNorm1_norm0'
]

for z in ms:
    
    f = open(z + ".txt", "a")

    trainings = glob(z + '/*')
    trainings.sort(key=lambda c: int(c.split('ep_')[1].split('.')[0]))
    a = torch.load(trainings[0])
    
    print("Epoch: {}".format(int(trainings[0].split('ep_')[1].split('.')[0])), file=f)
    
    # PARAMETERS

    cfg = {
        'out_func': z.split("_")[6],
        'use_norm': int(z.split("_")[-1][-1]),
        'dataset': z.split("_")[4],
        'network': z.split("_")[5],
        'dim': int(z.split("_")[10][1:]),
        'loss': z.split("_")[-4],
        'date': z.split("_")[0]
    }

    if cfg['dataset'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=torchvision.transforms.ToTensor())
        
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=False, transform=torchvision.transforms.ToTensor())
        
    elif cfg['dataset'] == 'cub':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])

        trainset = Cub2011(root='../data', train=True, transform=transform)
        trainset.class_to_idx = range(200)
        testset = Cub2011(root='../data', train=False, transform=transform)
    
    elif cfg['dataset'] == 'cars':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/stanford-cars/car_data/car_data/train/', 
                                                    transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/stanford-cars/car_data/car_data/test/', 
                                                   transform=transform)
        
    elif cfg['dataset'] == 'dogs':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor()
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/stanford-dogs/train/', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/stanford-dogs/test/', transform=transform)
        
    elif cfg['dataset'] == 'imagenet':
        
        trainset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/train/',
                                                    transform=torchvision.transforms.ToTensor())
        
        testset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/val/',
                                                    transform=torchvision.transforms.ToTensor())
        
    LEVEL = 2 if cfg['dataset'] == 'cifar10' else 3
    
    EDGE = {
        'cifar10': 32,
        'cub': 224,
        'cars': 224,
        'dogs': 224,
        'imagenet': 64
    }
    
    if cfg['dataset'] in ['cifar10', 'cifar100']:
        
        if cfg['network'] in ['ae', 'vae']:
            
            if cfg['network'] == 'ae':
                model = UNet_AE(cfg['dim'], LEVEL, EDGE[cfg['dataset']], cfg['out_func'])
            
            elif cfg['network'] == 'vae':
                model = UNet_VAE(cfg['dim'], LEVEL, EDGE[cfg['dataset']], cfg['out_func'])
            
        else:

            if cfg['network'] == 'resnet18':

                model = torchvision.models.resnet18(pretrained=False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])

            elif cfg['network'] == 'inception':

                model = torchvision.models.googlenet(pretrained=True, progress=True, transform_input=True)
                model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])
                
            elif cfg['network'] == 'resnet50':
            
                model = torchvision.models.resnet50(pretrained=True)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])
 
    else:
        
        print("Cub, Cars, Tiny ImageNet and Dogs model init.")
        
        if cfg['network'] in ['ae', 'vae']:
            if cfg['network'] == 'ae':
                model = UNet_AE(cfg['dim'], LEVEL, EDGE[cfg['dataset']], cfg['out_func'])
            elif cfg['network'] == 'vae':
                model = UNet_VAE(cfg['dim'], LEVEL, EDGE[cfg['dataset']], cfg['out_func'])
        else:
            
            if cfg['network'] == 'resnet50':

                model = torchvision.models.resnet50(pretrained=True)

                if cfg['dataset'] == 'imagenet':
                    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])

            elif cfg['network'] == 'inception':

                model = torchvision.models.googlenet(pretrained=True, progress=True, transform_input=True)

                if cfg['dataset'] == 'imagenet':
                    model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])
                
            elif cfg['network'] == 'resnet18':

                model = torchvision.models.resnet18(pretrained=True)

                if cfg['dataset'] == 'imagenet':
                    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

                model.fc = BranchNet(model.fc.in_features, cfg['dim'], cfg['out_func'], EDGE[cfg['dataset']])
                    
   
    model.load_state_dict(a['state_dict'])
    model = model.cuda()
    model.eval()
    
    d = trainset
    
    print("Config", cfg)

    # ---------------------------------

    loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=False, num_workers=0)
    
    points=[]
    t = []

    with torch.no_grad():
        with tqdm(total=len(loader), dynamic_ncols=True) as progress:
            for idx, (inputs, target) in enumerate(loader):
                
                if cfg['network'] == 'ae':
                    _, out = model.forward(inputs.cuda())
                elif cfg['network'] == 'vae':
                    _, out, _ = model.forward(inputs.cuda())
                else:
                    out, _  = model.forward(inputs.cuda())

                for i, x in enumerate(out.cpu().data.numpy()):
                    points.append(x.tolist())
                    t.append(target[i].item())

                progress.update(1)
                
    neigh = KNeighborsClassifier(n_neighbors=1)
    train_X = points
    train_y = t
    neigh.fit(train_X, train_y)
     
    neigh_4 = NearestNeighbors(n_neighbors=4)
    neigh_4.fit(train_X)
    
    neigh_8 = NearestNeighbors(n_neighbors=8)
    neigh_8.fit(train_X)
    
    neigh_16 = NearestNeighbors(n_neighbors=16)
    neigh_16.fit(train_X)

    
    d = testset

    # ---------------------------------

    loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=False, num_workers=0)

    points=[]
    t = []

    with torch.no_grad():
        with tqdm(total=len(loader), dynamic_ncols=True) as progress:
            for idx, (inputs, target) in enumerate(loader):

                if cfg['network'] == 'ae':
                    _, out = model.forward(inputs.cuda())
                elif cfg['network'] == 'vae':
                    _, out, _ = model.forward(inputs.cuda())
                else:
                    out, _ = model.forward(inputs.cuda())

                for i, x in enumerate(out.cpu().data.numpy()):
                    points.append(x.tolist())
                    t.append(target[i].item())

                progress.update(1)
                
    print("Train:", len(train_X), len(train_y), file=f)
    print("Test:", len(points), len(t), file=f)

    p = neigh.predict(points)
    print("R@1:", accuracy_score(t,p), file=f)
        
    nes_4 = neigh_4.kneighbors(points)
    nes_8 = neigh_8.kneighbors(points)
    nes_16 = neigh_16.kneighbors(points)
    
    n4 = np.sum([1 if x in [train_y[y] for y in nes_4[1][i]] else 0 for i, x in enumerate(t)]) / len(t)
    n8 = np.sum([1 if x in [train_y[y] for y in nes_8[1][i]] else 0 for i, x in enumerate(t)]) / len(t)
    n16 = np.sum([1 if x in [train_y[y] for y in nes_16[1][i]] else 0 for i, x in enumerate(t)]) / len(t)
    
    print("R@4:", n4, file=f)
    print("R@8:", n8, file=f)
    print("R@16:", n16, file=f)
    
    
    if 'centersByEpoch' in a:
        centers = a['centersByEpoch'][-1].numpy()[:classes[cfg['dataset']]]
    
    x = np.array(points)

    print(x.shape, file=f)
    
    if 'centersByEpoch' in a:
        print(centers.shape, file=f)
    
    pred_cluster = []
    
    if 'centersByEpoch' in a:
        for p in x:
            pred_cluster.append(((p-centers)**2).sum(axis=-1).argmin())

        print("NMI: ", nmi(t, pred_cluster, average_method='arithmetic'), file=f)
    
    colors = cm.rainbow(np.linspace(0, 1, classes[cfg['dataset']]))

    
    X = np.array(points)
    colors = cm.rainbow(np.linspace(0, 1, classes[cfg['dataset']]))

    
    X_embedded = TSNE(n_components=2).fit_transform(X)
    
    p = X_embedded
    plt.figure(figsize=(6,6))
    plt.scatter(p[:,0], p[:,1],color=colors[t])
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.savefig(z + '.test.tsne.pdf', dpi=25, bbox_inches='tight')
    plt.clf()
    
    
    X_embedded = TSNE(n_components=2).fit_transform(np.array(train_X))
    
    p = X_embedded
    plt.figure(figsize=(6,6))
    plt.scatter(p[:,0], p[:,1],color=colors[train_y])
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.savefig(z + '.train.tsne.pdf', dpi=25, bbox_inches='tight')
    plt.clf()
    
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    
    p = embedding
    plt.figure(figsize=(6,6))
    plt.scatter(p[:,0], p[:,1],color=colors[t])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(z + '.test.umap.pdf', dpi=25, bbox_inches='tight')
    plt.clf()
    
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(np.array(train_X))
    
    p = embedding
    plt.figure(figsize=(6,6))
    plt.scatter(p[:,0], p[:,1],color=colors[train_y])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(z + '.train.umap.pdf', dpi=25, bbox_inches='tight')
    plt.clf()
    """
    
    plt.close('all')    
    f.close()
    print(z, 'finished')

