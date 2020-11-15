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


def cosineDistance(m1, m2):
    return 1 - (m1*m2).sum(dim=-1)/((m1**2).sum().sqrt() * (m2**2).sum(dim=-1).sqrt() + 1e-32)

def angularDistance(m1, m2):
    return torch.acos((m1 * m2).sum(dim=-1))
    

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

classes = {
    'cifar10': 10,
    'cifar100': 100,
    'cub': 200,
    'cars': 196,
    'dogs': 120,
    'imagenet': 200,
    'flowers': 102,
    'aircraft': 100
}


das = {
    'cifar10': 'CIFAR-10',
    'cub': 'CUB',
    'cars': 'CARS',
    'dogs': 'DOGS',
    'imagenet': 'IMAGENET',
    'flowers': 'FLOWERS',
    'aircraft': 'AIRCRAFT'
}


ms = [
    '07132020_134435_PAPER_model_imagenet_resnet50_linear_ep100_lr0.0001_b32_d200_m0.1_l19_decay0.05_adaptive1_norm0_reg0.1'
]



for z in ms:
    
    f = open(z + ".txt", "w+")

    trainings = glob(z + '/*')
    trainings.sort(key=lambda c: int(c.split('ep_')[1].split('.')[0]))
    a = torch.load(trainings[0])
    
    print("Epoch: {}".format(int(trainings[0].split('ep_')[1].split('.')[0])), file=f)
    
    # PARAMETERS

    cfg = {
        'out_func': z.split("_")[6],
        'use_norm': int(z.split("_")[15][-1]),
        'dataset': z.split("_")[4],
        'network': z.split("_")[5],
        'dim': int(z.split("_")[10][1:]),
        'loss': z.split("_")[12],
        'date': z.split("_")[0]
    }

    if cfg['dataset'] == 'cifar10':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261)
            )
        ])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=False, transform=transform)

    elif cfg['dataset'] == 'cifar100':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])

        trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                 download=True, transform=transform)

        testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                                download=True, transform=transform)

    elif cfg['dataset'] == 'cub':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = Cub2011(root='../data', train=True, transform=transform)
        testset = Cub2011(root='../data', train=False, transform=transform)
        trainset.class_to_idx = range(200)

    elif cfg['dataset'] == 'cars':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = torchvision.datasets.ImageFolder(root='../data/stanford-cars/car_data/car_data/train/',
                                                    transform=transform)
        
        testset = torchvision.datasets.ImageFolder(root='../data/stanford-cars/car_data/car_data/test/',
                                                   transform=transform)
        
    elif cfg['dataset'] == 'dogs':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/stanford-dogs/train/', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/stanford-dogs/test/', transform=transform)
        
    elif cfg['dataset'] == 'imagenet':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/train/', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/val/', transform=transform)
        
    elif cfg['dataset'] == 'flowers':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/flowers-102/train/', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/flowers-102/test/', transform=transform)
        
    elif cfg['dataset'] == 'aircraft':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/aircraft-100/train/', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../data/aircraft-100/test/', transform=transform)
    
            
    if cfg['dataset'] in ['cifar10', 'cifar100']:
        if cfg['network'] == 'resnet18':
            model = ResNet18(cfg['dim'], 0, 0, 'dummy').cuda()
    else:
        print("Cub, Cars, Tiny ImageNet and Dogs model init.")
        
        model = torchvision.models.resnet50(pretrained=False)
        in_ftr  = model.fc.in_features
        out_ftr = cfg['dim']
        model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
        
        if cfg['dataset'] == 'imagenet':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            
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

                for i, x in enumerate(out.cpu().data.numpy()):
                    points.append(x.tolist())
                    t.append(target[i].item())

                progress.update(1)
                
    # ----------- EUCLIDEAN-based -----------
    
    if cfg['loss'] in ['l1', 'l2', 'l17', 'l11', 'l18', 'l19']:
                    
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

        loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=False, num_workers=0)

        points=[]
        t = []

        with torch.no_grad():
            with tqdm(total=len(loader), dynamic_ncols=True) as progress:
                for idx, (inputs, target) in enumerate(loader):

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
                        elif cfg['out_func'] == 'twohead':
                            out = nn.Tanh()(out)
                        elif cfg['out_func'] == 'relu':
                            out = nn.ReLU()(out)

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

        test_X = np.array(points)
        
    # ----------- ANGULAR-based -----------
    
    elif cfg['loss'] in ['l3', 'l4']:
        
        d = testset
        train_X = points
        train_y = t

        loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=False, num_workers=0)

        points = []
        t = []

        with torch.no_grad():
            with tqdm(total=len(loader), dynamic_ncols=True) as progress:
                for idx, (inputs, target) in enumerate(loader):

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
                        elif cfg['out_func'] == 'twohead':
                            out = nn.Tanh()(out)
                        elif cfg['out_func'] == 'relu':
                            out = nn.ReLU()(out)

                    for i, x in enumerate(out.cpu().data.numpy()):
                        points.append(x.tolist())
                        t.append(target[i].item())

                    progress.update(1)

        print("Train:", len(train_X), len(train_y), file=f)
        print("Test:", len(points), len(t), file=f)
        
        nes_1 = 0.
        nes_4 = 0.
        nes_8 = 0.
        nes_16 = 0.
        
        np_test_X = torch.from_numpy(np.array(points))
        np_train_X = torch.from_numpy(np.array(train_X))
        np_train_y = np.array(train_y)
        np_test_y = np.array(t)
        
        for i, test_emb in enumerate(np_test_X):
            
            if cfg['date'] < '06012020':
                # COSINE-DISTANCE BASED
                distances = cosineDistance(test_emb, np_train_X)   
            else:
                # ANGULAR-DISTANCE BASED
                distances = angularDistance(test_emb, np_train_X)

            distances = distances.argsort()
            
            nes_1 += (np_test_y[i] == np_train_y[distances[0]])
            nes_4 += (np_test_y[i] in np_train_y[distances[:4]])
            nes_8 += (np_test_y[i] in np_train_y[distances[:8]])
            nes_16 += (np_test_y[i] in np_train_y[distances[:16]])
            
        print("R@1:", nes_1/len(np_test_y), file=f)
        print("R@4:", nes_4/len(np_test_y), file=f)
        print("R@8:", nes_8/len(np_test_y), file=f)
        print("R@16:", nes_16/len(np_test_y), file=f)
        
        if 'centersByEpoch' in a:
            centers = torch.from_numpy(a['centersByEpoch'][-1].numpy()[:classes[cfg['dataset']]]).double()
            
        x = np.array(points)

        print(x.shape, file=f)

        if 'centersByEpoch' in a:
            print(centers.shape, file=f)

        pred_cluster = []

        if 'centersByEpoch' in a:
            for test_emb in np_test_X:
                
                if cfg['date'] < '06012020':
                    # COSINE-DISTANCE BASED
                    distances = cosineDistance(test_emb, centers)   
                else:
                    # ANGULAR-DISTANCE BASED
                    distances = angularDistance(test_emb, centers)
                
                pred_cluster.append(distances.argmin().item())

            print("NMI: ", nmi(t, pred_cluster, average_method='arithmetic'), file=f)
            
        test_X = np.array(points)
        
    colors = cm.rainbow(np.linspace(0, 1, classes[cfg['dataset']]))
    
    X_embedded = TSNE(n_components=2).fit_transform(test_X)
    
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
    embedding = reducer.fit_transform(test_X)
    
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
