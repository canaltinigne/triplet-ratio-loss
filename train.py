import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
import trainer
from models import *
import argparse
from datetime import datetime
from helpers import *
import os
from cub2011 import Cub2011
import PIL
from tcl_losses import TripletCenter40LossAllClass

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
    parser.add_argument('-a', '--adaptive', type=int, default=0, help='Adaptive center update rate')
    parser.add_argument('-dec', '--decay', type=float, required=True, help='Adaptive center update exponential decay rate')
    parser.add_argument('-reg', '--regularization', type=float, required=True, help='L2 reg. on embeddings')
    
    args = parser.parse_args()
    
    assert args.dataset in ['dogs', 'cifar10', 'cifar100', 'cub', 'cars', 'imagenet', 'flowers', 'aircraft'], "Dataset not found" 
    
    # -------------- DATASET PREPARATION --------------
    
    if args.dataset == 'cifar10':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261)
            )
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        
    elif args.dataset == 'cifar100':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])
        
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
        
    elif args.dataset == 'cub':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = Cub2011(root='./data', train=True, transform=transform)
        trainset.class_to_idx = range(200)
    
    elif args.dataset == 'cars':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/stanford-cars/car_data/car_data/train/', transform=transform)
        
    elif args.dataset == 'dogs':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/stanford-dogs/train/', transform=transform)
        
    elif args.dataset == 'imagenet':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/tiny-imagenet-200/train/', transform=transform)
        
    elif args.dataset == 'flowers':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/flowers-102/train/', transform=transform)
    
    elif args.dataset == 'aircraft':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224], interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageFolder(root='data/aircraft-100/train/', transform=transform)
        
    # -------------- INITIALIZATIONS ----------------
    
    
    CLASS_NUM = len(trainset.class_to_idx)
    DIM = args.dim
    EPOCH = args.epoch
    LR = args.rate
    BATCH_SIZE = args.batch
    DECAY_RATE = args.decay
    
    if args.dataset == 'imagenet':
        BATCH_SIZE = 32
        
    assert args.network in ['resnet18', 'resnet34', 'resnet50', 'resnet101'], "Model not found" 
    assert args.output in ['linear', 'tanh', 'softmax', 'sigmoid', 'twohead', 'relu'], "Activation function not found" 
    
    useMaxPool = True if args.maxpool == 1 else False
    useMin = True if args.usemin == 1 else False
    useNorm = True if args.l2norm == 1 else False
    
    #timer = {
    #    'center_distance': [],
    #    'backprop': [],
    #    'center_update': [],
    #    'loss': []
    #}
    
    if args.loss == 1:
        loss_fn = tripletCenterLoss
        centers = torch.from_numpy(np.eye(DIM))
    elif args.loss == 2:
        loss_fn = tripletCenterLossV2
        centers = torch.from_numpy(-np.ones((DIM,DIM)) + 2*np.eye(DIM))
    elif args.loss == 3:
        loss_fn = angularTripletLossUpdated
        centers = torch.from_numpy(np.eye(DIM))
        #centers = torch.empty((DIM,DIM)).normal_(mean=0,std=0.01)
        #centers = torch.from_numpy(-np.ones((DIM,DIM)) + 2*np.eye(DIM))
    elif args.loss == 4:
        loss_fn = originalAngularTripletLossUpdated
        centers = torch.empty((DIM,DIM)).normal_(mean=0,std=0.01)
    elif args.loss == 5:
        loss_fn = angularOursCombined
    elif args.loss == 6:
        loss_fn = sigmoidLoss
    elif args.loss == 7:
        loss_fn = logLoss
    elif args.loss == 8:
        loss_fn = sigmoidLossV2
    elif args.loss == 9:
        loss_fn = logLossV2
    elif args.loss == 10:
        loss_fn = logLossV3
    elif args.loss == 11:
        loss_fn = expLoss
        centers = torch.from_numpy(np.eye(DIM))
    elif args.loss == 12:
        loss_fn = ratioLoss
        centers = torch.from_numpy(np.eye(DIM))
    elif args.loss == 13:
        loss_fn = variantExpLoss
    elif args.loss == 14:
        loss_fn = taylorLoss
    elif args.loss == 15:
        loss_fn = expLossEuclidean
    elif args.loss == 16:
        loss_fn = taylorLossSquared
    elif args.loss == 18:
        loss_fn = originalTRL
        centers = torch.from_numpy(np.eye(DIM))
        

    #centers = torch.from_numpy(-np.ones((DIM,DIM)) + 2*np.eye(DIM))
    
    if DIM < CLASS_NUM:
        
        if args.loss in [4]:
            centers = torch.empty((CLASS_NUM, DIM)).normal_(mean=0,std=0.01)
        
        elif args.loss in [1,2,3,11,12,18]:
            if args.loss in [1, 3, 11, 12,18]:
                centers = torch.zeros(CLASS_NUM, DIM)
            elif args.loss in [2]: 
                centers = torch.from_numpy(-np.ones((CLASS_NUM, DIM)))

            for cl in range(CLASS_NUM):
                rand_ind = torch.randint(0, DIM, [4])
                centers[cl][rand_ind] = 1.

                
    if args.output == 'softmax':
        m = 2
    elif args.output == 'tanh':
        l = 2
        #m = 2*l/(CLASS_NUM**(1/DIM))
        m = l*np.sqrt(DIM)/2
    elif args.output == 'sigmoid':
        l = 1
        m = 2*l/(CLASS_NUM**(1/DIM))
        m = l*np.sqrt(DIM)/2
    elif args.output == 'linear':
        l = 2
        m = 2*l/(CLASS_NUM**(1/DIM))
        m = m**2
    elif args.output == 'twohead':
        l = 2
        m = 2*l/(CLASS_NUM**(1/DIM))
    elif args.output == 'relu':
        m = 0
        l = 1
        
    if args.loss == 17:
        loss_fn = TripletCenter40LossAllClass(5, CLASS_NUM, DIM).cuda()
        centers = torch.empty((DIM,DIM)).normal_(mean=0,std=0.01)
    elif args.loss == 19:
        loss_fn = expLossLearnable(CLASS_NUM, DIM).cuda()
        centers = torch.empty((DIM,DIM)).normal_(mean=0,std=0.01)
        
        
    centers = centers.type(torch.cuda.FloatTensor)
    print('Centers initialized.')
        
    SAVE_DIR = datetime.now().strftime("%m%d%Y_%H%M%S") + '_PAPER_model_' + args.dataset + '_' + args.network + '_' + args.output + '_ep{}_lr{}_b{}_d{}_m{}_l{}_decay{}_adaptive{}_norm{}_reg{}/'.format(EPOCH, LR, BATCH_SIZE, DIM, args.learnm, args.loss, args.decay, args.adaptive, args.l2norm, args.regularization)
    
    os.makedirs('models/' + SAVE_DIR)
     
    lowest_error = float('inf')
    best_ep = -1
    
    t_losses = []
    m_values = []
    centerChange = []
    START_EPOCH = 0
    checkpoint = None
    
    cfg = {
        'out_func': args.output,
        'use_norm': useNorm,
        'lossType': args.loss,
        'class_num': CLASS_NUM,
        'originalTCL': args.loss == 17 or args.loss == 19,
        'centerOptim': torch.optim.SGD(loss_fn.parameters(), lr=0.1) if (args.loss == 17 or args.loss == 19) else None,
        'reg': args.regularization
    }
    
    
    # -------------- MODEL PREPARATION --------------
    
    
    if args.resume != "":
        checkpoint = torch.load(args.resume)
        DIM = checkpoint['dim']
        
    if args.dataset in ['cifar10', 'cifar100']:
        if args.network == 'resnet18':
            model = ResNet18(DIM, useMaxPool, useNorm, args.output).cuda()
        elif args.network == 'resnet34':
            model = ResNet34(DIM, useMaxPool, useNorm, args.output).cuda()
        elif args.network == 'resnet50':
            model = ResNet50(DIM, useMaxPool, useNorm, args.output).cuda()
        elif args.network == 'resnet101':
            model = ResNet101(DIM, useMaxPool, useNorm, args.output).cuda()
    else:
        print("Cub, Cars, Tiny ImageNet, Flowers and Dogs model init.")
        
        model = torchvision.models.resnet50(pretrained=True)
        in_ftr  = model.fc.in_features
        out_ftr = DIM
        model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
        
        if args.dataset == 'imagenet':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            
        model = model.cuda()
    
  
    #optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    # -------------- RESUME TRAINING ---------------
    
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        START_EPOCH = checkpoint['epoch']
        BATCH_SIZE = checkpoint['batch_size']
        centerChange = checkpoint['centersByEpoch']
        centers = centerChange[-1].cuda()
        m_values = checkpoint['m_values']
        t_losses = checkpoint['train_loss']
        m = m_values[0]
        print("Margin:", m)
        print("Model loaded.")
    

    # ------------------ TRAINING ------------------
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
    for epoch in range(START_EPOCH, START_EPOCH+EPOCH):
        
        #torch.cuda.synchronize()
        #start_time = time.time()
        
        center_distances = None 
        angular_distances = None #timer
        
        if args.loss == 4:
            m = 0.7
        
        if args.loss in set([2,3,5,6,7,8,9,10,11,12,13,14,15,16,18,19]):
            
            center_distances = dict()
            angular_distances = dict()
        
            for classId in range(CLASS_NUM):
                
                diffClassInd = [x for x in range(CLASS_NUM) if x != classId]
                
                if args.loss == 2:
                    distances = ((centers[classId]-centers[diffClassInd])**2).sum(dim=-1)
                    center_distances[classId] = centers[diffClassInd][distances.argmin()]
                    
                    """
                    n = centers.size(0)
                    m_ = centers.size(0)
                    inputs = centers

                    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m_) + \
                        torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(m_, n).t()
                    dist.addmm_(1, -2, inputs, centers.t())
                    dist += (torch.eye(n)*1e32).cuda()
                    center_distances = dict(zip(range(CLASS_NUM), dist.argmin(dim=-1)))
                    break
                    """
                
                elif args.loss == 3:
                    distances = torch.acos((centers[classId] * centers).sum(dim=-1))
                    distances[classId] = 1e32
                    m = 0.7
                    center_distances[classId] = distances.argmin()
                
                elif args.loss == 5:
                    c_distances = ((centers[classId]-centers[diffClassInd])**2).sum(dim=-1)
                    a_distances = angularDistance(centers[classId], centers[diffClassInd])
                    
                    center_distances[classId] = centers[diffClassInd][c_distances.argmin()]
                    angular_distances[classId] = centers[diffClassInd][a_distances.argmin()]
                
                elif args.loss in set([6,7,8,9,10,11,12,13,14,15,16,18]):
                    distances = ((centers[classId]-centers[diffClassInd])**2).sum(dim=-1)
                    center_distances[classId] = centers[diffClassInd][distances.argmin()]
                 
                elif args.loss == 19:
                    distances = ((loss_fn.centers[classId]-loss_fn.centers)**2).sum(dim=-1)
                    distances[classId] = 1e32
                    center_distances[classId] = distances.argmin()
                    
                    
                
        c_classes = dict()
 
        m_values.append(m if args.learnm != -1 else abs(model.m.clone().data.item()))
        
        if not cfg['originalTCL']:
            centerChange.append(centers.clone().cpu().data)
        else:
            centerChange.append(loss_fn.centers.cpu().data)
            
        #torch.cuda.synchronize()
        #timer['center_distance'].append(time.time() - start_time)
        
        #torch.cuda.synchronize()
        #start_time = time.time()
        
        avg_t_loss = trainer.train(
            model, trainloader, optimizer, loss_fn, 
            t_losses, m, epoch, centers, c_classes, 
            cfg, center_distances, angular_distances
        )
    
        print('Epoch: {} | Loss: {:.4f}'.format(epoch+1, avg_t_loss))
        
        #torch.cuda.synchronize()
        #timer['backprop'].append(time.time() - start_time)
        
        #torch.cuda.synchronize()
        #start_time = time.time()
        
        if not (cfg['originalTCL'] or args.loss == 4):
            centerUpdate(centers, c_classes, CLASS_NUM, args.adaptive, epoch, DECAY_RATE, useNorm)
        
        if avg_t_loss <= lowest_error:
            lowest_error = avg_t_loss
            
            state = {
                'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': t_losses,
                'm_values': m_values,
                'centersByEpoch': centerChange,
                'dim': DIM,
                'lr': LR,
                'batch_size': BATCH_SIZE
            }

            if os.path.exists('models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(best_ep)):
                os.remove('models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(best_ep))

            torch.save(state, 'models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(epoch+1))
            
            best_ep = epoch+1
            
        #torch.cuda.synchronize()
        #timer['center_update'].append(time.time() - start_time)
    
    state = {
        'epoch': EPOCH, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': t_losses,
        'm_values': m_values,
        'centersByEpoch': centerChange,
        'dim': DIM,
        'lr': LR,
        'batch_size': BATCH_SIZE
    }

    torch.save(state, 'models/' + SAVE_DIR + 'ep_{}.pth.tar'.format(EPOCH))
    #torch.save(timer, 'models/' + SAVE_DIR + 'times.pth.tar') 
