python train.py -e 100 -lr 1e-3 -d 10 -m 0.1 -l 17 -n resnet18 -data cifar10 -out linear 
python train.py -e 100 -lr 1e-4 -d 200 -m 0.1 -l 17 -n resnet50 -data cub -out linear 
python train.py -e 100 -lr 1e-5 -d 196 -m 0.1 -l 17 -n resnet50 -data cars -out linear 
python train.py -e 100 -lr 1e-4 -d 120 -m 0.1 -l 17 -n resnet50 -data dogs -out linear 
python train.py -e 100 -lr 1e-5 -d 200 -m 0.1 -l 17 -n resnet50 -data imagenet -out linear 