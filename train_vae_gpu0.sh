# python autoencoder_train.py -e 80 -lr 1e-4 -data cifar10 -d 512 -m vae -out sigmoid
# python autoencoder_train.py -e 80 -lr 1e-4 -data cifar10 -d 512 -m vae -out tanh
python autoencoder_train.py -e 80 -lr 1e-4 -data cub -d 128 -m vae -out sigmoid
python autoencoder_train.py -e 80 -lr 1e-4 -data cub -d 128 -m vae -out tanh
