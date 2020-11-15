python autoencoder_train.py -e 80 -lr 1e-5 -data imagenet -d 512 -m ae -out sigmoid
python autoencoder_train.py -e 80 -lr 1e-4 -data cub -d 512 -m vae -out sigmoid
python autoencoder_train.py -e 80 -lr 1e-5 -data imagenet -d 512 -m ae -out tanh
