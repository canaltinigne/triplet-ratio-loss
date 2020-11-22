python autoencoder_train.py -e 80 -lr 1e-4 -data cars -d 256 -m ae -out sigmoid
python autoencoder_train.py -e 80 -lr 1e-4 -data cars -d 256 -m vae -out tanh
python autoencoder_train.py -e 80 -lr 1e-4 -data cars -d 256 -m ae -out tanh
python autoencoder_train.py -e 80 -lr 1e-4 -data imagenet -d 512 -m ae -out tanh
python autoencoder_train.py -e 80 -lr 1e-4 -data cub -d 512 -m vae -out tanh
