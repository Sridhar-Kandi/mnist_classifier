import torch

#-----Training Configuration-----#
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10


#-----Hardware Configuration-----#
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#-----Data Configuration-----#
DATA_DIR = './data' # Folder to download and store the MNIST data.

#-----Model Configuration-----#
SAVE_PATH = './models/mnist_model.pth' # Path to save the trained model.