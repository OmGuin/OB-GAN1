import torch

#hyperparameters
BATCH_SIZE = 8
RESIZE_TO = 1024 
NUM_EPOCHS = 15 
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'FinalDataset/train'
# validation images and XML files directory
VALID_DIR = 'FinalDataset/valid'

# classes: 0 index is reserved forW background
CLASSES = [
    '__background__', 'Nodule'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'modelandplots'