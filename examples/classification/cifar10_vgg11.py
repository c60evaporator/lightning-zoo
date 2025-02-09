#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
# Add the root directory of the repository to system pathes
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch

# General Parameters
EPOCHS = 20
BATCH_SIZE = 128
NUM_WORKERS = 4
DATA_ROOT = './datasets/CIFAR10'
PRETRAINED = True
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Select the device
DEVICE = 'cuda'
if DEVICE == 'cuda':
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'mps':
    accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    accelerator = 'cpu'
# Set the random seed
torch.manual_seed(42)
# Multi GPU (https://github.com/pytorch/pytorch/issues/40403)
NUM_GPU = 1

# %% Define DataModule
###### 2. Define the dataset (DataModule) ######
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from lightning_zoo.datamodule.classification.cifar import CIFAR10DataModule

# Preprocessing (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])

# Datamodule
datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATA_ROOT,
                               train_transform=train_transform, eval_transform=eval_transform)
datamodule.setup()
# Validate the dataset
#datamodule.validate_dataset(output_normal_annotation=True)

# Display the first minibatch
datamodule.show_first_minibatch(image_set='train')

# %% Create PyTorch Lightning module
###### 3. Define the model (LightningModule) ######
from lightning_zoo.lightning.classification.vgg import VGGModule

model = VGGModule(class_to_idx=datamodule.class_to_idx, opt_name='sgd', pretrained=PRETRAINED)

# %% Training
###### 4. Training (Trainer) ######
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt

# CSV logger
logger = CSVLogger(save_dir=f'./log/{datamodule.dataset_name}/{model.model_name}',
                   name=model.model_weight)
trainer = Trainer(accelerator, devices=NUM_GPU, max_epochs=EPOCHS,
                  logger=logger, profiler="simple")
trainer.fit(model, datamodule=datamodule)

# Show the training results
model.plot_train_history()
plt.show()

# %% Test
trainer.test(model, datamodule=datamodule)
        
# %%
