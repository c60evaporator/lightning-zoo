#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
# Add the root directory of the repository to system pathes
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch

# General Parameters
EPOCHS = 30
BATCH_SIZE = 128
NUM_WORKERS = 4
DATA_ROOT = './datasets/CIFAR10'
PRETRAINED = True
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.05
WEIGHT_DECAY = 0
MOMENTUM = 0  # For SGD and RMSprop
RMSPROP_ALPHA = 0.99  # For RMSprop
EPS = 1e-8  # For RMSprop, Adam, and AdamW
ADAM_BETAS = (0.9, 0.999)  # For Adam and AdamW
# LR Scheduler Parameters
LR_SCHEDULER = None
LR_GAMMA = 0.1
LR_STEP_SIZE = 8  # For StepLR
LR_STEPS = [16, 24]  # For MultiStepLR
LR_T_MAX = EPOCHS  # For CosineAnnealingLR
LR_PATIENCE = 10  # For ReduceLROnPlateau
# Model Parameters
DROPOUT = 0.5

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

# Transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Transforms for training
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
# Transforms for validation and test (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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

model = VGGModule(class_to_idx=datamodule.class_to_idx, pretrained=PRETRAINED,
                  opt_name=OPT_NAME, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                  rmsprop_alpha=RMSPROP_ALPHA, adam_betas=ADAM_BETAS, eps=EPS,
                  lr_scheduler=LR_SCHEDULER, lr_gamma=LR_GAMMA, 
                  lr_step_size=LR_STEP_SIZE, lr_steps=LR_STEPS, lr_T_max=LR_T_MAX, lr_patience=LR_PATIENCE,
                  dropout=DROPOUT)

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
