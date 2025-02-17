#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 40
BATCH_SIZE = 128  # Bigger batch size is faster but less accurate (https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU)
NUM_WORKERS = 4
DATA_ROOT = './datasets/CIFAR10'
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.03
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

# %% Define the dataset
###### 2. Define the dataset (DataModule) ######
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from lightning_zoo.datamodule.classification.cifar import CIFAR10DataModule

# Transforms for training (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization from uint8 [0, 255] to float32 [-1.0, 1.0]
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
])
# Transforms for validation and test (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization from uint8 [0, 255] to float32 [-1.0, 1.0]
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
])

# Datamodule
datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATA_ROOT,
                               train_transform=train_transform, eval_transform=eval_transform)
datamodule.setup()
#datamodule.validate_dataset(output_normal_annotation=True)

# Display the first minibatch
datamodule.show_first_minibatch(image_set='train')

# %% Create original PyTorch model
###### 3. Define the model (LightningModule) ######
from torch import nn

class LeNet(nn.Module):
    """LeNet model for CIFAR-10 (https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch)"""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.fetrures = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*64, 500),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(500, 10)
        )
    def forward(self, x):
      x = self.fetrures(x)
      x = x.view(-1, 4*4*64)
      x = self.classifier(x)
      return x

# %% Create original PyTorch Lightning module
# Original Lightning Module
from lightning_zoo.lightning.classification.base_classification import ClassificationModule

class LeNetModule(ClassificationModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 dropout=0.5):
        super().__init__(class_to_idx,
                         model_name='lenet',
                         criterion=criterion, pretrained=False, tuned_layers=None,
                         opt_name=opt_name, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         rmsprop_alpha=rmsprop_alpha, adam_betas=adam_betas, eps=eps,
                         lr_scheduler=lr_scheduler, lr_step_size=lr_step_size, lr_steps=lr_steps,
                         lr_gamma=lr_gamma, lr_T_max=lr_T_max, lr_patience=lr_patience)
        self.model: LeNet
        # Model parameters
        self.dropout = dropout

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        return LeNet(dropout=self.dropout)

    @property
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        pass

model = LeNetModule(class_to_idx=datamodule.class_to_idx, 
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
                   name=model.model_name)
trainer = Trainer(accelerator, devices=NUM_GPU, max_epochs=EPOCHS, 
                  logger=logger, profiler="simple")
trainer.fit(model, datamodule=datamodule)

# Show the training results
model.plot_train_history()

# %% Show the predictions with ground truths
model.plot_prediction_from_val_dataset()

# %% Test
trainer.test(model, datamodule=datamodule)

# %%
