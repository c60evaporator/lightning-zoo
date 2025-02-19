#%% Select the device
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 10
BATCH_SIZE = 32  # Effective Batch Size. 
NUM_WORKERS = 2  # 2 * Number of devices (GPUs) is appropriate in general, but this number doesn't matter in Object Detection.
DATA_ROOT = '../detection/datasets/VOC2012'
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.005  # Effective Learning Rate (https://lightning.ai/forums/t/effective-learning-rate-and-batch-size-with-lightning-in-ddp/101/2)
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9  # For SGD and RMSprop
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
MODEL_WEIGHT = 'deeplabv3_resnet50'

# Annotation Information
BORDER_IDX = 255 # Border index (None: No border, 255: VOC2012)
BG_IDX = 0

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

from lightning_zoo.datamodule.semantic_segmentation.voc import VOCSemanticSegDataModule

# ImageNet Normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Transforms for training
train_transform = A.Compose([
    A.Resize(520, 520),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ImageNet Normalization
    ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
])
# Transforms for validation and test
eval_transform = A.Compose([
    A.Resize(520, 520),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ImageNet Normalization
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
])

# Datamodule
datamodule = VOCSemanticSegDataModule(batch_size=int(BATCH_SIZE/NUM_GPU), num_workers=NUM_WORKERS, root=DATA_ROOT,
                                      dataset_name='VOC2012SemanticSeg',
                                      train_transforms=train_transform, eval_transforms=eval_transform,
                                      border_idx=BORDER_IDX, bg_idx=BG_IDX)
datamodule.prepare_data()
datamodule.setup()

# Validate the dataset
#datamodule.validate_dataset(output_normal_annotation=True, ignore_transforms=False)

# Display the first minibatch
datamodule.show_first_minibatch(image_set='train')

# %% Create PyTorch Lightning module
###### 3. Define the model (LightningModule) ######
from lightning_zoo.lightning.semantic_segmentation.deeplabv3 import DeepLabV3Module

model = DeepLabV3Module(class_to_idx=datamodule.class_to_idx, border_idx=datamodule.border_idx,
                        opt_name=OPT_NAME, lr=LR*NUM_GPU, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                        rmsprop_alpha=RMSPROP_ALPHA, adam_betas=ADAM_BETAS, eps=EPS,
                        lr_scheduler=LR_SCHEDULER, lr_gamma=LR_GAMMA, 
                        lr_step_size=LR_STEP_SIZE, lr_steps=LR_STEPS, lr_T_max=LR_T_MAX, lr_patience=LR_PATIENCE,
                        model_weight=MODEL_WEIGHT)

# %% Training
###### 4. Training (Trainer) ######
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

# CSV logger
logger = CSVLogger(save_dir=f'./log/{datamodule.dataset_name}/{model.model_name}',
                   name=model.model_weight, version=0)
trainer = Trainer(accelerator, devices=NUM_GPU, max_epochs=EPOCHS, logger=logger)
trainer.fit(model, datamodule=datamodule)
        
# %%
