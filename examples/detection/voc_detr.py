#%% Select the device
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 4
BATCH_SIZE = 4  # Effective Batch Size. Bigger batch size increase the training time in Object Detection. Very mall batch size (E.g., n=1, 2) results in bad accuracy and poor Batch Normalization.
NUM_WORKERS = 2  # 2 * Number of devices (GPUs) is appropriate in general, but this number doesn't matter in Object Detection.
DATA_ROOT = './datasets/VOC2012'
# Optimizer Parameters
OPT_NAME = 'adamw'
LR = 2e-5  # Effective Learning Rate (https://lightning.ai/forums/t/effective-learning-rate-and-batch-size-with-lightning-in-ddp/101/2)
WEIGHT_DECAY = 1e-4
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
LR_BACKBONE = 5e-6  # Learning rate for the backbone
# Model Parameters
MODEL_WEIGHT = 'facebook/detr-resnet-50'

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
from transformers import DetrImageProcessor

from lightning_zoo.datamodule.detection.voc import VOCDetectionDataModule

# Image Processor (https://huggingface.co/docs/transformers/preprocessing#computer-vision)
image_processor = DetrImageProcessor.from_pretrained(MODEL_WEIGHT)

# Augmentation (Resize, Normalize, and ToTensor are not needed because the image_processor does it)
# Transforms for training
train_transform = A.Compose([
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
# Transforms for validation and test
eval_transform = A.Compose([
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# Datamodule
datamodule = VOCDetectionDataModule(batch_size=int(BATCH_SIZE/NUM_GPU), num_workers=NUM_WORKERS, root=DATA_ROOT,
                                    dataset_name='VOC2012Detection',
                                    train_transforms=train_transform, eval_transforms=eval_transform,
                                    out_fmt='transformers', processor=image_processor)
datamodule.prepare_data()
datamodule.setup()

# Validate the dataset
#datamodule.validate_dataset(output_normal_annotation=True, ignore_transforms=False)

# Display the first minibatch
datamodule.show_first_minibatch(image_set='train')

# %% Create PyTorch Lightning module
###### 3. Define the model (LightningModule) ######
from lightning_zoo.lightning.detection.detr import DetrModule

model = DetrModule(class_to_idx=datamodule.class_to_idx, 
                   opt_name=OPT_NAME, lr=LR*NUM_GPU, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                   rmsprop_alpha=RMSPROP_ALPHA, adam_betas=ADAM_BETAS, eps=EPS,
                   lr_scheduler=LR_SCHEDULER, lr_gamma=LR_GAMMA, 
                   lr_step_size=LR_STEP_SIZE, lr_steps=LR_STEPS, lr_T_max=LR_T_MAX, lr_patience=LR_PATIENCE,
                   model_weight=MODEL_WEIGHT, lr_backbone=LR_BACKBONE)

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
