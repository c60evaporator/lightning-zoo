import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import cv2

from torch_extend.dataset import CIFAR10TV

from .base_classification import ClassificationDataModule

###### Main Class ######
class CIFAR10DataModule(ClassificationDataModule):
    def __init__(self, batch_size, num_workers,
                 root, download=True,
                 dataset_name='CIFAR10',
                 train_transform=None, train_target_transform=None,
                 eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None):
        super().__init__(batch_size, num_workers, dataset_name, 
                         train_transform, train_target_transform,
                         eval_transform, eval_target_transform,
                         out_fmt, processor)
        self.root = root
        self.download = download
    
    ###### Dataset Methods ######
    def _get_datasets(self, ignore_transforms=False):
        """Get Train/Validation/Test datasets"""
        train_dataset = CIFAR10TV(
            self.root,
            train=True,
            transform=self._get_transform('train', ignore_transforms),
            target_transform=self._get_target_transform('train', ignore_transforms),
            download=self.download
        )
        val_dataset = CIFAR10TV(
            self.root,
            train=False,
            transform=self._get_transform('val', ignore_transforms),
            target_transform=self._get_target_transform('val', ignore_transforms),
            download=self.download
        )
        test_dataset = CIFAR10TV(
            self.root,
            train=False,
            transform=self._get_transform('test', ignore_transforms),
            target_transform=self._get_target_transform('test', ignore_transforms),
            download=self.download
        )
        return train_dataset, val_dataset, test_dataset
    
    ###### Validation methods ######

    ###### Transform Methods ######
    @property
    def default_train_transform(self) -> v2.Compose | A.Compose:
        """Default transform for training (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)"""
        return A.Compose([
            A.Resize(32,32),
            A.HorizontalFlip(),
            A.Rotate(limit=10, interpolation=cv2.INTER_NEAREST),
            A.Affine(rotate=0, shear=10, scale=(0.8,1.2)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization to [-1, 1]
            ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
        ])
    
    @property
    def default_eval_transform(self) -> v2.Compose | A.Compose:
        """Default transform for validation and test (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)"""
        return A.Compose([
            A.Resize(32,32),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
