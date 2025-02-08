from lightning.pytorch import LightningDataModule
import albumentations as A
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class TorchVisionDataModule(LightningDataModule, ABC):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        # Transform for training
        if train_transforms is None and train_transform is None and train_target_transform is None:
            self.train_transforms = self.default_train_transforms
            self.train_transform = self.default_train_transform
            self.train_target_transform = self.default_train_target_transform
        else:
            self.train_transforms = train_transforms
            self.train_transform = train_transform
            self.train_target_transform = train_target_transform
        # Transform for validation and test
        if eval_transforms is None and eval_transform is None and eval_target_transform is None:
            self.eval_transforms = self.default_eval_transform
            self.eval_transform = self.default_eval_transform
            self.eval_target_transform = self.default_eval_target_transform
        else:
            self.eval_transforms = eval_transforms
            self.eval_transform = eval_transform
            self.eval_target_transform = eval_target_transform
        # Check whether all the image sizes are the same during training
        self.same_img_size = False
        image_transform = self.train_transforms if self.train_transforms is not None else self.train_transform
        for tr in image_transform:
            if (isinstance(tr, v2.Resize) and len(tr.size) == 2) or isinstance(tr, A.Resize):
                self.same_img_size = True
                break
        # Other
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    ###### Dataset Methods ######
    @abstractmethod
    def _setup(self):
        """Dataset initialization"""
        raise NotImplementedError
    
    def setup(self, stage=None):
        self._setup()
    
    def train_dataloader(self) -> list[str]:
        """Create train dataloader"""
        raise NotImplementedError
    
    def val_dataloader(self) -> list[str]:
        """Create validation dataloader"""
        raise NotImplementedError
    
    def test_dataloader(self) -> list[str]:
        """Create test dataloader"""
        raise NotImplementedError
    
    ###### Display methods ######
    def _denormalize_image(self, img, image_set='train'):
        """Denormalize the image for showing it"""
        if image_set == 'train':
            image_transform = self.train_transforms if self.train_transforms is not None else self.train_transform
        elif image_set == 'val':
            image_transform = self.eval_transforms if self.eval_transforms is not None else self.eval_transform
        for tr in image_transform:
            if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
                reverse_transform = v2.Compose([
                    v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                        std=[1/std for std in tr.std])
                ])
                return reverse_transform(img)
        return img
    
    @abstractmethod
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None):
        """Show the image and the target"""
        raise NotImplementedError

    def show_first_minibatch(self, image_set='train'):
        # Check whether all the image sizes are the same
        if image_set == 'train':
            loader = self.train_dataloader()
        elif image_set == 'val':
            loader = self.val_dataloader()
        else:
            raise RuntimeError('The `image_set` argument should be "train" or "val"')
        batch_iter = iter(loader)
        imgs, targets = next(batch_iter)

        for i, (img, target) in enumerate(zip(imgs, targets)):
            self._show_image_and_target(img, target, image_set)
            plt.show()
    
    ###### Validation Methods ######
    
    ###### Transform Methods ######
    @property
    @abstractmethod
    def default_train_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for training"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def default_train_transform(self) -> v2.Compose | A.Compose:
        """Default transform for training"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def default_train_target_transform(self) -> v2.Compose | A.Compose:
        """Default target_transform for training"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def default_eval_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for validation and test"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def default_eval_transform(self) -> v2.Compose | A.Compose:
        """Default transform for validation and test"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def default_eval_target_transform(self) -> v2.Compose | A.Compose:
        """Default target_transform for validation and test"""
        raise NotImplementedError
