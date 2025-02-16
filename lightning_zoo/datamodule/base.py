from lightning.pytorch import LightningDataModule
import albumentations as A
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import inspect

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
    def _get_transform(self, image_set, ignore_transforms=False):
        if ignore_transforms:
            return v2.ToTensor()
        elif image_set == 'train':
            return self.train_transform
        elif image_set == 'val' or image_set == 'test':
            return self.eval_transform
        
    def _get_target_transform(self, image_set, ignore_transforms=False):
        if ignore_transforms:
            return None
        elif image_set == 'train':
            return self.train_target_transform
        elif image_set == 'val' or image_set == 'test':
            return self.eval_target_transform
    
    def _get_transforms(self, image_set, ignore_transforms=False):
        if ignore_transforms:
            return None
        elif image_set == 'train':
            return self.train_transforms
        elif image_set == 'val' or image_set == 'test':
            return self.eval_transforms
            
    @abstractmethod
    def _get_datasets(self, ignore_transforms):
        """Get Train/Validation/Test datasets"""
        raise NotImplementedError
    
    @abstractmethod
    def _setup(self):
        """Dataset initialization"""
        raise NotImplementedError
    
    def setup(self, stage=None):
        self._setup()
    
    def train_dataloader(self) -> list[str]:
        """Create train dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
    
    def val_dataloader(self) -> list[str]:
        """Create validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
    
    def test_dataloader(self) -> list[str]:
        """Create test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
    
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
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None, anomaly_indices=None):
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
    @abstractmethod
    def _output_filtered_annotation(self, df_img_results, output_dir, image_set):
        """Output an annotation file whose anomaly images are excluded"""
        raise NotImplementedError
    
    @abstractmethod
    def _validate_annotation(self, imgs, targets, i_baches, batch_size, anomaly_save_path, denormalize, shuffle):
        """Validate the annotation"""
        raise NotImplementedError
    
    def _validate_dataset(self, image_set, dataloader, result_dir, output_normal_annotation, ignore_transforms):
        # Create anomaly image folder
        anomaly_image_path = f'{result_dir}/anomaly_images/{image_set}'
        os.makedirs(anomaly_image_path, exist_ok=True)
        # Validate the annotation of the dataset
        print(f'Validate the annotations of {image_set}_dataset')
        start = time.time()
        img_results = []
        target_results = []
        for i, (imgs, targets) in enumerate(dataloader):
            img_validations, target_validations = self._validate_annotation(imgs, targets, i, dataloader.batch_size, anomaly_image_path, 
                                                                            denormalize=not ignore_transforms,
                                                                            shuffle=isinstance(dataloader.sampler, RandomSampler))
            img_results.extend(img_validations)
            target_results.extend(target_validations)
            if i%100 == 0:  # Show progress every 100 times
                print(f'Validating the annotations of {image_set}_dataset: {i}/{len(dataloader)}, elapsed_time: {time.time() - start}')
        # Output the validation result
        df_img_results = pd.DataFrame(img_results)
        df_img_results.to_csv(f'{result_dir}/{image_set}_img_validation.csv')
        df_target_results = pd.DataFrame(target_results)
        df_target_results.to_csv(f'{result_dir}/{image_set}_target_validation.csv')
        print(f'Number of anomaly images in {image_set}_dataset: {df_img_results["anomaly"].sum()}')
        # Output a new annotation file whose anomaly images are excluded if `output_normal_annotation` is True
        if output_normal_annotation:
            self._output_filtered_annotation(df_img_results, result_dir, image_set)

    def validate_dataset(self, result_dir=None, output_normal_annotation=False, ignore_transforms=True,
                         validate_testset=False):
        """Validate the annotations"""
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError('Run the `setup()` method before the validation')
        if result_dir == None:
            result_dir = f'./ann_validation/{self.dataset_name}'
        # Get the datasets for the annotation validation
        if ignore_transforms:
            trainset, valset, testset = self._get_datasets(ignore_transforms=ignore_transforms)
            trainloader = DataLoader(trainset, batch_size=1, 
                                     shuffle=False, num_workers=self.num_workers,
                                     collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
            valloader = DataLoader(valset, batch_size=1, 
                                   shuffle=False, num_workers=self.num_workers,
                                   collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
            testloader = DataLoader(testset, batch_size=1,
                                    shuffle=False, num_workers=self.num_workers,
                                    collate_fn=self.collate_fn if 'collate_fn' in [mem[0] for mem in inspect.getmembers(self)] else None)
        else:
            trainloader = self.train_dataloader()
            valloader = self.val_dataloader()
            testloader = self.test_dataloader()
            
        os.makedirs(result_dir, exist_ok=True)

        # Validate the annotation of train_dataset
        self._validate_dataset('train', trainloader, result_dir, output_normal_annotation, ignore_transforms)
        # Validate the annotation of val_dataset
        self._validate_dataset('val', valloader, result_dir, output_normal_annotation, ignore_transforms)
        # Validate the annotation of test_dataset
        if validate_testset:
            self._validate_dataset('test', testloader, result_dir, output_normal_annotation, ignore_transforms)
    
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
