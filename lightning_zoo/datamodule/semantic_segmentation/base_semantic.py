from typing import TypedDict
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import abstractmethod

from torch_extend.display.semantic_segmentation import show_segmentations
from torch_extend.validate.common import validate_same_img_size

from ..base import TorchVisionDataModule

###### Annotation Validation TypeDicts for Semantic Segmentation ######
class SemanticSegImageValidationResult(TypedDict):
    img_id: int
    img_path: str
    img_width: int
    img_height: int
    n_labels: int
    anomaly: bool

class SemanticSegMaskValidationResult(TypedDict):
    img_id: int
    img_path: str
    label: int
    label_name: str
    area: int
    center_x: float
    center_y: float
    anomaly: bool
    anomaly_label_idx: bool

###### Main Class ######
class SemanticSegDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None,
                 border_idx=None, bg_idx=0):
        super().__init__(batch_size, num_workers, dataset_name,
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform,
                         out_fmt, processor)
        self.class_to_idx = None
        self.idx_to_class = None
        self.border_idx = border_idx
        self.bg_idx = bg_idx
    
    ###### Dataset Methods ######    
    def _setup(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
        # Class to index dict
        if 'class_to_idx' in vars(self.train_dataset):
            self.class_to_idx = self.train_dataset.class_to_idx
        else:
            raise ValueError('`class_to_idx` is not defined in the train_dataset. Please define `class_to_idx` as a member variable in the dataset class.')
        # Index to class dict
        if 'idx_to_class' in vars(self.train_dataset):
            self.idx_to_class = self.train_dataset.idx_to_class
        else:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            na_cnt = 0
            for i in range(max(self.class_to_idx.values())):
                if i not in self.class_to_idx.values():
                    na_cnt += 1
                    self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Same image size validation
        if not validate_same_img_size(self.train_transforms) and not validate_same_img_size(self.train_transform):
            raise ValueError('The image size should be the same after the transforms for batch training. Please add `Resize` or `Crop` to `train_transforms` or `train_transform`.')
        self.same_img_size_train = True
        if validate_same_img_size(self.train_transform) or validate_same_img_size(self.train_transforms):
            self.same_img_size_eval = True
        else:
            self.same_img_size_eval = False
    
    ###### Display methods ######
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None, anomaly_indices=None):
        """Show the image and the target"""
        if denormalize:  # Denormalize if normalization is included in transforms
            img = self.denormalize_image(img, image_set=image_set)
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        # Show the image with the target mask
        show_segmentations(img, target, self.idx_to_class, bg_idx=self.bg_idx, border_idx=self.border_idx)

    ###### Validation methods ######
    @abstractmethod
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        """Output an annotation file whose anomaly images are excluded"""
        raise NotImplementedError

    def _validate_annotation(self, imgs, targets, i_baches, batch_size, anomaly_save_path, denormalize, shuffle):
        """Validate the annotation"""
        if shuffle:
            raise ValueError('`shuffle` should be False for validation in semantic segmentation task.')
        
        img_validations: list[SemanticSegImageValidationResult] = []
        mask_validations: list[SemanticSegMaskValidationResult]  = []
        for i, (img, target) in enumerate(zip(imgs, targets)):
            # Image information
            img_result: SemanticSegImageValidationResult = {}
            image_path, mask_path = self.train_dataset.get_image_target_path(i_baches*batch_size + i)
            image_id = int(os.path.splitext(os.path.basename(image_path))[0])
            img_result['image_id'] = image_id
            img_result['image_path'] = image_path
            img_result['image_width'] = img.size()[-1]
            img_result['image_height'] = img.size()[-2]
            mask_labels = target.unique(sorted=True)  # Extract mask label list
            img_result['n_masks'] = len(mask_labels)
            img_result['anomaly'] = False
            anomaly_indices = []
            # Bounding box (target) validation
            for i_label, label in enumerate(mask_labels):
                mask_result: SemanticSegMaskValidationResult = {}
                mask_result['image_id'] = image_id
                mask_result['image_path'] = image_path
                mask_result['label'] = label.item()
                mask_result['label_name'] = str(mask_result['label']) if self.idx_to_class is None else self.idx_to_class[mask_result['label']]
                # Extract mask of the label
                mask = target == label
                x = torch.range(mask.size()[-1])
                hist_x = mask.sum(dim=0)
                y = torch.range(mask.size()[-2])
                hist_y = mask.sum(dim=1)
                area = mask.sum().item()
                mask_result['area'] = area
                mask_result['center_x'] = (hist_x * x).sum().item() / area
                mask_result['center_y'] = (hist_y * y).sum().item() / area
                # Label index validation
                mask_result['anomaly_label_idx'] = mask_result['label'] not in self.idx_to_class.keys()
                # Final anomaly judgement
                mask_result['anomaly'] = mask_result['anomaly_label_idx']
                if mask_result['anomaly']:
                    img_result['anomaly'] = True
                    anomaly_indices.append(i_label)
                mask_validations.append(mask_result)
            # Save the anomaly image
            if img_result['anomaly']:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                self._show_image_and_target(img, target, denormalize=denormalize, ax=ax, anomaly_indices=anomaly_indices)
                fig.savefig(f'{anomaly_save_path}/{os.path.basename(target["image_path"])}')
                plt.show()
            img_validations.append(img_result)
        return img_validations, mask_validations
    
    ###### Transform Methods ######
    @property
    def default_train_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        # Based on TorchVision default transforms (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L146)
        return A.Compose([
            A.Resize(520, 520),  # Resize the image to (520, 520)
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet Normalization
            ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
        ])
    
    @property
    def default_eval_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        return A.Compose([
            A.Resize(520, 520),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    @property
    def default_train_transform(self) -> v2.Compose | A.Compose:
        return None
    
    @property
    def default_train_target_transform(self) -> v2.Compose | A.Compose:
        return None
    
    @property
    def default_eval_transform(self) -> v2.Compose | A.Compose:
        return None
    
    @property
    def default_eval_target_transform(self) -> v2.Compose | A.Compose:
        return None
