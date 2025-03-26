from typing import TypedDict
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
from abc import abstractmethod

from torch_extend.display.instance_segmentation import show_instance_masks
from torch_extend.data_converter.instance_segmentation import convert_batch_to_torchvision

from ..base import TorchVisionDataModule

###### Main Class ######
class InstanceSegDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None):
        super().__init__(batch_size, num_workers, dataset_name, 
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform,
                         out_fmt, processor)
        self.class_to_idx = None
        self.idx_to_class = None

    ###### Dataset Methods ######
    def collate_fn_same_img_size(self, batch):
        """Collate function for the dataloader when the image sizes are the same"""
        # Convert the batch to the torchvision format
        if self.out_fmt == 'torchvision':
            return tuple(zip(*batch))
        # Convert the batch to the transformers (DETR) format
        elif self.out_fmt == 'transformers':
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
            mask_labels = [item['mask_labels'] for item in batch]
            class_labels = [item['class_labels'] for item in batch]
            return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

    def collate_fn_different_img_size(self, batch):
        """Collate function for the dataloader when the image sizes are not the same"""
        # Convert the batch to the torchvision format
        if self.out_fmt == 'torchvision':
            return tuple(zip(*batch))
        # Convert the batch to the transformers (DETR) format
        elif self.out_fmt == 'transformers':
            images = [item['images'] for item in batch]
            # Pad the images and masks
            segmentation_maps = [item['segmentation_maps'] for item in batch]
            instance_id_to_semantic_id = [item['instance_id_to_semantic_id'] for item in batch]
            encoding = self.processor.encode_inputs(images, segmentation_maps,
                                                    instance_id_to_semantic_id,
                                                    return_tensors="pt")
            return {"pixel_values": encoding['pixel_values'],
                    'pixel_mask': encoding['pixel_mask'],
                    "class_labels": encoding["class_labels"],
                    "mask_labels": encoding["mask_labels"]}
    
    def _setup(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
        # Set the background and the border index
        self.bg_idx = self.train_dataset.bg_idx if hasattr(self.train_dataset, 'bg_idx') else 0
        self.border_idx = self.train_dataset.border_idx if hasattr(self.train_dataset, 'border_idx') else 255
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
    
    ###### Display methods ######
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None, anomaly_indices=None):
        """Show the image and the target"""
        if denormalize:  # Denormalize if normalization is included in transforms
            img = self.denormalize_image(img, image_set=image_set)
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        boxes, labels, masks = target['boxes'], target['labels'], target['masks']
        show_instance_masks(img, masks=masks, boxes=boxes,
                            border_mask=target['border_mask'] if 'border_mask' in target else None,
                            labels=labels,
                            bg_idx=self.bg_idx, border_idx=self.border_idx,
                            idx_to_class=self.idx_to_class, ax=ax)
        
    def _convert_batch_to_torchvision(self, batch):
        """Convert the batch to the torchvision format images and targets"""
        if self.out_fmt == 'torchvision':
            return batch
        elif self.out_fmt == 'transformers':
            return convert_batch_to_torchvision(batch, in_fmt='transformers')

    ###### Validation methods ######
    @abstractmethod
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        """Output an annotation file whose anomaly images are excluded"""
        raise NotImplementedError

    def _validate_annotation(self, imgs, targets, i_baches, batch_size, anomaly_save_path, denormalize, shuffle):
        """Validate the annotation"""
        # TODO: Implement the method
        pass
    
    ###### Transform Methods ######
    @property
    def default_train_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        # Based on TorchVision default transforms (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L22)
        return A.Compose([
            A.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
            ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    @property
    def default_eval_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        return A.Compose([
            A.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

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
