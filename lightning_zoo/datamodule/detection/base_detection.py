from typing import TypedDict
import torch
from torch.utils.data import DataLoader
import albumentations as A
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
from abc import abstractmethod

from ..base import TorchVisionDataModule
from torch_extend.display.detection import show_bounding_boxes

###### Annotation Validation TypeDicts for Object Detection ######
class DetImageValidationResult(TypedDict):
    img_id: int
    img_path: str
    img_width: int
    img_height: int
    n_boxes: int
    anomaly: bool

class DetBoxValidationResult(TypedDict):
    img_id: int
    img_path: str
    label: int
    label_name: str
    bbox: list[float]
    box_width: float
    box_height: float
    anomaly: bool
    anomaly_box_width: bool
    anomaly_box_height: bool
    anomaly_label_idx: bool

###### Main Class ######
class DetectionDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name, 
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform)
        self.class_to_idx = None
        self.idx_to_class = None
    
    ###### Dataset Methods ######
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
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
    
    ###### Display methods ######
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None, anomaly_indices=None):
        """Show the image and the target"""
        if denormalize:  # Denormalize if normalization is included in transforms
            img = self._denormalize_image(img, image_set=image_set)
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        boxes, labels = target['boxes'], target['labels']
        show_bounding_boxes(img, boxes, labels=labels,
                            idx_to_class=self.idx_to_class,
                            anomaly_indices=anomaly_indices, ax=ax)

    ###### Validation methods ######
    @abstractmethod
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        """Output an annotation file whose anomaly images are excluded"""
        raise NotImplementedError

    def _validate_annotation(self, imgs, targets, i_baches, batch_size, anomaly_save_path, denormalize):
        """Validate the annotation"""
        img_validations: list[DetImageValidationResult] = []
        box_validations: list[DetBoxValidationResult]  = []
        for img, target in zip(imgs, targets):
            # Image information
            img_result: DetImageValidationResult = {}
            image_id = int(os.path.splitext(os.path.basename(target['image_path']))[0])
            img_result['image_id'] = image_id
            img_result['image_path'] = target['image_path']
            img_result['image_width'] = img.size()[-1]
            img_result['image_height'] = img.size()[-2]
            img_result['n_boxes'] = len(target['boxes'])
            img_result['anomaly'] = False
            anomaly_indices = []
            # Bounding box (target) validation
            for i_box, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                box_list = box.tolist()
                box_result: DetBoxValidationResult = {}
                box_result['image_id'] = image_id
                box_result['image_path'] = target['image_path']
                box_result['label'] = label.item()
                box_result['label_name'] = str(box_result['label']) if self.idx_to_class is None else self.idx_to_class[box_result['label']]
                box_result['bbox'] = box_list
                box_result['box_width'] = box_list[2] - box_list[0]
                box_result['box_height'] = box_list[3] - box_list[1]
                # Negative box width
                box_result['anomaly_box_width'] = box_result['box_width'] <= 0
                # Negative box width
                box_result['anomaly_box_height'] = box_result['box_height'] <= 0
                # Label index validation
                box_result['anomaly_label_idx'] = box_result['label'] not in self.idx_to_class.keys()
                # Final anomaly judgement
                box_result['anomaly'] = box_result['anomaly_box_width'] or box_result['anomaly_box_height'] or box_result['anomaly_label_idx']
                if box_result['anomaly']:
                    img_result['anomaly'] = True
                    anomaly_indices.append(i_box)
                box_validations.append(box_result)
            # Save the anomaly image
            if img_result['anomaly']:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                self._show_image_and_target(img, target, denormalize=denormalize, ax=ax, anomaly_indices=anomaly_indices)
                fig.savefig(f'{anomaly_save_path}/{os.path.basename(target["image_path"])}')
                plt.show()
            img_validations.append(img_result)
        return img_validations, box_validations

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
