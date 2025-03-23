from typing import TypedDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import os

from torch_extend.validate.common import validate_same_img_size

from ..base import TorchVisionDataModule

###### Annotation Validation TypeDicts for Classification ######
class ClsImageValidationResult(TypedDict):
    img_id: int
    img_width: int
    img_height: int
    label: int
    label_name: str
    anomaly: bool
    anomaly_label_idx: bool

###### Main Class ######
class ClassificationDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transform=None, train_target_transform=None,
                 eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None):
        super().__init__(batch_size, num_workers, dataset_name, 
                         None, train_transform, train_target_transform,
                         None, eval_transform, eval_target_transform,
                         out_fmt, processor)
        self.class_to_idx = None
        self.idx_to_class = None
    
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
        if not validate_same_img_size(self.train_transform):
            raise ValueError('The image size should be the same after the transforms for batch training. Please add `Resize` or `Crop` to `train_transform`.')
        self.same_img_size_train = True
        if validate_same_img_size(self.train_transform):
            self.same_img_size_eval = True
        else:
            self.same_img_size_eval = False
    
    ###### Display methods ######    
    def _show_image_and_target(self, img, target, image_set='train', denormalize=True, ax=None, anomaly_indices=None):
        # If ax is None, use matplotlib.pyplot.gca()
        if ax is None:
            ax=plt.gca()
        # Denormalize if normalization is included in transforms
        if denormalize:
            img = self.denormalize_image(img, image_set=image_set)
        img_permute = img.permute(1, 2, 0)
        ax.imshow(img_permute)  # Display the image
        class_ok = self.idx_to_class is not None and target.item() in self.idx_to_class.keys()
        ax.set_title(f'label: {self.idx_to_class[target.item()] if class_ok else target.item()}')

    ###### Validation methods ######
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        """
        Output an annotation file whose anomaly images are excluded.

        In the case of classification task, the normal images are saved aligned with `torchvision.datasets.ImageFolder` format. Please use `torchvision.datasets.ImageFolder` to load the normal images.
        """
        print('Exporting the filtered dataset...')
        df_norm_img_results: pd.DataFrame = df_img_results[~df_img_results['anomaly']]
        labels = list(self.idx_to_class.values())
        # Create image_set folder
        os.makedirs(f'{result_dir}/filtered_dataset', exist_ok=True)
        os.makedirs(f'{result_dir}/filtered_dataset/{image_set}', exist_ok=True)
        # Create label folder
        for label in labels:
            os.makedirs(f'{result_dir}/filtered_dataset/{image_set}/{label}', exist_ok=True)
        # Save the normal images
        for i, row in df_norm_img_results.iterrows():
            # Get the image and target from the dataset
            if image_set == 'train':
                img, target = self.train_dataset[row['img_id']]
            elif image_set == 'val':
                img, target = self.val_dataset[row['img_id']]
            elif image_set == 'test':
                img, target = self.test_dataset[row['img_id']]
            # Dataset iteration validation
            if target != row['label']:
                raise Exception('Dataset iteration is not aligned with the dataframe.')
            # Save the image
            save_image(img, f'{result_dir}/filtered_dataset/{image_set}/{self.idx_to_class[target]}/{row["img_id"]}.png')

    def _validate_annotation(self, imgs, targets, i_baches, batch_size, anomaly_save_path, denormalize, shuffle):
        """Validate the annotation"""
        if shuffle:
            raise ValueError('`shuffle` should be False for validation in classification task.')
        
        img_validations: list[ClsImageValidationResult] = []
        for i, (img, target) in enumerate(zip(imgs, targets)):
            # Image information
            img_result: ClsImageValidationResult = {}
            img_result['img_id'] = i_baches*batch_size + i
            img_result['img_width'] = img.size()[-1]
            img_result['img_height'] = img.size()[-2]
            img_result['label'] = target.item()
            img_result['anomaly'] = False
            # Label index validation
            if target.item() not in self.idx_to_class.keys():
                img_result['label_name'] = self.idx_to_class[target.item()]
                img_result['anomaly_label_idx'] = True
                img_result['anomaly'] = True
            else:
                img_result['label_name'] = ''
                img_result['anomaly_label_idx'] = False
            # Save the anomaly image
            if img_result['anomaly']:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                self._show_image_and_target(img, target, denormalize=denormalize, ax=ax)
                fig.savefig(f'{anomaly_save_path}/{img_result["img_id"]}.png')
                plt.show()
            img_validations.append(img_result)
        return img_validations, []

    ###### Transform Methods ######
    @property
    def default_train_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        # Based on TorchVision default transforms (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38)
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
            ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    @property
    def default_eval_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    @property
    def default_train_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        return None

    @property
    def default_train_target_transform(self) -> v2.Compose | A.Compose:
        """Default target_transform for training"""
        return None
    
    @property
    def default_eval_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for validation and test"""
        return None

    @property
    def default_eval_target_transform(self) -> v2.Compose | A.Compose:
        """Default target_transform for validation and test"""
        return None
