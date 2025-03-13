from transformers import BaseImageProcessor
from torchvision.transforms import v2
from abc import ABC, abstractmethod

from .base_detection import DetectionDataModule
from torch_extend.data_converter.detection import convert_batch_to_torchvision

class TransformersDetectionDataModule(DetectionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None,
                 model_name=None, processor=None):
        super().__init__(batch_size, num_workers, dataset_name,
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform)
        self.model_name = model_name
        # Image processor
        if processor is None:
            self.processor: BaseImageProcessor = self.default_processor()
        else:
            self.processor: BaseImageProcessor = processor

    ###### Display methods ######
    def _denormalize_image(self, img, image_set='train'):
        """Denormalize the image for showing it"""
        # Denormalization based on the transforms
        img = super()._denormalize_image(img, image_set)
        # Denormalization based on the processor or Transformers
        if self.processor.do_normalize:
            denormalize_image = v2.Compose([
                v2.Normalize(mean=[-mean/std for mean, std in zip(self.processor.image_mean, self.processor.image_std)],
                            std=[1/std for std in self.processor.image_std])
            ])
            img = denormalize_image(img)
        return img
    
    def _convert_batch_to_torchvision(self, batch):
        """Convert the batch to the torchvision format image and target"""
        return convert_batch_to_torchvision(batch, in_fmt='transformers')
    
    @abstractmethod
    def default_processor(self) -> BaseImageProcessor:
        """Default transforms for training"""
        raise NotImplementedError
