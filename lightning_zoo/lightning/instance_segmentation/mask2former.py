import torch
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

from torch_extend.data_converter.instance_segmentation import convert_batch_to_torchvision

from .base_instance import InstanceSegModule

class Mask2FormerModule(InstanceSegModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='adamw', lr=2e-5, weight_decay=1e-4, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 semantic_metrics_score_threshold=0.2,
                 model_weight='facebook/mask2former-swin-small-coco-instance', no_object_weight=0.1, dice_weight=5.0, class_weight=2.0, mask_weight=5.0,
                 post_process_score_threshold=0.1):
        super().__init__(class_to_idx,
                         'mask2former',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         semantic_metrics_score_threshold)
        self.model_weight = model_weight
        self.model: Mask2FormerForUniversalSegmentation
        # Model parameters
        self.no_object_weight = no_object_weight
        self.dice_weight = dice_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.post_process_score_threshold = post_process_score_threshold
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load Mask2Former model based on the `model_weight`"""
        if self.pretrained:
            # Add the background index (0) if the labels are not reduced
            if 0 not in self.idx_to_class.values():
                id2label = {0: 'background', **self.idx_to_class}
            else:
                id2label = self.idx_to_class
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_weight,
                id2label=id2label,
                ignore_mismatched_sizes=True,
                no_object_weight=self.no_object_weight,
                dice_weight=self.dice_weight,
                class_weight=self.class_weight,
                mask_weight=self.mask_weight)
        else:
            config = Mask2FormerConfig(use_pretrained_backbone=False,
                                       id2label=id2label)
            model = Mask2FormerForUniversalSegmentation(config)
        return model

    @property
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Transformers models need to specify transferred and fine-tuned layers when initializing the model instance.
        pass

    def _set_model_and_params(self) -> torch.nn.Module:
        # Set the model
        self.model = self._get_model()
        # Transformers models need to specify transferred and fine-tuned layers when initializing the model instance.
    
    ###### Training ######    
    def _default_criterion(self, outputs):
        """Default criterion (Sum of all the losses)"""
        return outputs.loss
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        mask_labels = batch["mask_labels"]
        class_labels = batch["class_labels"]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                             mask_labels=mask_labels, class_labels=class_labels)
        return self.criterion(outputs)
    
    ###### Validation ######
    def _val_predict(self, batch):
        """Predict the validation batch"""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        mask_labels = batch["mask_labels"]
        class_labels = batch["class_labels"]
        preds = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                           mask_labels=mask_labels, class_labels=class_labels)
        return preds, {'mask_labels': mask_labels, 'class_labels': class_labels}
    
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return self.criterion(preds)
    
    def _convert_preds_targets_to_torchvision(self, preds, targets):
        """Convert the predictions and targets to TorchVision format"""    
        # Post-process the predictions
        target_sizes = [target.shape[-2:] for target in targets["mask_labels"]]
        results = self.trainer.datamodule.processor.post_process_instance_segmentation(
            preds, target_sizes=target_sizes,
            threshold=self.post_process_score_threshold, return_binary_maps=True
        )
        preds = []
        for result, target_size in zip(results, target_sizes):
            # Extract result whose area is 0
            areas = [mask.sum() for mask in result['segmentation']]
            instance_ids = [seg_info['id'] for seg_info, area in zip(result['segments_info'], areas) if area > 0]
            if len(instance_ids) != len(result['segments_info']):
                print('Some instances have area 0.')
            # Create masks and labels from the predictions
            if len(instance_ids) > 0:
                masks = torch.round(result['segmentation']).to(torch.uint8)
                labels = torch.tensor([seg_info['label_id'] for seg_info in result['segments_info']], dtype=torch.int64,
                                      device=self.device)
                scores = torch.tensor([seg_info['score'] for seg_info in result['segments_info']], dtype=torch.float32,
                                      device=self.device)
            else:
                masks = torch.empty((0, *target_size), dtype=torch.uint8, device=self.device)
                labels = torch.empty(0, dtype=torch.int64, device=self.device)
                scores = torch.empty(0, dtype=torch.float32, device=self.device)
                # print('No instance detected. Creating empty masks and labels.')

            # Create boxes from the predicted masks
            nonzero_masks = [mask.nonzero() for mask in masks]
            boxes = torch.tensor([
                [nonzero[:, 1].min(), nonzero[:, 0].min(),
                nonzero[:, 1].max(), nonzero[:, 0].max()]
                for nonzero in nonzero_masks
            ], dtype=torch.float32, device=self.device)
            preds.append({"masks": masks, "labels": labels, "scores": scores, "boxes": boxes})
        # Create masks and labels from the targets
        targets = [
            {"masks": masks.to(torch.uint8), "labels": labels}
            for masks, labels in zip(targets['mask_labels'], targets['class_labels'])
        ]
        # Create boxes from the target masks
        for target in targets:
            nonzero_masks = [mask.nonzero() for mask in target['masks']]
            target['boxes'] = torch.tensor([
                [nonzero[:, 1].min(), nonzero[:, 0].min(),
                nonzero[:, 1].max(), nonzero[:, 0].max()]
                for nonzero in nonzero_masks
            ], dtype=torch.float32, device=self.device)
        # Return as TorchVision format
        return preds, targets
    
    def _convert_images_for_pred_to_torchvision(self, batch):
        """Convert the displayed image to TorchVision format"""
        proc_imgs, _ = convert_batch_to_torchvision(batch, in_fmt='transformers')
        return proc_imgs
