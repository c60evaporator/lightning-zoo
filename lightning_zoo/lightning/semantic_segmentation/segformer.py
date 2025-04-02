import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from torch_extend.data_converter.semantic_segmentation import convert_batch_to_torchvision

from .base_semantic import SemanticSegModule

class SegformerModule(SemanticSegModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 save_first_prediction=True, num_saved_predictions=4,
                 model_weight='deeplabv3_resnet50'):
        super().__init__(class_to_idx,
                         'deeplabv3',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         save_first_prediction, num_saved_predictions)
        self.model_weight = model_weight
        self.model: SegformerForSemanticSegmentation
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load SegFormer model based on the `model_weight`"""
        if self.pretrained:
            # Add the background index (0) if the labels are not reduced
            if 0 not in self.idx_to_class.values():
                id2label = {0: 'background', **self.idx_to_class}
            else:
                id2label = self.idx_to_class
            model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_weight,
                id2label=id2label)
        else:
            config = SegformerConfig(use_pretrained_backbone=False,
                                     id2label=self.idx_to_class)
            model = SegformerForSemanticSegmentation(config)
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
        """Default criterion (Sum of cross entropy of out and aux outputs)"""
        return outputs.loss
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return self.criterion(outputs)
    
    ###### Validation ######
    def _val_predict(self, batch):
        """Predict the validation batch"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        preds = self.model(pixel_values=pixel_values, labels=labels)
        return preds, labels
    
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return self.criterion(preds)
    
    def _convert_preds_targets_to_torchvision(self, preds, targets):
        """Convert the predictions and targets to TorchVision format"""    
        # Post-process the predictions
        target_size = targets.shape[-2:]
        upsampled_logits = torch.nn.functional.interpolate(
            preds.logits, size=target_size, mode="bilinear", align_corners=False
        )
        return upsampled_logits, targets
    
    def _convert_images_for_pred_to_torchvision(self, batch):
        """Convert the displayed image to TorchVision format"""
        proc_imgs, _ = convert_batch_to_torchvision(batch, in_fmt='transformers')
        return proc_imgs
