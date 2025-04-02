import torch
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2.functional as F
from transformers import DetrForObjectDetection, DetrConfig

from torch_extend.data_converter.detection import convert_batch_to_torchvision

from .base_detection import DetectionModule

class DetrModule(DetectionModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='adamw', lr=2e-5, weight_decay=1e-4, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 save_first_prediction=True, num_saved_predictions=4,
                 first_epoch_lr_scheduled=False,
                 model_weight='facebook/detr-resnet-50', lr_backbone=5e-6):
        super().__init__(class_to_idx,
                         'detr',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         save_first_prediction, num_saved_predictions,
                         first_epoch_lr_scheduled)
        self.model_weight = model_weight
        self.model: DetrForObjectDetection
        # Model parameters
        self.lr_backbone = lr_backbone
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load DETR model based on the `model_weight`"""
        if self.pretrained:
            model = DetrForObjectDetection.from_pretrained(self.model_weight,
                                                        revision="no_timm",
                                                        num_labels=len(self.idx_to_class),
                                                        ignore_mismatched_sizes=True)
        else:
            if 'resnet-50' in self.model_weight:
                backbone = 'resnet50'
            config = DetrConfig(use_pretrained_backbone=False, backbone=backbone,
                                id2label=self.idx_to_class)
            model = DetrForObjectDetection(config)
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
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return self.criterion(outputs)
    
    ###### Validation ######
    def _val_predict(self, batch):
        """Predict the validation batch"""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]
        preds = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return preds, [t for t in batch["labels"]]
    
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return self.criterion(preds)
    
    def _convert_preds_targets_to_torchvision(self, preds, targets):
        """Convert the predictions and targets to TorchVision format"""
        # Post-process the predictions
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results = self.trainer.datamodule.processor.post_process_object_detection(
            preds, target_sizes=orig_target_sizes, threshold=0
        )
        # Convert the targets
        targets = [{
            "boxes": box_convert(target["boxes"], 'cxcywh', 'xyxy') \
                    * torch.tensor([orig[1], orig[0], orig[1], orig[0]], dtype=torch.float32, device=self.device) if target["boxes"].shape[0] > 0 \
                    else torch.zeros(size=(0, 4), dtype=torch.float32, device=self.device),
            "labels": target["class_labels"]
        } for target, orig in zip(targets, orig_target_sizes)]
        # Return as TorchVision format
        return results, targets
    
    def _convert_images_for_pred_to_torchvision(self, batch):
        """Convert the displayed image to TorchVision format"""
        proc_imgs, _ = convert_batch_to_torchvision(batch, in_fmt='transformers')
        orig_sizes = [label["orig_size"] for label in batch["labels"]]
        return [F.resize(img, orig_size.tolist()) for img, orig_size in zip(proc_imgs, orig_sizes)]
    
    ##### Optimizers and Schedulers ######
    def _extract_optimizer_params(self):
        """Extract the parameters for the optimizer"""
        # Reference (https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)
        return [{"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                 "lr": self.lr_backbone}]
