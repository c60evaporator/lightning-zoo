import torch
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2.functional as F
from transformers import DetrImageProcessor

from torch_extend.data_converter.detection import convert_batch_to_torchvision

from .base_detection import DetectionModule

class DetrModule(DetectionModule):
    def __init__(self, class_to_idx, processor,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 lr_backbone=1e-5,
                 first_epoch_lr_scheduled=False, n_batches=None,
                 ap_iou_threshold=0.5, ap_conf_threshold=0.0,
                 model_weight='fasterrcnn_resnet50_fpn'):
        super().__init__(class_to_idx,
                         'fasterrcnn',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         first_epoch_lr_scheduled, n_batches,
                         ap_iou_threshold, ap_conf_threshold)
        self.model_weight = model_weight
        self.model: faster_rcnn.FasterRCNN
        self.processor: DetrImageProcessor = processor
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load FasterRCNN model based on the `model_weight`"""
        if self.model_weight == 'fasterrcnn_resnet50_fpn':
            model = faster_rcnn.fasterrcnn_resnet50_fpn(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_resnet50_fpn_v2':
            model = faster_rcnn.fasterrcnn_resnet50_fpn_v2(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            model = faster_rcnn.fasterrcnn_mobilenet_v3_large_320_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_mobilenet_v3_large_fpn':
            model = faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if self.pretrained else None)
        else:
            raise RuntimeError(f'Invalid `model_weight` {self.model_weight}.')
        return model

    @property
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Replace the box_predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
    
    ###### Training ######    
    def _default_criterion(self, outputs):
        """Default criterion (Sum of all the losses)"""
        return sum(loss for loss in outputs.values())
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        outputs = self.model(inputs, targets)
        return self.criterion(outputs)
    
    ###### Validation ######
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return self.criterion(preds)
    
    def _convert_preds_targets_to_torchvision(self, preds, targets):
        """Convert the predictions and targets to TorchVision format"""
        # Post-process the predictions
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results = self.processor.post_process_object_detection(
            preds, target_sizes=orig_target_sizes, threshold=0
        )
        # Convert the targets
        targets = [{
            "boxes": box_convert(target["boxes"], 'cxcywh', 'xyxy') \
                    * torch.tensor([orig[1], orig[0], orig[1], orig[0]], dtype=torch.float32).to(self.device) if target["boxes"].shape[0] > 0 \
                    else torch.zeros(size=(0, 4), dtype=torch.float32).to(self.device),
            "labels": target["class_labels"]
        } for target, orig in zip(targets, orig_target_sizes)]
        # Return as TorchVision format
        return results, targets
    
    def _convert_images_to_torchvision(self, batch):
        """Convert the predictions and targets to TorchVision format"""
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
