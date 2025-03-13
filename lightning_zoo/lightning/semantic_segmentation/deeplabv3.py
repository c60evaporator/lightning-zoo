from torchvision.models.segmentation import deeplabv3, fcn
from torch import nn

from .base_semantic import SemanticSegModule

class DeepLabV3Module(SemanticSegModule):
    def __init__(self, class_to_idx, border_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 first_epoch_lr_scheduled=False, n_batches=None,
                 model_weight='deeplabv3_resnet50'):
        super().__init__(class_to_idx, border_idx,
                         'deeplabv3',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, momentum, weight_decay, rmsprop_alpha, adam_betas, eps,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         first_epoch_lr_scheduled, n_batches)
        self.model_weight = model_weight
        self.model: deeplabv3.DeepLabV3
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load DeepLabV3 model based on the `model_weight`"""
        if self.model_weight == 'deeplabv3_resnet50':
            model = deeplabv3.deeplabv3_resnet50(weights=deeplabv3.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if self.pretrained else None)
        elif self.model_weight == 'deeplabv3_resnet101':
            model = deeplabv3.deeplabv3_resnet101(weights=deeplabv3.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if self.pretrained else None)
        elif self.model_weight == 'deeplabv3_mobilenet_v3_large':
            model = deeplabv3.deeplabv3_mobilenet_v3_large(weights=deeplabv3.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if self.pretrained else None)
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
        self.model.aux_classifier = fcn.FCNHead(1024, self.num_classes)
        self.model.classifier = deeplabv3.DeepLabHead(2048, self.num_classes)
    
    ###### Training ######    
    def _default_criterion(self, outputs, targets):
        """Default criterion (Sum of cross entropy of out and aux outputs)"""
        losses = {}
        for name, x in outputs.items():
            losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=self.border_idx)
        if len(losses) == 1:
            return losses["out"]
        return losses["out"] + 0.5 * losses["aux"]
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)
