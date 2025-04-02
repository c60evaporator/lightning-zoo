from torchvision.models.detection import faster_rcnn

from .base_detection import DetectionModule

class FasterRCNNModule(DetectionModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 save_first_prediction=True, num_saved_predictions=4,
                 first_epoch_lr_scheduled=False,
                 model_weight='fasterrcnn_resnet50_fpn'):
        super().__init__(class_to_idx,
                         'fasterrcnn',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         save_first_prediction, num_saved_predictions,
                         first_epoch_lr_scheduled)
        self.model_weight = model_weight
        self.model: faster_rcnn.FasterRCNN
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
