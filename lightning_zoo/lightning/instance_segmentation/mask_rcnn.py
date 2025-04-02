from torchvision.models.detection import mask_rcnn, faster_rcnn

from .base_instance import InstanceSegModule

class MaskRCNNModule(InstanceSegModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 save_first_prediction=True, num_saved_predictions=4,
                 semantic_metrics_score_threshold=0.2,
                 model_weight='maskrcnn_resnet50_fpn'):
        super().__init__(class_to_idx,
                         'maskrcnn',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         save_first_prediction, num_saved_predictions,
                         semantic_metrics_score_threshold)
        self.model_weight = model_weight
        self.model: mask_rcnn.MaskRCNN
        # Save hyperparameters
        self.save_hyperparameters()

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load MaskRCNN model based on the `model_weight`"""
        if self.model_weight == 'maskrcnn_resnet50_fpn':
            model = mask_rcnn.maskrcnn_resnet50_fpn(weights=mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'maskrcnn_resnet50_fpn_v2':
            model = mask_rcnn.maskrcnn_resnet50_fpn_v2(weights=mask_rcnn.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if self.pretrained else None)
        else:
            raise RuntimeError(f'Invalid `model_weight` {self.model_weight}.')
        return model

    @property
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Replace layers for transfer learning
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.model.roi_heads.mask_predictor.conv5_mask.out_channels
        self.model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)
    
    ###### Training ######    
    def _default_criterion(self, outputs):
        """Default criterion (Sum of all the losses)"""
        return sum(loss for loss in outputs.values())
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        outputs = self.model(inputs, targets)
        return self.criterion(outputs)
