import torch
from abc import abstractmethod
import numpy as np
from matplotlib.figure import Figure
from torchmetrics.detection import MeanAveragePrecision

from torch_extend.metrics.detection import average_precisions
from torch_extend.display.detection import show_predicted_bboxes, show_average_precisions

from ..base import TorchVisionModule

class DetectionModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 first_epoch_lr_scheduled=False, n_batches=None,
                 ap_iou_threshold=0.5, ap_conf_threshold=0.0):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         first_epoch_lr_scheduled, n_batches)
        # Class to index dict
        self.class_to_idx = class_to_idx
        self.num_classes = max(self.class_to_idx.values()) + 1
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Thresholds for AP validation
        self.ap_iou_threshold = ap_iou_threshold
        self.ap_conf_threshold = ap_conf_threshold

    ###### Set the model and the fine-tuning settings ######
    @property
    @abstractmethod
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        raise NotImplementedError
    
    @abstractmethod
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        raise NotImplementedError
    
    def _setup(self):
        """Additional processes during the setup"""
        pass
    
    ###### Validation ######
    def _val_predict(self, batch):
        """Predict the validation batch"""
        return self.model(batch[0]), batch[1]

    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return None
    
    def _get_preds_cpu(self, preds):
        """Get the predictions and store them to CPU as a list"""
        return [{k: v.cpu() for k, v in pred.items()}
                for pred in preds]
    
    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        return [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                for target in targets]
    
    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        # Calculate the mean Average Precision
        map_metric = MeanAveragePrecision(iou_type=["bbox"], class_metrics=True, extended_summary=True)
        map_metric.update(preds, targets)
        map_score = map_metric.compute()
        self.last_preds = preds
        self.last_targets = targets
        return {'BoxAP_50-95': map_score["map"].item(), 'BoxAP_50': map_score["map_50"].item(), 'BoxAP_75': map_score["map_75"].item()}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets, n_images=10) -> list[Figure]:
        """Plot the images with predictions and ground truths"""
        figures = show_predicted_bboxes(images, preds, targets, self.idx_to_class,
                                        max_displayed_images=n_images)
        return figures
        
    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        if metric_name is None:
            metric_name = 'average_precision'
        # Plot the average precisions
        if metric_name == 'average_precision':
            show_average_precisions(self.last_preds, self.last_targets, self.idx_to_class)
