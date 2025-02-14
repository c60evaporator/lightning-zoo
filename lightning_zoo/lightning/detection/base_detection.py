import torch
from abc import abstractmethod
import numpy as np

from ..base import TorchVisionModule
from torch_extend.metrics.detection import average_precisions
from torch_extend.display.detection import show_predicted_bboxes, show_average_precisions

class DetectionModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 first_epoch_lr_scheduled=False, n_batches=None,
                 ap_iou_threshold=0.5, ap_conf_threshold=0.0):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, momentum, weight_decay, rmsprop_alpha, adam_betas, eps,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         first_epoch_lr_scheduled, n_batches)
        # Class to index dict
        self.class_to_idx = class_to_idx
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Index to class dict with background
        self.idx_to_class_bg = {k: v for k, v in self.idx_to_class.items()}
        self.idx_to_class_bg[-1] = 'background'
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
    def _calc_val_loss(self, batch):
        """Calculate the validation loss from the batch"""
        return None
    
    def _get_preds_cpu(self, inputs):
        """Get the predictions and store them to CPU as a list"""
        return [{k: v.cpu() for k, v in pred.items()} 
                for pred in self.model(inputs)]
    
    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        return [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                for target in targets]
    
    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        # Calculate the mean Average Precision
        aps = average_precisions(preds, targets,
                                 self.idx_to_class_bg, 
                                 iou_threshold=self.ap_iou_threshold, conf_threshold=self.ap_conf_threshold)
        mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
        self.aps = aps
        print(f'mAP={mean_average_precision}')
        return {'mAP': mean_average_precision}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets):
        """Plot the images with predictions and ground truths"""
        show_predicted_bboxes(images, preds, targets, self.idx_to_class_bg)
    
    def plot_average_precisions(self):
        """Plot the average precisions"""
        show_average_precisions(self.aps)
