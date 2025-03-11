import torch
from abc import abstractmethod
import numpy as np
import pandas as pd

from torch_extend.metrics.semantic_segmentation import segmentation_ious
from torch_extend.display.semantic_segmentation import show_predicted_segmentations

from ..base import TorchVisionModule

class SemanticSegModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int], border_idx,
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 first_epoch_lr_scheduled=False, n_batches=None):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, momentum, weight_decay, rmsprop_alpha, adam_betas, eps,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         first_epoch_lr_scheduled, n_batches)
        # Class to index dict
        self.class_to_idx = class_to_idx
        self.border_idx = border_idx  # Border index that is ignored in the evaluation
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'

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
        if isinstance(batch[0], torch.Tensor):
            preds = self.model(batch[0])
            if isinstance(preds, dict) and 'out' in preds.keys():
                preds = preds['out']
        elif isinstance(batch[0], tuple):
            preds = [self.model(img.unsqueeze(0))
                     for img in batch[0]]
            preds = tuple([pred.squeeze(0) if isinstance(pred, torch.Tensor) else pred['out'].squeeze(0)
                        for pred in preds])
        return preds, batch[1]
    
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the batch"""
        return None
    
    def _get_preds_cpu(self, preds):
        """Get the predictions and store them to CPU as a list"""
        if isinstance(preds, torch.Tensor):  # Batch images of torch.Tensor
            return [pred for pred in preds.cpu()]
        elif isinstance(preds, tuple):  # Tuple images by collate_fn
            return [pred.cpu() for pred in preds]
    
    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        return [target.cpu() for target in targets]
    
    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        # Calculate the mean Average Precision
        tps, fps, fns, ious = segmentation_ious(preds, targets, self.idx_to_class, self.border_idx)
        mean_iou = np.mean(ious)
        self.ious = {
            k: {
                'label_name': v,
                'tp': tps[i],
                'fp': fps[i],
                'fn': fns[i],
                'iou': ious[i]
            }
            for i, (k, v) in enumerate(self.idx_to_class.items())
        }
        return {'mean_iou': mean_iou}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets, n_images=10):
        """Plot the images with predictions and ground truths TODO: bg_idx should be from the datamodule (Trainer)"""
        show_predicted_segmentations(images, preds, targets, self.idx_to_class,
                                     bg_idx=0, border_idx=self.border_idx, plot_raw_image=True,
                                     max_displayed_images=n_images)
        
    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        if metric_name is None:
            metric_name = 'iou'
        # Plot the IOUs of each class
        if metric_name == 'iou':
            print(pd.DataFrame([v for k, v in self.ious.items()]))
    