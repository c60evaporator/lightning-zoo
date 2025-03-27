import torch
from abc import abstractmethod
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from torch_extend.metrics.semantic_segmentation import segmentation_ious
from torch_extend.display.semantic_segmentation import show_predicted_segmentations

from ..base import TorchVisionModule

class SemanticSegModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         False)
        # Index to class dict
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Class to index dict
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.num_classes = max(self.class_to_idx.values()) + 1

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
        """Calculate the validation loss from the predictions and targets"""
        return None
    
    def _get_preds_cpu(self, preds):
        """Get the predictions and store them to CPU as a list"""
        # Raw logits data uses too much memory, so only store the predicted labels
        if isinstance(preds, torch.Tensor):  # Batch images of torch.Tensor
            return [pred.argmax(0) for pred in preds.cpu()]
        elif isinstance(preds, tuple):  # Tuple images by collate_fn
            return [pred.argmax(0).cpu() for pred in preds]
    
    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        return [target.cpu() for target in targets]
    
    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        # Calculate the mean Average Precision
        tps, fps, fns, mean_ious, label_indices, confmat = segmentation_ious(
            preds, targets, self.idx_to_class, pred_type='label',
            bg_idx=self.trainer.datamodule.bg_idx, 
            border_idx=self.trainer.datamodule.border_idx,
            confmat_calc_platform='torch'
        )
        mean_iou = np.mean(mean_ious)
        self.ious = {
            'per_class': {
                label: {
                    'label_index': label,
                    'label_name': self.idx_to_class[label] if label in self.idx_to_class.keys() else 'background' if label == self.trainer.datamodule.bg_idx else 'unknown',
                    'tps': tps[i],
                    'fps': fps[i],
                    'fns': fns[i],
                    'iou': mean_ious[i],
                }
                for i, label in enumerate(label_indices)
            },
            'confmat': confmat
        }
        return {'mean_iou': mean_iou}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets, n_images=4) -> list[Figure]:
        """Plot the images with predictions and ground truths"""
        figures = show_predicted_segmentations(images, preds, targets, self.idx_to_class,
                                               bg_idx=self.trainer.datamodule.bg_idx, 
                                               border_idx=self.trainer.datamodule.border_idx,
                                               plot_raw_image=True,
                                               max_displayed_images=n_images)
        return figures
        
    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        if metric_name is None:
            metric_name = 'iou'
        # Plot the IOUs of each class
        if metric_name == 'iou':
            print(pd.DataFrame([v for k, v in self.ious.items()]))
