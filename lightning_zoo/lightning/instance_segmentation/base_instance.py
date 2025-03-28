import torch
from abc import abstractmethod
import numpy as np
from matplotlib.figure import Figure
from torchmetrics.detection import MeanAveragePrecision

from torch_extend.display.detection import show_average_precisions
from torch_extend.display.instance_segmentation import show_predicted_instances
from torch_extend.metrics.instance_segmentation import instance_mean_ious

from ..base import TorchVisionModule

class InstanceSegModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, weight_decay=None, momentum=None, rmsprop_alpha=None, eps=None, adam_betas=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 semantic_metrics_score_threshold=0.2):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, weight_decay, momentum, rmsprop_alpha, eps, adam_betas,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         False)
        self.semantic_metrics_score_threshold = semantic_metrics_score_threshold
        # Index to class dict
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        # Fill the missing indexes
        na_cnt = 0
        for i in range(1, max(self.idx_to_class.keys())):  # 0 is reserved for background
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
        return self.model(batch[0]), batch[1]

    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the predictions and targets"""
        return None
    
    def _convert_preds_targets_to_torchvision(self, preds, targets):
        """Convert the predictions and targets to TorchVision format"""
        # Mask float32(N, 1, H, W) -> uint8(N, H, W)
        preds = [{k: torch.round(v.squeeze(1)).to(torch.uint8)
                if k == 'masks' else v for k, v in pred.items()}
                for pred in preds]
        return preds, targets
    
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
        if self.trainer.num_devices == 1:
            map_metric = MeanAveragePrecision(iou_type=["bbox", "segm"], class_metrics=True, extended_summary=True)
        else:  # Avoid Runtime Error in DDP strategy (https://github.com/Lightning-AI/pytorch-lightning/issues/18803#issuecomment-2355778741)
            map_metric = MeanAveragePrecision(iou_type=["bbox", "segm"], class_metrics=True, extended_summary=True,
                                              compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True)
        map_metric.update(preds, targets)
        map_score = map_metric.compute()
        self.last_preds = preds
        self.last_targets = targets
        # Calculate semantic segmentation metrics
        tps, fps, fns, mean_ious, label_indices, confmat = instance_mean_ious(
            preds, targets, self.idx_to_class, 
            bg_idx=self.trainer.datamodule.bg_idx, 
            border_idx=self.trainer.datamodule.border_idx,
            score_threshold=self.semantic_metrics_score_threshold,
            confmat_calc_platform='torch'
        )
        semantic_mean_iou = np.mean(mean_ious)
        self.last_ious = {
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
        return {'MaskAP_50-95': map_score["segm_map"].item(), 'MaskAP_50': map_score["segm_map_50"].item(),
                'BoxAP_50-95': map_score["bbox_map"].item(), 'BoxAP_50': map_score["bbox_map_50"].item(),
                'SemanticMeanIoU': semantic_mean_iou}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets, n_images=10) -> list[Figure]:
        """Plot the images with predictions and ground truths"""
        figures = show_predicted_instances(images, preds, targets, self.idx_to_class,
                                           border_mask=targets['border_mask'] if 'border_mask' in targets else None,
                                           bg_idx=self.trainer.datamodule.bg_idx, border_idx=self.trainer.datamodule.border_idx,
                                           max_displayed_images=n_images)
        return figures
        
    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        if metric_name is None:
            metric_name = 'average_precision'
        # Plot the average precisions
        if metric_name == 'average_precision':
            show_average_precisions(self.last_preds, self.last_targets, self.idx_to_class)
