import torch
import torch.nn as nn
from abc import abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..base import TorchVisionModule

class ClassificationModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None):
        super().__init__(model_name, criterion, pretrained, tuned_layers,
                         opt_name, lr, momentum, weight_decay, rmsprop_alpha, adam_betas, eps,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience,
                         False, None)
        # Class to index dict
        self.class_to_idx = class_to_idx
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'

    ###### Set the model and the fine-tuning settings ######
    @property
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        pass
    
    def _setup(self):
        """Additional processes during the setup (E.g., List for storing predictions in validation)"""
        pass

    ###### Training ######
    @property
    def _default_criterion(self):
        """Default criterion (Sum of all the losses)"""
        return nn.CrossEntropyLoss()
    
    def _calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)
    
    ###### Validation ######
    def _calc_val_loss(self, batch):
        """Calculate the validation loss from the batch"""
        # Calculate losses in the same way as training.
        return self._calc_train_loss(batch)
    
    def _get_preds_cpu(self, inputs):
        """Get the predictions and store them to CPU as a list"""
        return [pred.cpu() for pred in self.model(inputs)]

    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        return [target.item() for target in targets]

    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        # Calculate the accuracy, precision, recall, and f1 score
        predicted_labels = [torch.argmax(pred).item() for pred in preds]
        accuracy = accuracy_score(targets, predicted_labels)
        precision_macro = precision_score(targets, predicted_labels, average='macro')
        recall_macro = recall_score(targets, predicted_labels, average='macro')
        f1_macro = f1_score(targets, predicted_labels, average='macro')
        return {'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro}
