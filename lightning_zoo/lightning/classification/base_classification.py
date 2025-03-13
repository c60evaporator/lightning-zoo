import torch
import torch.nn as nn
from abc import abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.num_classes = max(self.class_to_idx.values()) + 1
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
    def _val_predict(self, batch):
        """Predict the validation batch"""
        # Predict the batch
        if isinstance(batch[0], torch.Tensor):  # Batch images of torch.Tensor
            preds = self.model(batch[0])
        elif isinstance(batch[0], tuple):  # Tuple of batch images by collate_fn
            preds = torch.stack([self.model(img.unsqueeze(0)).squeeze(0)
                                for img in batch[0]])
        # Get the targets
        if isinstance(batch[1], torch.Tensor):
            targets = batch[1]
        elif isinstance(batch[1], tuple):
            targets = torch.tensor(batch[1], dtype=torch.long)
        return preds, targets
    
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the batch"""
        # Calculate losses in the same way as training.
        return self.criterion(preds, targets)
    
    def _get_preds_cpu(self, preds):
        """Get the predictions and store them to CPU as a list"""
        return [pred.cpu() for pred in preds]

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
        self.confusion_matrix = confusion_matrix(targets, predicted_labels)
        return {'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro}
    
    ##### Display ######
    def _plot_predictions(self, images, preds, targets, n_images=10):
        """Plot the images with predictions and ground truths"""
        for i, (img, pred, target) in enumerate(zip(images, preds, targets)):
            predicted_label = torch.argmax(pred).item()
            img_permute = img.permute(1, 2, 0)
            plt.imshow(img_permute)
            plt.title(f'pred: {self.idx_to_class[predicted_label]}, true: {self.idx_to_class[target.item()]}')
            plt.show()
            if i >= n_images:
                break

    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        if metric_name is None:
            metric_name = 'confusion_matrix'
        # Plot the IOUs of each class
        if metric_name == 'confusion_matrix':
            df_cm = pd.DataFrame(self.confusion_matrix, 
                                 index=[self.idx_to_class[i] for i in range(len(self.idx_to_class))], 
                                 columns=[self.idx_to_class[i] for i in range(len(self.idx_to_class))])
            plt.figure(figsize=(len(self.idx_to_class), len(self.idx_to_class)*0.8))
            sns.heatmap(df_cm, annot=True, fmt=".5g", cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
        # Print the accuracies of each class
        elif metric_name == 'accuracy':
            print('Accuracy of each class')
            class_accuracies = df_cm.values.diagonal() / df_cm.sum(axis=1).values
            class_accuracies = sorted(class_accuracies)
            print(pd.Series(class_accuracies, index=self.idx_to_class.values()))
