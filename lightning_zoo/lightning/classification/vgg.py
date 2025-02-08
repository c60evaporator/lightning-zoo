import torch.nn as nn
from torchvision import models

from .base_classification import ClassificationModule

class VGGModule(ClassificationModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 model_weight='vgg11', dropout=0.5):
        super().__init__(class_to_idx,
                         'vgg',
                         criterion, pretrained, tuned_layers,
                         opt_name, lr, momentum, weight_decay, rmsprop_alpha, adam_betas, eps,
                         lr_scheduler, lr_step_size, lr_steps, lr_gamma, lr_T_max, lr_patience)
        self.model_weight = model_weight
        self.model: models.VGG
        # Model parameters
        self.dropout = dropout

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load VGG model based on the `model_name`"""
        if self.model_weight == 'vgg11':
            model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg11_bn':
            model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg13':
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg13_bn':
            model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg16_bn':
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_weight == 'vgg19_bn':
            model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        else:
            raise RuntimeError(f'Invalid `model_weight` {self.model_weight}.')
        return model
    
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Replace the classifier
        num_classes = max(self.class_to_idx.values()) + 1
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, num_classes),
        )
        # Initialize the weights
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
