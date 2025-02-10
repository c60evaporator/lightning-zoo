# Define the Model (LightningModule)

## Basic Usage

At least, the following factors should be specified

- [Model Type]()
- [Optimizer]()

Now we introduce how to select the factors above and related tips.

### Model type

You can select the model type by creating an instance of the corresponding LightingModule.

Furthermore, you can create original model. See [here]()

#### Classification

|LightningModule Class|Model type|Base Model class|Pretrained Weights|Available size by `model_weight` argument|
|---|---|---|---|---|
|VGGModule|[VGG](https://arxiv.org/abs/1409.1556)|`torchvision.models.VGG`|○|`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`|
|ResNetModule|[ResNet](https://arxiv.org/abs/1512.03385)|`torchvision.models.ResNet`|○|`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d`, `wide_resnet50_2`, `wide_resnet101_2`|

#### Object Detection

#### Semantic Segmentation

#### Instance Segmentation

|LightningModule Class|Model type|Base Model class|Pretrained Weights|Available size by `model_weight` argument|
|---|---|---|---|---|
|MaskRCNNModule|[Mask R-CNN](https://arxiv.org/abs/1703.06870)|`torchvision.models.MaskRCNN`|○||
|Mask2FormerModule (`task_type="instance"`)|[Mask2Former](https://arxiv.org/abs/2112.01527)|`transformers.models.Mask2FormerForUniversalSegmentation`|○||
|SwinTransformerModule|[SwinTransformer](https://arxiv.org/abs/2103.14030)|`timm.models.SwinTransformer`|○||
|EvaModule|[EVA](https://arxiv.org/abs/2211.07636)|`timm.models.Eva`|○||

#### Panoptic Segmentation

|LightningModule Class|Model type|Base Model class|Pretrained Weights|Available size by `model_weight` argument|
|---|---|---|---|---|
|Mask2FormerModule (`task_type="panoptic"`)|[Mask2Former](https://arxiv.org/abs/2112.01527)|`transformers.models.Mask2FormerForUniversalSegmentation`|○||

#### Anomaly Detection


### Optimizer

You can select the optimizer by specifying `opt_name` argument in LightningModule.

|`opt_name` argument|Optimizer in `torch.optim`|Hyperparameters|
|---|---|---|
|"sgd"|SGD|`lr`, `momentum`, `weight_decay`|
|"sgd_nesterov"|SGD with `nestrov=True`|`lr`, `momentum`, `weight_decay`|
|"rmsprop"|RMSprop|`lr`, `momentum`, `weight_decay`, `rmsprop_alpha`(`alpha` in RMSprop), `eps`|
|"adam"|Adam|`lr`, `weight_decay`, `adam_betas`(`betas` in Adam), `eps`|
|"adamw"|AdamW|`lr`, `weight_decay`, `adam_betas`(`betas` in AdamW), `eps`|

You can change the hyperparameters of the optimizers by the arguments  in LightningModule, such as `lr`, `momentum`, `weight_decay`, `rmsprop_alpha`, `adam_betas`, `eps`

Example of Adam optimizer in VGG model.

```python
model = VGGModule(class_to_idx,
    opt_name="adam", lr=0.001, weight_decay=0.01, adam_betas=(0.9, 0.999), eps=1e-8
)
```

The optimizer in the model above is compatible with

```python
optimizer = torch.optim.Adam(
    parameters=[p for p in model.model.parameters() if p.requires_grad],
    lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
)
```

## Advanced Usage

- [Pretrained Model]()
- [Learning Rate Scheduler]()
- [Fine Tuning]()

### Pretrained Model

Most of the preset models here has pretrained-weight. You can choose with or without pretrained-weight by `pretrained` argument.



### Fine Tuning

|LightningModule Class|Transferred layers|Default Fine-tuned layers|
|---|---|---|
|"vgg"|classifier (`torchvision.models.VGG.classifier`)|[] (None)|


You change the fine-tuned layers by specifying `tuned_layers` argument.

Example of change the fine-tuned layers in VGG11

```python
model = VGGModule(class_to_idx,
    model_weight='vgg11',
    tuned_layers=[
        'features.16.weight',
        'features.16.bias',
        'features.18.weight',
        'features.18.bias'
    ]
)
```

The transferred and fine-tuned layers in the model above is as follows.

![image]()

### Learning Rate Scheduler

For precise optimization and avoiding overfitting, it is useful to vary learning rate during training process. Learning Rate Scheduler (lr_scheduler) can change the learning rate during the training based on several methods.

|`opt_name` argument|Summary|Class in `torch.optim.lr_scheduler`|Hyperparameters (Name of the argument)|
|---|---|---|---|
|"steplr"||StepLR|`lr_step_size`, `lr_gamma` (`step_size` and `gamma` in StepLR)|
|"multisteplr"||MultiStepLR|`lr_steps`, `lr_gamma` (`milestones` and `gamma` in MultiStepLR)|
|"exponentiallr"||ExponentialLR|`lr_gamma`(`gamma` in ExponentialLR)|
|"cosineannealinglr"||CosineAnnealingLR|`lr_T_max` (`T_max` in CosineAnnealingLR)|
|"reducelronplateau"||ReduceLROnPlateau|`lr_gamma`, `lr_patience` (`factor` and `patience` in ReduceLROnPlateau)|



## Create original model

- Create orignal PyTorch model
- Create orignal LightningModule
