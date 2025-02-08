# Lightning Zoo

## Usage

### LightningModule

You can select the model type by creating an instance of the corresponding LightingModule.

|LightningModule Class|Model type|Model class|Available size by `model_weight` argument|
|---|---|---|
|"vgg"|[VGG]()|`torchvision.models.VGG`|`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`|




#### Optimizer

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

#### Fine Tuning

|LightningModule Class|Transferred layers|Default Fine-tuned layers|
|---|---|---|
|"vgg"|classifier (`torchvision.models.VGG.classifier`)|[] (None)|


You change the fine-tuned layers by specifying `tuned_layers` arguments.

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
