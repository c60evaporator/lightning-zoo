# Define the Dataset (DataModule)

## Basic Usage

At least, the following factors should be specified

- [Transforms]()
- [Dataset Type]()
- [Display images and annotations]()

### Transforms

Transforms are data preprocessing to improve model performance, such as augmentation. Transforms change the shape, colour and other characteristics of the image or the target (annotation).

In this package, both TorchVision (`torchvision.transforms.v2.Compose`) and Albumentations (`albumentations.Compose`) transforms are available.
There are six arguments (`train_transform`, `train_target_transform`, `train_transforms`, `eval_transform`, `eval_target_transform`, `eval_transforms`) to pass the Transforms to the DataModule, but the difference between `train_*` and `eval_*` are just for training or for validation and test. Thus we introduce the difference between `train_transform`, `train_target_transform`, and `train_transforms` here.

The recommended argument for passing Transforms depend on the Task Type as follows.

|Task Type|Argument for image transform|Argument for target (annotation) transform|Recommended Transform Platform|
|---|---|---|---|
|Classification|`train_transform`|-|Both TorchVision and Albumentations|
|Object Detection|`train_transforms` or `train_transform`|`train_transforms` or `train_target_transform`|Both TorchVision and Albumentations|
|Semantic Segmentation|`train_transforms` |`train_transforms`|Albumentations|

#### Classification

In Classification task, `train_transform` is the recommended argument for passing the Transform for the image. Transform for the target is not needed generally because it's just a intager label and there is hardly any room for conversion.

Both TorchVision and Albumentations Transform can be passed to `train_transform` argument.

Example of TorchVision

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.Resize((32,32)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(5),
    v2.RandomAffine(0, shear=10, scale=(0.9,1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
eval_transform = A.Compose([
    v2.Resize((32,32)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

Example of Albumentations (Watch out for the order of `ToTensorV2` and `Normalize`)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Transform for image in training
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2()
])
# Transform for image in validation and test
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2()
])
```

Some transformations in `train_transform` are for data augumentation and they are not needed for the validation, so these transformations are removed from `eval_transform`.

Generally speaking, PyTorch Dataset load data as `PIL.Image` and it should be converted into `torch.Tensor` by the Transforms. Therefore at least `ToTensor` (PyTorch) or `ToTensorV2` (Albumentation) is needed for this conversion.

#### Object detection

### Dataset type

You can select the model type by creating an instance of the corresponding DataModule.

Furthermore, you can create original dataset. See [here]()

#### Classification

|DataModule Class|Dataset Name|Base PyTorch Dataset class|Available `image_set`|
|---|---|---|---|---|
|CIFAR10DataModule|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)|[torchvision.datasets.CIFAR10]()|`train`, `val` (`val` and `test` are the same)|

#### Object Detection

|DataModule Class|Dataset Name|Base PyTorch Dataset class|Available `image_set`|Number of Images|Number of Annotations|
|---|---|---|---|---|
|VOCDetectionDataModule|[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)|[torchvision.datasets.VOCDetection]()|`train`, `val` (`val` and `test` are the same)|
|CocoDetectionDataModule|[MS COCO](https://cocodataset.org/#home)|[torchvision.datasets.CocoDetection]()|`train`, `val` (`val` and `test` are the same)|

#### Semantic Segmentation

VOC
Cityscapes

#### Instance Segmentation

|DataModule Class|Dataset Name|Base PyTorch Dataset class|Available `image_set`|
|---|---|---|---|---|
|CocoSegmentationDataModule|[MS COCO](https://cocodataset.org/#home)|[torch_extend.datasets.CocoSegmentation]()|`train`, `val` (`val` and `test` are the same)|

#### Panoptic Segmentation

Cityscapes

### Display images and annotations

## Advanced Usage

- [Dataset Validation]()

### Dataset Validation

You can validate the dataset by calling ``

#### Classification

- Target label is less than the max value of `self.class_to_idx`

#### Object detection

- Negative (zero) bounding box width
- Negative (zero) bounding box height
- Target label is less than the max value of `self.class_to_idx`

For example

## Create original DataModule

- Create original PyTorch Dataset
- Create original DataModule

### Create original PyTorch Dataset

The PyTorch Dataset of the following tasks should include `class_to_idx` member variable.

#### Classification

#### Object Detection

#### Semantic Segmentation

#### Instance Segmentation
