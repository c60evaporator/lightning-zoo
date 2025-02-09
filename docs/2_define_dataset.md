



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

- Classification
- Object Detection
- Semantic Segmentation
- Instance Segmentation
