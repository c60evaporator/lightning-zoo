
## Basic Usage

- [Start Training]()
- [Logger]()

### Start Training

### Logger

#### CSV Logger

#### MLflow

## Advanced Usage

- [Bottleneck analysis (Profiling)]()
- [Early Stopping]()
- [Multi-GPU Training]()

### Bottleneck analysis (Profiling)

https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html

#### Early Stopping

You can enable early stopping using [EarlyStopping Callback](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)

|Task Type|Available Metrics|
|---|---|
|Classification|"val_loss","val_accuracy","val_f1_macro","val_precision_macro","val_recall_macro"|
|Object Detection|"val_mAP"|

Example

```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3)
trainer = Trainer(callbacks=[early_stop_callback])
trainer.fit(model)
```



#### 


