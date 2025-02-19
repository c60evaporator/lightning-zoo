import torch
from torchvision.transforms import v2
import albumentations as A
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

from abc import ABC, abstractmethod

class TorchVisionModule(pl.LightningModule, ABC):
    def __init__(self, model_name, criterion=None,
                 pretrained=False, tuned_layers=None,
                 opt_name='sgd', lr=None, momentum=None, weight_decay=None, rmsprop_alpha=None, adam_betas=None, eps=None,
                 lr_scheduler=None, lr_step_size=None, lr_steps=None, lr_gamma=None, lr_T_max=None, lr_patience=None,
                 first_epoch_lr_scheduled=False, n_batches=None):
        super().__init__()
        self.model_name = model_name
        # Save the criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = self._default_criterion

        # Pretraining configuration
        self.pretrained = pretrained
        if tuned_layers is not None:
            self.tuned_layers = tuned_layers
        else:
            self.tuned_layers = self._default_tuned_layers

        # Optimizer parameters
        self.opt_name = opt_name
        if opt_name.startswith("sgd"):
            self.lr = 0.001 if lr is None else lr
            self.momentum = 0 if momentum is None else momentum
            self.weight_decay = 0 if weight_decay is None else weight_decay
        elif opt_name == "rmsprop":
            self.lr = 0.01 if lr is None else lr
            self.momentum = 0 if momentum is None else momentum
            self.weight_decay = 0 if weight_decay is None else weight_decay
            self.rmsprop_alpha = 0.99 if rmsprop_alpha is None else rmsprop_alpha
            self.eps = 1e-8 if eps is None else eps
        elif opt_name == "adam" or opt_name == "adamw":
            self.lr = 0.001 if lr is None else lr
            self.adam_betas = (0.9, 0.999) if adam_betas is None else adam_betas
            self.eps = 1e-8 if eps is None else eps
            if opt_name == "adam":
                self.weight_decay = 0 if weight_decay is None else weight_decay
            elif opt_name == "adamw":
                self.weight_decay = 0.01 if weight_decay is None else weight_decay
        else:
            raise RuntimeError(f'Invalid optimizer {opt_name}. Only "sgd", "sgd_nesterov", "rmsprop", "adam", and "adamw" are supported.')

        # Learning scheduler parameters
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is None:
            print("No lr_scheduler is used.")
        elif lr_scheduler == "steplr":
            self.lr_step_size = 8 if lr_step_size is None else lr_step_size
            self.lr_gamma = 0.1 if lr_gamma is None else lr_gamma
        elif lr_scheduler == "multisteplr":
            self.lr_steps = [16, 24] if lr_steps is None else lr_steps
            self.lr_gamma = 0.1 if lr_gamma is None else lr_gamma
        elif lr_scheduler == "exponentiallr":    
            self.lr_gamma = 0.1 if lr_gamma is None else lr_gamma
        elif lr_scheduler == "cosineannealinglr":
            self.lr_T_max = lr_T_max
        elif lr_scheduler == "reducelronplateau":
            self.lr_patience = 10 if lr_patience is None else lr_patience
            self.lr_gamma = 0.1 if lr_gamma is None else lr_gamma
        else:
            raise RuntimeError(f'Invalid lr_scheduler {lr_scheduler}. Only "steplr", "multisteplr", "exponentiallr", "cosineannealinglr", and "reducelronplateau" are supported.')
        
        self.lr_step_size = lr_step_size
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma
        self.lr_T_max = lr_T_max
        self.lr_patience = lr_patience
        # first_epoch_lr_scheduler (https://github.com/pytorch/vision/blob/main/references/detection/engine.py)
        self.first_epoch_lr_scheduled = first_epoch_lr_scheduled
        self.first_epoch_lr_scheduler: torch.optim.lr_scheduler.LinearLR = None
        # Set the number batches for lr_scheduler
        self.n_batches = n_batches  # for first_epoch_lr_scheduler

        # Other
        self.model = None
        self.schedulers = None
    
    ###### Set the model and the fine-tuning settings ######
    @abstractmethod
    def _get_model(self) -> torch.nn.Module:
        """Default model"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def _default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        raise NotImplementedError
    
    @abstractmethod
    def _replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        raise NotImplementedError

    def _set_model_and_params(self) -> torch.nn.Module:
        # Set the model
        self.model = self._get_model()
        # Fine tuning setting
        if self.pretrained:
            for name, param in self.model.named_parameters():
                if name in self.tuned_layers:  # Fine tuned layers
                    param.requires_grad = True
                else:  # No-tuned layers
                    param.requires_grad = False
            # Transfer learning setting
            self._replace_transferred_layers()

    def _setup(self):
        """Additional processes during the setup"""
        raise NotImplementedError
    
    def setup(self, stage: str | None = None):
        """Setup the model"""
        self.start_time = time.time()
        # Set the model and its parameters
        self._set_model_and_params()
        # Number of devices
        print('Number of devices=' + str(self.trainer.num_devices))
        
        self.i_epoch = 0
        # For training logging
        self.train_epoch_losses = []
        self.train_step_losses = []
        # For validation logging
        self.val_epoch_losses = []
        self.val_step_losses = []
        self.val_metrics_all = []
        self.val_batch_targets = []
        self.val_batch_preds = []
        # For test logging
        self.test_epoch_metrics = []
        self.test_batch_targets = []
        self.test_batch_preds = []

        # Additional processes during the setup
        self._setup()
        print(f"Setup completed: elapsed_time={time.time()-self.start_time:.2f} sec")

    ###### Training ######
    @abstractmethod
    def _default_criterion(self):
        """Default criterion"""
        raise NotImplementedError
    
    @abstractmethod
    def _calc_train_loss(self, batch) -> torch.Tensor:
        """Calculate the training loss from the batch"""
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training"""
        loss = self._calc_train_loss(batch)
        # Record the loss
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=len(batch[0]),
                 sync_dist=True if self.trainer.num_devices > 1 else False)
        self.train_step_losses.append(loss.item())
        # first_epoch_lr_scheduler
        if self.first_epoch_lr_scheduler is not None:
            self.first_epoch_lr_scheduler.step()
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Epoch end processes during the training"""
        # Record the epoch loss
        epoch_train_loss = sum(self.train_step_losses) / len(self.train_step_losses)
        self.train_epoch_losses.append(epoch_train_loss)
        self.train_step_losses = []
        # Disable first_epoch_lr_scheduler
        self.first_epoch_lr_scheduler = None
        # LR Scheduler is automatically called by PyTorch Lightning because `interval` is "epoch" (See https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling)

        print(f'Epoch {self.current_epoch}{" device" + str(self.global_rank) if self.trainer.num_devices > 1 else ""}: finished! train_loss={epoch_train_loss}, elapsed_time={time.time()-self.start_time:.2f} sec')
    
    ###### Validation ######
    @abstractmethod
    def _val_predict(self, batch):
        """Predict the validation batch"""
        raise NotImplementedError
    
    @abstractmethod
    def _calc_val_loss(self, preds, targets):
        """Calculate the validation loss from the batch"""
        raise NotImplementedError
    
    @abstractmethod
    def _get_preds_cpu(self, preds):
        """Get the predictions and store them to CPU as a list"""
        raise NotImplementedError

    @abstractmethod
    def _get_targets_cpu(self, targets):
        """Get the targets and store them to CPU as a list"""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Validation step (E.g., calculate the loss and store the predictions and targets)"""
        # Predict
        preds, targets = self._val_predict(batch)
        # Calculate the loss
        loss = self._calc_val_loss(preds, targets)
        # Record the loss
        if loss is not None:
            self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, 
                     logger=True, batch_size=len(batch[0]),
                     sync_dist=True if self.trainer.num_devices > 1 else False)
            self.val_step_losses.append(loss.item())
        # Store the predictions and targets for calculating metrics
        self.val_batch_preds.extend(self._get_preds_cpu(preds))
        self.val_batch_targets.extend(self._get_targets_cpu(targets))

    @abstractmethod
    def _calc_epoch_metrics(self, preds, targets):
        """Calculate the metrics from the targets and predictions"""
        raise NotImplementedError

    def on_validation_epoch_end(self):
        """Epoch end processes during the validation (E.g., calculate the loss and metrics)"""
        # Record the epoch loss
        if len(self.val_step_losses) > 0:
            epoch_val_loss = sum(self.val_step_losses) / len(self.val_step_losses)
            self.val_epoch_losses.append(epoch_val_loss)
            self.val_step_losses = []
            print(f'Epoch {self.current_epoch}{" device" + str(self.global_rank) if self.trainer.num_devices > 1 else ""}: val_loss={epoch_val_loss}')
        # Calculate the metrics
        metrics = self._calc_epoch_metrics(self.val_batch_preds, self.val_batch_targets)
        print(f'Epoch {self.current_epoch}{" device" + str(self.global_rank) if self.trainer.num_devices > 1 else ""}: ' + ' '.join([f'{k}={v}' for k, v in metrics.items()]))
        self.val_metrics_all.append(metrics)
        # Initialize the lists for the next epoch
        self.val_batch_targets = []
        self.val_batch_preds = []
        # Record the metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value,
                     sync_dist=True if self.trainer.num_devices > 1 else False)

    ###### Test ######
    def test_step(self, batch, batch_idx):
        """Validation step (E.g., store the predictions and targets)"""
        # Store the predictions and targets
        self.test_batch_targets.extend(self._get_targets_cpu(batch))
        self.test_batch_preds.extend(self._get_preds_cpu(batch))

    def on_test_epoch_end(self):
        """Epoch end processes during the test (E.g., calculate the metrics)"""
        # Calculate the metrics
        metrics = self._calc_epoch_metrics(self.test_batch_preds, self.test_batch_targets)
        print(f'Epoch {self.current_epoch} device{self.global_rank}: ' + ' '.join([f'{k}={v}' for k, v in metrics.items()]))
        self.test_epoch_metrics.append(metrics)        
        # Initialize the lists for the next epoch
        self.test_batch_targets = []
        self.test_batch_preds = []
        # Record the metrics
        # for metric_name, metric_value in metrics.items():
        #     self.log(f"test_{metric_name}", metric_value)

    ###### Prediction ######
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
    
    ##### Optimizers and Schedulers ######
    def configure_optimizers(self):
        """Configure optimizers and LR schedulers"""
        # Optimizer
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay, nesterov="nesterov" in self.opt_name)
        elif self.opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(parameters, lr=self.lr, momentum=self.momentum, 
                                            weight_decay=self.weight_decay, eps=self.eps, alpha=self.rmsprop_alpha)
        elif self.opt_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps, betas=self.adam_betas)
        elif self.opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps, betas=self.adam_betas)
        
        # lr_schedulers (https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling)
        lr_scheduler_config = None
        if self.lr_scheduler == "steplr":
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma),
                "interval": "epoch",
                "frequency": 1
            }
        elif self.lr_scheduler == "multisteplr":
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_steps, gamma=self.lr_gamma),
                "interval": "epoch",
                "frequency": 1
            }
        elif self.lr_scheduler == "exponentiallr":
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma),
                "interval": "epoch",
                "frequency": 1
            }
        elif self.lr_scheduler == "cosineannealinglr":
            if self.lr_T_max is None:
                raise RuntimeError(f'The `lr_T_max` argument should be specified if "cosineannealinglr" is selected as the lr_scheduler. Tipically, lr_T_max is set to the number of epochs.')
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.lr_T_max),
                "interval": "epoch",
                "frequency": 1
            }
        elif self.lr_scheduler == "reducelronplateau":
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_gamma, patience=self.lr_patience),
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        
        # first_epoch_lr_scheduler (https://github.com/pytorch/vision/blob/main/references/detection/engine.py)
        if self.first_epoch_lr_scheduled:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, self.n_batches - 1) if self.n_batches is not None else 1000
            self.first_epoch_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        print(f"Optimizers and LR schedulers are configured: elapsed_time={time.time()-self.start_time:.2f} sec, optimizer={optimizer}, lr_scheduler={lr_scheduler_config}")
        if self.lr_scheduler is None:
            # Return single optimizer (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)
            return optimizer
        else:
            # Optimizer and LR Scheduler are returned as a dictionary (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config
            }
        
    ##### Display ######
    def plot_train_history(self):
        """Plot the training history TODO: Will be moved to Trainer"""
        # Create a figure and axes
        n_metrics = len(self.val_metrics_all[0])
        fig, axes = plt.subplots(n_metrics+1, 1, figsize=(5, 4*(n_metrics+1)))
        colors = plt.get_cmap('tab10').colors
        # Plot the training and validation losses
        axes[0].plot(range(1, self.current_epoch+1), self.train_epoch_losses, label='train_loss', color=colors[0])
        if len(self.val_epoch_losses) > 0:
            axes[0].plot(range(0, self.current_epoch+1), self.val_epoch_losses, label='val_loss', color=colors[1])
        axes[0].legend()
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Losses in each epoch')
        # Plot the validation metrics
        for i, metric_name in enumerate(self.val_metrics_all[0].keys()):
            axes[i+1].plot(range(0, self.current_epoch+1),
                           [metrics[metric_name] for metrics in self.val_metrics_all],
                           label=f'val_{metric_name}',
                           color=colors[1])
            axes[i+1].set_xlabel('Epoch')
            axes[i+1].set_ylabel(metric_name)
            axes[i+1].set_title(f'Validation {metric_name}')
        fig.tight_layout()
        plt.show()

    @abstractmethod
    def _plot_predictions(self, images, preds, targets, n_images=10):
        """Plot the images with predictions"""
        raise NotImplementedError

    def plot_predictions_from_val_dataset(self, n_images=10):
        """Plot the predictions in the first minibatch of the validation dataset TODO: Will be moved to Trainer"""
        # Get the first minibatch
        inputs, targets = next(iter(self.val_dataloader()))
        torch.set_grad_enabled(False)
        self.model.eval()
        # Predict
        preds = self._get_preds_cpu(inputs)
        # Denormalize the images
        images = inputs
        for tr in self.val_dataloader().dataset.transform:
            if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
                reverse_transform = v2.Compose([
                    v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                        std=[1/std for std in tr.std])
                ])
                images = reverse_transform(inputs)
                continue
        # Display the images
        self._plot_predictions(images, preds, targets, n_images)
        torch.set_grad_enabled(True)
        self.model.train()

    def plot_predictions_from_image(self, image_path):
        """Plot the predictions from an image TODO: Will be moved to Trainer"""
        if isinstance(image_path, str):
            image_pathes = [image_path]
        elif isinstance(image_path, list) or isinstance(image_path, tuple):
            image_pathes = image_path
        self.model.eval()
        torch.set_grad_enabled(False)
        # Loop over the images
        for img_path in image_pathes:
            # Load the image
            image = Image.open(img_path)
            # Preprocess the image using the same transform as the validation dataset
            transform = self.val_dataloader().dataset.transform
            if isinstance(transform, A.Compose):
                image_tensor = transform(image=np.array(image))
            elif isinstance(transform, v2.Compose):
                image_tensor = transform(image)
            else:
                raise RuntimeError('The `transform` argument should be an instance of `albumentations.Compose` or `torchvision.transforms.Compose`.')
            # Predict
            outputs = self.model(image_tensor)
            # Display the image

        torch.set_grad_enabled(True)
        self.model.train()

    @abstractmethod
    def _plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics"""
        raise NotImplementedError

    def plot_metrics_detail(self, metric_name=None):
        """Plot the detail of the metrics TODO: Will be moved to Trainer"""
        self._plot_metrics_detail(metric_name)
