from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from ThermalClassifier.transforms.prepare_to_models import Model2Transforms
from ThermalClassifier.models.resnet import resnet18


class BboxMultiClassClassifier(pl.LightningModule):
    def __init__(self, class2idx, model_name, model_kwargs={}, learning_rate = 1e-3, optimizer: str = "adam"):
        super().__init__()
        self.num_target_classes = len(class2idx)
        self.class2idx = class2idx
        self.idx2class = {v: k for k, v in class2idx.items()}
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = resnet18(num_target_classes=self.num_target_classes, **model_kwargs)
        self.model_transforms = self.model.transforms
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

        metrics = MetricCollection([
            MulticlassAccuracy(self.num_target_classes, average=None), 
            MulticlassPrecision(self.num_target_classes,average=None), 
            MulticlassRecall(self.num_target_classes,average=None)
        ])

        self.metrices = {
            'train': metrics.clone(prefix='train_'),
            'val': metrics.clone(prefix='val_'),
            'test': metrics.clone(prefix='test_')

        }
        self.save_hyperparameters(ignore=['metrices', 'model'])

    def shared_step(self, batch, batch_idx, split):
        imgs, labels = batch
        logits, _ = self.model(imgs)
        loss = self.loss(logits, labels)
        
        self.metrices[split](logits.detach().cpu(), labels.detach().cpu())
        
        return loss

    def log_metrices(self, split):
        metrices = self.metrices[split].compute()
        remaining_metrices = metrices.copy()
        for metric_name, metric_value in metrices.items():
            try:
                if metric_value.shape[0] == self.num_target_classes:
                    metric_dict = {f"{class_name}_{metric_name}": metric_value[i] 
                     for i, class_name in enumerate(self.class2idx.keys())}
                    metric_dict[f"{metric_name}"] = metric_value.mean()
                    self.log_dict(metric_dict, logger=True, on_step=False, on_epoch=True)
                    remaining_metrices.pop(metric_name)

                else:
                    continue
            except:
                continue             
        if len(remaining_metrices) != 0:
            self.log_dict(remaining_metrices, logger=True, on_step=False, on_epoch=True)
        # remember to reset metrics at the end of the epoch
        self.metrices[split].reset()


    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_metrices('train')

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'val')
        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, logger=True)
    
    def on_validation_epoch_end(self):
        self.log_metrices('val')
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'test')
        self.log('test_loss', loss.detach(), on_step=False, on_epoch=True, logger=True)

    def on_test_epoch_end(self):
        self.log_metrices('test')

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                              lr=self.learning_rate, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                               lr=self.learning_rate)
        return optimizer
    
    def predict_step(self, batch, transformed=True, get_features=False) -> Any:
        if not transformed:
            batch = self.model_transforms(batch)

        logits, features = self.model(batch, get_features)
        return logits, features

    def get_model_transforms(self):
         return self.model.transforms

    def on_save_checkpoint(self, checkpoint):
        # Save transform configuration to the checkpoint
        checkpoint['model_transforms'] = self.serialize_transform(self.model_transforms)

    def on_load_checkpoint(self, checkpoint):
        # Load transform configuration from the checkpoint
        self.model_transforms = self.deserialize_transform(checkpoint['model_transforms'])

    def serialize_transform(self, transform):
        # Convert transform to a serializable format (e.g., a list of dictionaries)
        if transform is None:
            return None
        
        return {'name': type(self.model_transforms).__name__,
                'args': self.model_transforms.get_config()}

    def deserialize_transform(self, transform_config):
        # Re-create the transform from the serialized configuration
        if transform_config is None:
            return None
        
        return Model2Transforms.registry[type(self.model).__name__](**transform_config['args'])