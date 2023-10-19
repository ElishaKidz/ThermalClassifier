import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

class ImageMultiClassTrainer(pl.LightningModule):
    def __init__(self, num_target_classes, model, learning_rate = 1e-3):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        metrics = MetricCollection([
            MulticlassAccuracy(self.num_target_classes), 
            MulticlassPrecision(self.num_target_classes), 
            MulticlassRecall(self.num_target_classes)
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def training_step(self, batch, batch_idx):
        imgs_batch , labels_batch = batch
        logits_batch = self.model(imgs_batch)
        
        loss = self.loss(logits_batch, labels_batch)
        
        self.train_metrics.update(logits_batch, labels_batch)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrices = self.train_metrics.compute()
        self.log_dict(metrices, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # remember to reset metrics at the end of the epoch
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        imgs_batch , labels_batch = batch
        logits_batch = self.model(imgs_batch)
        
        loss = self.loss(logits_batch,labels_batch)
        
        self.valid_metrics.update(logits_batch, labels_batch)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self):
        metrices = self.valid_metrics.compute()
        self.log_dict(metrices, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # remember to reset metrics at the end of the epoch
        self.valid_metrics.reset()
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)