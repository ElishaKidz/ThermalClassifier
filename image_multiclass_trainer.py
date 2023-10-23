import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

class ImageMultiClassTrainer(pl.LightningModule):
    def __init__(self, class2idx, model, learning_rate = 1e-3):
        super().__init__()
        self.num_target_classes = len(class2idx)
        self.class2idx = class2idx
        self.idx2class = {v: k for k, v in class2idx.items()}
        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

        metrics = MetricCollection([
            MulticlassAccuracy(self.num_target_classes, average=None), 
            MulticlassPrecision(self.num_target_classes), 
            MulticlassRecall(self.num_target_classes)
        ])

        self.metrices = {
            'train': metrics.clone(prefix='train_'),
            'val': metrics.clone(prefix='val_'),
            'test': metrics.clone(prefix='test_')

        }
        self.save_hyperparameters(ignore=['metrices', 'model'])

    def shared_step(self, batch, batch_idx, split):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss(logits, labels)
        
        self.metrices[split](logits.cpu(), labels.cpu())
        
        return loss

    def log_metrices(self, split):
        metrices = self.metrices[split].compute()
        
        accuracy_per_class = metrices.pop(f"{split}_MulticlassAccuracy")
        accuracy_dict = {f"{split}_{class_name}_acc": accuracy_per_class[i] 
                     for i, class_name in enumerate(self.class2idx.keys())}
        
        accuracy_dict[f"{split}_MulticlassAccuracy"] = accuracy_per_class.mean()

        self.log_dict(accuracy_dict, logger=True, on_step=False, on_epoch=True)
        self.log_dict(metrices, logger=True, on_step=False, on_epoch=True)

        # remember to reset metrics at the end of the epoch
        self.metrices[split].reset()


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_metrices('train')

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'val')
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
    
    def on_validation_epoch_end(self):
        self.log_metrices('val')
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'test')
        self.log('test_loss', loss.item(), on_step=False, on_epoch=True, logger=True)

    def on_test_epoch_end(self):
        self.log_metrices('test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)