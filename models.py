import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

class ThermalModel(pl.LightningModule):
    def __init__(self,num_target_classes,learning_rate = 1e-3,train_backbone = True):
        super().__init__()
        self.num_target_classes = num_target_classes
        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.num_target_classes)
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_backbone = train_backbone

        metrics = MetricCollection([
            MulticlassAccuracy(self.num_target_classes), MulticlassPrecision(self.num_target_classes), 
            MulticlassRecall(self.num_target_classes)
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x):
        if self.train_backbone:
            representations = self.feature_extractor(x).flatten(1)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)

        logits = self.classifier(representations)
        return logits
    
    # def _step(self,batch,metrices_dict:MetricCollection):
    #     imgs_batch , labels_batch = batch
    #     logits_batch = self(imgs_batch)
    #     loss = self.loss(logits_batch,labels_batch)
    #     metrices = metrices_dict(logits_batch,labels_batch)
    #     return loss, metrices



    def training_step(self,batch,batch_idx):
        imgs_batch , labels_batch = batch
        logits_batch = self(imgs_batch)
        loss = self.loss(logits_batch,labels_batch)
        metrices = self.train_metrics(logits_batch,labels_batch)
        self.log_dict(metrices,on_step=False,on_epoch=True)
        self.log('train_loss',loss,on_step=False,on_epoch=True,logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        imgs_batch , labels_batch = batch
        logits_batch = self(imgs_batch)
        loss = self.loss(logits_batch,labels_batch)
        self.valid_metrics.update(logits_batch,labels_batch)
        self.log('val_loss',loss,on_step=False,on_epoch=True)
    
    def on_validation_epoch_end(self):
        metrices = self.valid_metrics.compute()
        self.log_dict(metrices,prog_bar=True,logger=True)
        # remember to reset metrics at the end of the epoch
        self.valid_metrics.reset()
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)