import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from datasets.hit_uav_dataset import HitUavDataset
import wandb
from image_multiclass_trainer import ImageMultiClassTrainer
from models.resnet import resnet18 
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transforms import hit_uav_transforms
from torch.utils.data import DataLoader


classes = ['person', 'car']
new_class2index = {name.lower(): i for i, name in enumerate(classes)}
new_class2index['BACKGROUND'] = len(classes)
classes.append('BACKGROUND')
data_root_dir = '/home/xd_eshaar_gcp_idf_il/data'

def sweep():
    model = resnet18(num_target_classes=len(classes))
    area_scale = [config.min_area_scale, config.max_area_scale]
    resnet_resize = (config.resnet_h_size, config.resnet_w_size)

    lightning_model = ImageMultiClassTrainer(class2idx=new_class2index, model=model, optimizer=config.optimizer)
    train_transforms = hit_uav_transforms('train', new_class2index, area_scale, resnet_resize)
    val_transforms = hit_uav_transforms('val', new_class2index, area_scale, resnet_resize)

    train_dataset = HitUavDataset(data_root_dir, new_class2index, 
                                  split="train", transforms=train_transforms)
    val_dataset = HitUavDataset(data_root_dir, new_class2index, 
                                split="val", transforms=val_transforms)
    test_dataset = HitUavDataset(data_root_dir, new_class2index, 
                                split="test", transforms=val_transforms)


    train_loader = DataLoader(train_dataset, 
                          batch_size=128, 
                          num_workers=4)
    
    val_loader = DataLoader(val_dataset, 
                          batch_size=128, 
                          num_workers=4)
  
    test_loader = DataLoader(test_dataset, 
                          batch_size=128, 
                          num_workers=4)

    checkpoint_callback = ModelCheckpoint(dirpath=f"gcs://soi-models/VMD-classifier/classifier-HPT",
                                            filename=f"hit_uav_area_scale={area_scale}_resnet_resize={resnet_resize}_optimizer={config.optimizer}",
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=False)

    trainer = pl.Trainer(default_root_dir=f"gcs://soi-models/VMD-classifier/classifier-HPT",
                            accelerator='gpu',
                            callbacks=[checkpoint_callback],
                            max_epochs=20)
    
    trainer.fit(lightning_model, train_loader, val_loader)
    test_metrics = trainer.test(lightning_model, test_loader, ckpt_path='best', verbose=False)
    test_metrics = test_metrics[0]
    wandb.log({
        "test_loss": test_metrics['test_loss'],
        "test_MulticlassAccuracy": test_metrics['test_MulticlassAccuracy'],
        "test_background_acc": test_metrics['test_BACKGROUND_acc'],
        "test_person_acc": test_metrics['test_person_acc'],
        "test_car_acc": test_metrics['test_car_acc'],
        "min_area_scale": config.min_area_scale,
        "max_area_scale": config.max_area_scale,
        "resnet_h_size": config.resnet_h_size,
        "resnet_w_size": config.resnet_w_size,
        "optimizer": config.optimizer
    })

if __name__ == "__main__":
    hyperparameter_defaults = dict(
        min_area_scale = 1,
        max_area_scale = 2,
        resnet_h_size = 64,
        resnet_w_size = 64,
        optimizer = 'sgd'
    )
    wandb.init(config=hyperparameter_defaults, project="hit-uav-HPT-sgd")
    config = wandb.config
    sweep()
