from pathlib import Path
from models import ThermalModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules import HitUavDataModule
from lightning.pytorch.loggers import WandbLogger
import os

root_data_dir = Path(f"{os.getcwd()}/data")
exp_name = "debug"

# mapping the index from old dataset to the classes we want
classes = ['Person', 'Car', 'OtherVehicle', 'BACKGROUND']
classes = {name: i for i, name in enumerate(classes)}

dataset_class_mapper = HitUavDataModule.CLASS_NAME_TO_CLASS_VALUE_DICT
class_mapper = {idx: classes[name] for name, idx in dataset_class_mapper.items() 
                                                if name in classes}

class_mapper['BACKGROUND'] = len(classes) - 1
###


model = ThermalModel(num_target_classes=len(classes))
datamodule = HitUavDataModule(root_data_dir, class_mapper=class_mapper)
checkpoint_callback = ModelCheckpoint(  dirpath=f"gcs://soi-models/VMD-classifier/{exp_name}/checkpoints",
                                        monitor='val_MulticlassAccuracy',
                                        mode='max',
                                        verbose=True)
callbacks = [checkpoint_callback]
wandb_logger = WandbLogger(project="VMD-classifier")

trainer = pl.Trainer(default_root_dir=f"gcs://soi-models/VMD-classifier/{exp_name}",
                    accelerator='gpu',
                    callbacks=callbacks,
                    logger=wandb_logger,
                    max_epochs=100)

trainer.fit(model, datamodule=datamodule)
