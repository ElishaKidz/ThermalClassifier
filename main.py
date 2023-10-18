from pathlib import Path
from models import ThermalModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules import HitUavDataModule
from lightning.pytorch.loggers import TensorBoardLogger


root_data_dir = Path('data')
allowed_classes = ['Person','Car','OtherVehicle']
model = ThermalModel(num_target_classes=len(allowed_classes)+1)
datamodule = HitUavDataModule(root_data_dir,allowed_classes=allowed_classes)
checkpoint_callback = ModelCheckpoint(dirpath=Path('checkpoints'),monitor='val_loss',verbose=True,mode='min')
callbacks = [checkpoint_callback]
logger = TensorBoardLogger(Path('logs'), name="initial_exp")


trainer = pl.Trainer(accelerator='gpu',callbacks=callbacks,logger=logger,max_epochs=100)
trainer.fit(model,datamodule=datamodule)
