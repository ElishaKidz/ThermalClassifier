import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ThermalClassifier.data_module import GenericDataModule
from lightning.pytorch.loggers import WandbLogger
from ThermalClassifier.image_multiclass_trainer import BboxMultiClassClassifier
import os
from SoiUtils.load import load_yaml

args = load_yaml('configs/thermal.yaml')
args['root_data_dir'] = os.environ['root_data_dir']
new_class2index = {name.lower(): i for i, name in enumerate(args['classes'])}

if args['add_background_label']:
    new_class2index['BACKGROUND'] = len(args['classes'])
    args['classes'].append('BACKGROUND')
###

model = BboxMultiClassClassifier(class2idx=new_class2index, model_name=args['model'])

data_module = GenericDataModule(root_dir=args['root_data_dir'],
                                train_datasets_names=args['train_datasets'],
                                val_datasets_names=args['val_datasets'],
                                test_datasets_names=args['test_datasets'],
                                class2idx=new_class2index,
                                model_transforms=model.get_model_transforms())

checkpoint_callback = ModelCheckpoint(dirpath=f"gcs://soi-models/VMD-classifier/{args['exp_name']}/checkpoints",
                                      monitor='val_MulticlassAccuracy',
                                      mode='max',
                                      verbose=True)

callbacks = [checkpoint_callback]
wandb_logger = WandbLogger(project="VMD-classifier")


trainer = pl.Trainer(default_root_dir=f"gcs://soi-models/VMD-classifier/{args['exp_name']}",
                    accelerator='gpu',
                    devices=args['devices'],
                    callbacks=callbacks,
                    logger=wandb_logger,
                    max_epochs=args['epochs'])

trainer.fit(model, datamodule=data_module)

if args['test_datasets'] != []:
    trainer.test(model, datamodule=data_module, ckpt_path='best')
