import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ThermalClassifier.data_module import GenericDataModule
from lightning.pytorch.loggers import WandbLogger
from ThermalClassifier.image_multiclass_trainer import BboxMultiClassClassifier
from SoiUtils.load import load_yaml
import argparse
import json
import os.path as osp

def update_cfg(data, updates):
    for key, value in updates.items():
        data[key] = value

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='YAML path')
parser.add_argument('--root_data_dir', type=str, required=True, help='root data dir')
parser.add_argument('--updates', type=str, help='JSON-like string of key-value pairs for updates \
                    (e.g., {"epochs": 10, "learning_rate": 0.001, "labels": ["label1", "label2", "label3"]})')
args = parser.parse_args()

cfg = load_yaml(args.config_path)
cfg['root_data_dir'] = args.root_data_dir

if args.updates:
    try:
        updates = json.loads(args.updates)
        update_cfg(cfg, updates)
        print(f'Updated parameters in {args.config_path}')
    except json.JSONDecodeError:
        print('Invalid JSON-like syntax. Please provide updates in the correct format.')


new_class2index = {name.lower(): i for i, name in enumerate(cfg['classes'])}

if cfg['add_background_label']:
    new_class2index['BACKGROUND'] = len(cfg['classes'])
    cfg['classes'].append('BACKGROUND')
###

model = BboxMultiClassClassifier(class2idx=new_class2index, model_name=cfg['model'])

data_module = GenericDataModule(root_dir=cfg['root_data_dir'],
                                train_datasets_names=cfg['train_datasets'],
                                val_datasets_names=cfg['val_datasets'],
                                test_datasets_names=cfg['test_datasets'],
                                class2idx=new_class2index,
                                model_transforms=model.get_model_transforms())

checkpoint_callback = ModelCheckpoint(dirpath=f"gcs://soi-models/VMD-classifier/{cfg['exp_name']}/checkpoints",
                                      monitor='val_MulticlassAccuracy',
                                      mode='max',
                                      verbose=True)

callbacks = [checkpoint_callback]
wandb_logger = WandbLogger(project="VMD-classifier")


trainer = pl.Trainer(default_root_dir=osp.join(cfg['gcp_dir_name'], cfg['exp_name']),
                    accelerator='gpu',
                    devices=cfg['devices'],
                    callbacks=callbacks,
                    logger=wandb_logger,
                    max_epochs=cfg['epochs'])

trainer.fit(model, datamodule=data_module)

if 'test_datasets' not in cfg:
    trainer.test(model, datamodule=data_module, ckpt_path='best')
