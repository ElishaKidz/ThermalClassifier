import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ThermalClassifier.data_module import GenericDataModule
from lightning.pytorch.loggers import WandbLogger
from ThermalClassifier.image_multiclass_trainer import ImageMultiClassTrainer
from ThermalClassifier.models import models_dict
from args import args
from ThermalClassifier.datasets import datasets_data


classes = ['person', 'vehicle']
new_class2index = {name.lower(): i for i, name in enumerate(classes)}

if args.add_background_label:
    new_class2index['BACKGROUND'] = len(classes)
    classes.append('BACKGROUND')
###

train_datasets = args.train_datasets_names.strip().split(",")
val_datasets = args.val_datasets_names.strip().split(",")
test_datasets = args.test_datasets_names.strip().split(",")

chosen_datasets = set(train_datasets + val_datasets + test_datasets)
assert all(dataset_name in datasets_data for dataset_name in chosen_datasets), "one of the datasets is not supported"


model = models_dict[args.model](num_target_classes=len(classes))

data_module = GenericDataModule(root_dir=args.root_data_dir, 
                                train_datasets_names=train_datasets,
                                val_datasets_names=val_datasets,
                                test_datasets_names=test_datasets,
                                class2idx=new_class2index,
                                model_transforms=model.transforms)

lightning_model = ImageMultiClassTrainer(class2idx=new_class2index, model=model)

checkpoint_callback = ModelCheckpoint(dirpath=f"gcs://soi-models/VMD-classifier/{args.exp_name}/checkpoints",
                                    monitor='val_MulticlassAccuracy',
                                    mode='max',
                                    verbose=True)

callbacks = [checkpoint_callback]
wandb_logger = WandbLogger(project="VMD-classifier")


trainer = pl.Trainer(default_root_dir=f"gcs://soi-models/VMD-classifier/{args.exp_name}",
                    accelerator='gpu',
                    devices='1',
                    callbacks=callbacks,
                    logger=wandb_logger,
                    max_epochs=2)

trainer.fit(lightning_model, datamodule=data_module)

trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')
