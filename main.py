import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import GenericDataModule
from lightning.pytorch.loggers import WandbLogger
import os
from datasets import datasets_data
from image_multiclass_trainer import ImageMultiClassTrainer
from models.resnet import resnet18
from args import args

# TODO need to create func that can take class and can map to multi original classes !

# mapping the index from old dataset to the classes we want
classes = ['Person', 'Car', 'OtherVehicle']
new_class2index = {name: i for i, name in enumerate(classes)}

original_class2index = datasets_data[args.dataset_name]["CLASS_TO_IDX"]

# convert the old class id to new class id
class_mapper = {old_idx: new_class2index[class_name] for class_name, old_idx in original_class2index.items() 
                                                if class_name in classes}
if args.add_background_label:
    new_class2index['BACKGROUND'] = len(classes)
    classes.append('BACKGROUND')
###

data_module = GenericDataModule(root_dir=args.root_data_dir, 
                            dataset_name=args.dataset_name,
                            class2idx=new_class2index, 
                            class_mapper=class_mapper)

model = resnet18(num_target_classes=len(classes))
lightning_model = ImageMultiClassTrainer(num_target_classes=len(classes), model=model)

checkpoint_callback = ModelCheckpoint(dirpath=f"gcs://soi-models/VMD-classifier/{args.exp_name}/checkpoints",
                                    monitor='val_MulticlassAccuracy',
                                    mode='max',
                                    verbose=True)

callbacks = [checkpoint_callback]
wandb_logger = WandbLogger(project="VMD-classifier")



trainer = pl.Trainer(default_root_dir=f"gcs://soi-models/VMD-classifier/{args.exp_name}",
                    accelerator='gpu',
                    callbacks=callbacks,
                    logger=wandb_logger,
                    max_epochs=100)

trainer.fit(lightning_model, datamodule=data_module)
