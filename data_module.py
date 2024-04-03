import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from ThermalClassifier.datasets import datasets_data
from ThermalClassifier.datasets import datasets_data
from ThermalClassifier.transforms.prepare_to_models import Transform
from ThermalClassifier.transforms import datasets_transforms
from ThermalClassifier.datasets.bbox_classification_dataset import BboxClassificationDataset

from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose
from SoiUtils.cloud.storage import download_folder
class GenericDataModule(pl.LightningDataModule):
    def __init__(self, 
                train_datasets_names: list,
                val_datasets_names: list,
                test_datasets_names: list,
                class2idx: dict,
                root_dir: str,
                model_transforms: Transform,
                additional_datasets_parameters:dict=None,
                train_batch_size: int = 32, 
                val_batch_size: int = 256,
                test_batch_size: int = 256,
                train_num_workers: int = 8,
                val_num_workers: int = 8,
                test_num_workers: int = 8) -> None:

        super().__init__()

        self.train_datasets_names = train_datasets_names
        self.val_datasets_names = val_datasets_names
        self.test_datasets_names = test_datasets_names
        self.all_datasets_names = {dataset_name.split("/")[0] for dataset_name in 
                                    set(train_datasets_names + val_datasets_names + test_datasets_names)}
        
        self.additional_datasets_parameters = additional_datasets_parameters if additional_datasets_parameters is not None else {}
        self.root_dir = Path(root_dir)
        self.model_transforms = model_transforms
        self.class2idx = class2idx

        # dataloader params
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers


    def prepare_data(self):
        for dataset_name in self.all_datasets_names:
            dataset_metadata =  datasets_data[dataset_name]
            download_folder(self.root_dir, 
                            dataset_metadata['BUCKET_NAME'], dataset_metadata['DATASET_DIR'])
    
    def setup(self, stage: str) -> None:
        
        if stage == 'fit':
            self.train_dataset = self.get_dataset(self.train_datasets_names, deterministic=False)
            
            self.val_dataset = self.get_dataset(self.val_datasets_names) 

        if stage == 'test':
            self.test_dataset = self.get_dataset(self.test_datasets_names)

    def get_dataset(self, datasets_names, deterministic=True):
        datasets_list = []
        for dataset_name in datasets_names:
            dataset_name, annotation_file_name = dataset_name.split("/")
            dataset_transform = datasets_transforms.get(dataset_name, datasets_transforms['SOI'])
            transforms = Compose([dataset_transform(deterministic, self.class2idx), 
                                  self.model_transforms])
            
            additional_params = self.additional_datasets_parameters.get(dataset_name,{})
            dataset = BboxClassificationDataset(root_dir=f"{self.root_dir}/{dataset_name}",
                                                annotation_file_name=annotation_file_name,
                                                class2idx=self.class2idx,
                                                transforms=transforms,**additional_params)
            datasets_list.append(dataset)
        return ConcatDataset(datasets_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          num_workers=self.train_num_workers,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.val_batch_size, 
                          num_workers=self.val_num_workers,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.test_num_workers,
                          pin_memory=True)