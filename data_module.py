import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from ThermalClassifier.datasets.download_dataset import download_dataset
from ThermalClassifier.transforms import datasets_transforms
from ThermalClassifier.datasets.bbox_classification_dataset import BboxClassificationDataset
from torch.utils.data import ConcatDataset
import torchvision
from torchvision.transforms import Compose

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, 
                train_datasets_names: list,
                val_datasets_names: list,
                test_datasets_names: list,
                class2idx: dict,
                root_dir: str,
                model_transforms: torchvision.transforms,
                train_batch_size: int = 256, 
                val_batch_size: int = 256,
                test_batch_size: int = 256,
                train_num_workers: int = 8,
                val_num_workers: int = 8,
                test_num_workers: int = 8) -> None:

        super().__init__()

        self.train_datasets_names = train_datasets_names
        self.val_datasets_names = val_datasets_names
        self.test_datasets_names = test_datasets_names
        self.all_datasets_names = set(train_datasets_names + val_datasets_names + test_datasets_names)
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
            download_dataset(self.root_dir, dataset_name)
    
    def setup(self, stage: str) -> None:
        
        if stage == 'fit':
            self.train_dataset = self.get_dataset(self.train_datasets_names,'train')
            
            self.val_dataset = self.get_dataset(self.val_datasets_names, 'val') 

        if stage == 'test':
            self.test_dataset = self.get_dataset(self.test_datasets_names, 'test')

    def get_dataset(self, datasets_names, split):
        datasets_list = []
        for dataset_name in datasets_names:
            transforms = Compose([datasets_transforms[dataset_name](split, self.class2idx), 
                                  self.model_transforms])
            
            dataset = BboxClassificationDataset(data_root_dir=self.root_dir,
                                                dataset_name=dataset_name,
                                                split=split,
                                                class2idx=self.class2idx,
                                                transforms=transforms)
            datasets_list.append(dataset)
        return ConcatDataset(datasets_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          num_workers=self.train_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.val_batch_size, 
                          num_workers=self.val_num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.test_num_workers)