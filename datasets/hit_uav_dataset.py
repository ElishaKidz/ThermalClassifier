import datasets
from transforms import hit_uav_transforms
from torch.utils.data import Dataset
import os
from .classes import ImageSample
from . import datasets_data

class HitUavDataset(Dataset):
    def __init__(self, 
                data_root_dir: str, 
                class2idx: dict, 
                class_mapper: dict, 
                split: str, 
                data_file_extension='.jpg', 
                label_file_extension='.txt') -> None:
        
        self.dataset_name = datasets_data['hit-uav']['DATASET_NAME']
        self.images_dir_name = datasets_data['hit-uav']['IMAGES_DIR_NAME']
        self.label_dir_name = datasets_data['hit-uav']['LABELS_DIR_NAME']


        self.root_dir = f"{data_root_dir}/{self.dataset_name}"
        self.split = split
        self.data_file_extension = data_file_extension
        self.label_file_extension = label_file_extension

        self.class_mapper = class_mapper
        self.class2idx = class2idx
        self.transforms = hit_uav_transforms(self.split, self.class2idx)

        self.coupled_data_label_paths = []
        
        # filter irrelevant files in the data folder and in labels folder if exists
        data_files_paths = [f"{self.root_dir}/{self.images_dir_name}/{self.split}/{file_path}" for file_path in os.listdir(f"{self.root_dir}/{self.images_dir_name}/{self.split}") 
                                if file_path.endswith(data_file_extension)]
    
        for data_file_path in data_files_paths:
            # find the corresponding label by the data name and labels folder 
            label_file_path = f"{self.root_dir}/{self.label_dir_name}/{self.split}/{(data_file_path.split('/')[-1].split('.')[0] + self.label_file_extension)}"

            self.coupled_data_label_paths.append((data_file_path, label_file_path))
                

    def __len__(self):
        return len(self.coupled_data_label_paths)

    def __getitem__(self, idx):
        assert idx < len(self), OverflowError(f"{idx} is out of dataset range len == {len(self)}")

        data_path, label_path = self.coupled_data_label_paths[idx]
            
        sample = ImageSample.from_paths(data_path, label_path, self.class_mapper)

        #if self.transforms is not None:
        sample = self.transforms(sample)
        
        return sample.image, sample.label