
import pytorch_lightning as pl
from google.cloud import storage
from pathlib import Path
from datasets import ImageDataset
from torchvision.transforms import Compose
from transforms import AddShape,ChoseDetection,CropImage,DetectionToClassificaton,ParseTextLabelsToDetections,SelectCropCoordinates,PreapareToModel
from torch.utils.data import DataLoader
import torch
class HitUavDataModule(pl.LightningDataModule):
    
    BUCKET_NAME = 'civilian-benchmark-datasets'
    DATASET_NAME = 'hit-uav'
    IMAGES = 'images'
    LABELS = 'labels'
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'
    CLASS_NAME_TO_CLASS_VALUE_DICT = {'Person':0 ,'Car':1, 'Bicycle':2 ,'OtherVehicle':3 ,'DontCare':4}




    def __init__(self,root_dir,allowed_classes=None,train_batch_size=32,validation_batch_size=32,test_batch_size=32) -> None:
        super().__init__()


        self.root_dir = Path(root_dir)
        self.train_batch_size = train_batch_size
        self.validatoin_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        allowed_classes = allowed_classes if allowed_classes is not None else HitUavDataModule.CLASS_NAME_TO_CLASS_VALUE_DICT.keys()
        allowed_classes_keys = [HitUavDataModule.CLASS_NAME_TO_CLASS_VALUE_DICT[allowed_class] for allowed_class in allowed_classes]
        self.class_translation_dict = {allowed_classes_key:allowed_classes_keys.index(allowed_classes_key) for allowed_classes_key in allowed_classes_keys}

        self.fit_transforms = Compose([AddShape(),ParseTextLabelsToDetections(fields_sep=' ',image_size=(640,512)),
        ChoseDetection(allowed_classes_keys),
        SelectCropCoordinates(area_scale=[0.5,2],ratio=[1,1.5]),CropImage(),
        DetectionToClassificaton(self.class_translation_dict),PreapareToModel()])

    def prepare_data(self):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(HitUavDataModule.BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=HitUavDataModule.DATASET_NAME)  # Get list of files
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            file_split = blob.name.split("/")
            file_name = file_split[-1]
            relative_dir = Path("/".join(file_split[0:-1]))
            final_file_local_path = self.root_dir/relative_dir/file_name
            if final_file_local_path.exists():
                continue
            (self.root_dir/relative_dir).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(final_file_local_path)
        
    
    def setup(self, stage: str) -> None:
        
        if stage == 'fit':
            train_data_path = Path(self.root_dir)/Path(HitUavDataModule.DATASET_NAME)/Path(HitUavDataModule.IMAGES)/Path(HitUavDataModule.TRAIN)
            train_labels_path = Path(self.root_dir)/Path(HitUavDataModule.DATASET_NAME)/Path(HitUavDataModule.LABELS)/Path(HitUavDataModule.TRAIN)

            validation_data_path = Path(self.root_dir)/Path(HitUavDataModule.DATASET_NAME)/Path(HitUavDataModule.IMAGES)/Path(HitUavDataModule.VALIDATION)
            validation_labels_path = Path(self.root_dir)/Path(HitUavDataModule.DATASET_NAME)/Path(HitUavDataModule.LABELS)/Path(HitUavDataModule.VALIDATION)

            self.train_dataset = ImageDataset(data_dir_path=train_data_path,labels_dir_path=train_labels_path,transforms=self.fit_transforms)
            self.validation_dataset = ImageDataset(data_dir_path=validation_data_path,labels_dir_path=validation_labels_path,transforms=self.fit_transforms) 
        
        if stage == 'test':
            test_data_path = Path(self.root_dir)/Path(HitUavDataModule.DATASET_NAME)/Path(HitUavDataModule.IMAGES)/Path(HitUavDataModule.TEST)
            self.test_dataset = ImageDataset(data_dir_path=test_data_path)


    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validatoin_batch_size)