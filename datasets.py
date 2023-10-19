import imp
from torch.utils.data import Dataset
import os
from typing import Union,List
import numpy as np
import torch
import cv2 as cv
from dataclasses import dataclass
import pytorch_lightning as pl
from google.cloud import storage
from pathlib import Path
from PIL import Image
from pybboxes import BoundingBox

class HitUavDataset(Dataset):

    BUCKET_NAME = 'civilian-benchmark-datasets'
    DATASET_NAME = 'hit-uav'
    IMAGES = 'images'
    LABELS = 'labels'
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'
    CLASS_NAME_TO_CLASS_VALUE_DICT = {'Person':0 ,'Car':1, 'Bicycle':2 ,'OtherVehicle':3 ,'DontCare':4}


    def __init__(self, data_dir_path, class_mapper: dict, labels_dir_path=None, data_file_extension='.jpg', 
                        label_file_extension='.txt', transforms=None) -> None:
        
        assert os.path.isdir(data_dir_path) and os.path.isdir(labels_dir_path)
        self.data_dir_path = data_dir_path
        self.labels_dir_path = labels_dir_path
        self.transforms = transforms
        self.data_file_extension = data_file_extension
        self.label_file_extension = label_file_extension

        self.class_mapper = class_mapper
        self.coupled_data_label_paths = []
        # filter irrelevant files in the data folder and in labels folder if exists
        data_files_paths = [self.data_dir_path/file_path for file_path in os.listdir(self.data_dir_path) if file_path.endswith(data_file_extension)]

        for data_file_path in data_files_paths:
            # find the corresponding label by the data name and labels folder 
            label_file_path = self.labels_dir_path/(data_file_path.stem + self.label_file_extension)

            # Verify that the label exists and if not skip the record
            self.coupled_data_label_paths.append((data_file_path, label_file_path))
                

    def __len__(self):
        return len(self.coupled_data_label_paths)

    def __getitem__(self, idx):
        assert idx < len(self), OverflowError(f"{idx} is out of dataset range len == {len(self)}")

        data_path, label_path = self.coupled_data_label_paths[idx]
            
        sample = ImageSample.from_paths(data_path, label_path, self.class_mapper)

        if self.transforms is not None:
            return self.transforms(sample)
        
        return sample

        
@dataclass
class Detection:
    bbox: BoundingBox
    cls: Union[str,int] = None
    constructors_of_supported_formats = {'yolo':BoundingBox.from_yolo,}

    @classmethod
    def load_generic_mode(cls,bbox,cl=None,mode='yolo',**kwargs):
        bbox = Detection.constructors_of_supported_formats[mode](*bbox,**kwargs)
        return cls(bbox,cl)
    
@dataclass
class Detections:
    detections:dict
    NO_CLS = 'no_cls'

    @classmethod
    # fields_sep=' ',image_size=(640,512)
    def parse_from_text(cls, txt:str, class_mapper: dict, line_sep='\n', fields_sep=' ', image_size=(640,512)):
        detections = {}

        for detection_txt in txt.split(line_sep):
            fields = detection_txt.split(fields_sep)
            
            if len(fields) == 4: # only bbox without cls 
                bbox = fields
                bbox_class = None

            elif len(fields) == 5:
                bbox = fields[1:]
                bbox_class = int(fields[0])
            else:
                raise ValueError(f'Can not parse the following detection {detection_txt}')

            if bbox_class not in class_mapper:
                continue
            
            bbox = list(map(lambda coord: float(coord), bbox))
            bbox_class = class_mapper[int(bbox_class)]
            
            # add detection to detections
            if bbox_class not in detections:
                detections[bbox_class] = []
            
            detections[bbox_class].append(Detection.load_generic_mode(bbox=bbox, cl=bbox_class, image_size=image_size))

        return cls(detections)


@dataclass
class ImageSample():
    image: Union[np.array,torch.Tensor]
    label: Union[int,str,Detections]
    metadata = {}
    
    @classmethod
    # Notice that the default parser is and gray scale parser
    def from_paths(cls, image_path, label_path, class_mapper):
        # image =  cv.imread(str(image_path),flag)
        image =  Image.open(str(image_path)).convert("RGB")

        with open(label_path,mode='r') as file:
            label = file.read()
        
        label = Detections.parse_from_text(label, class_mapper)

        return cls(image,label)



