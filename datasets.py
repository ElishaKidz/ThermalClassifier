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
from pybboxes import BoundingBox



            

class ImageDataset(Dataset):
    def __init__(self,data_dir_path,labels_dir_path=None,data_file_extension='.jpg', label_file_extension='.txt',transforms=None) -> None:
        assert os.path.isdir(data_dir_path) and os.path.isdir(labels_dir_path)
        self.data_dir_path = data_dir_path
        self.labels_dir_path = labels_dir_path
        self.is_test = labels_dir_path is None
        self.transforms = transforms
        self.data_file_extension = data_file_extension
        self.label_file_extension=label_file_extension
        # filter irrelevant files in the data folder and in labels folder if exists
        data_files_paths = [self.data_dir_path/file_path for file_path in os.listdir(self.data_dir_path) if file_path.endswith(data_file_extension)]

        if not self.is_test:
            coupled_data_label_paths = []

            for data_file_path in data_files_paths:
                # find the corresponding label by the data name and labels folder 
                label_file_path = self.labels_dir_path/(data_file_path.stem + self.label_file_extension)

                # Verify that the label exists and if not skip the record
                if label_file_path.exists():
                    coupled_data_label_paths.append((data_file_path,label_file_path))
                else:
                    continue
            
            self.coupled_data_label_paths = coupled_data_label_paths


        else:
            self.coupled_data_label_paths = data_files_paths


    def __len__(self):
        return len(self.coupled_data_label_paths)

    def __getitem__(self, idx):
        assert idx < len(self), OverflowError(f"{idx} is out of dataset range len == {len(self)}")

        if not self.is_test:
            data_path, label_path = self.coupled_data_label_paths[idx]
        
        else:
            data_path = self.coupled_data_label_paths[idx]
            label_path = None

        

        sample = ImageSample.from_paths(data_path,label_path)

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
    def parse_from_text(cls,txt:str,line_sep='\n',fields_sep=',',**kwargs):
        detections = {}

        for detection_txt in txt.split(line_sep):
            fields = detection_txt.split(fields_sep)
            
            if len(fields) == 4: # only bbox without cls 
                bbox = fields
                cl = None
                cl_str = Detections.NO_CLS

            elif len(fields) == 5:
                bbox = fields[1:]
                cl = fields[0]
                cl_str = cl
            else:
                raise ValueError(f'Can not parse the following detection {detection_txt}')

            bbox = list(map(lambda coord: float(coord), bbox))
            cl = int(cl)
            

            # add detection to detections
            if cl not in detections:
                detections[cl] = []
            
            detections[cl].append(Detection.load_generic_mode(bbox=bbox,cl=cl,**kwargs))

        return cls(detections)


@dataclass
class ImageSample():
    image: Union[np.array,torch.Tensor]
    label: Union[int,str,Detections] = None
    metadata = {}
    
    @classmethod
    # Notice that the default parser is and gray scale parser
    def from_paths(cls,image_path,label_path=None,flag=0):
        image =  cv.imread(str(image_path),flag)
        label = None

        if label_path is not None:
            with open(label_path,mode='r') as file:
                label = file.read()
        
        return cls(image,label)



