from dataclasses import dataclass
from PIL import Image
from pybboxes import BoundingBox
from typing import Union
import numpy as np
import torch

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
        image =  Image.open(image_path).convert("RGB")

        with open(label_path, 'r') as file:
            label = file.read()
        
        label = Detections.parse_from_text(label, class_mapper)

        return cls(image,label)