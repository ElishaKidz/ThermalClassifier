from datasets import Detections,ImageSample,Detection
import numpy as np
from typing import Tuple
from pybboxes import BoundingBox
import torch
from torchvision import transforms
from typing import Union

class ParseTextLabelsToDetections():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    
    def __call__(self,sample:ImageSample):
        
        assert type(sample.label) == str, ValueError(f'The type of the label is not str but {type(sample.label)}')
        sample.label = Detections.parse_from_text(sample.label,**self.kwargs)
        return sample


class ChoseDetection():
    BACKGROUND = -1
    def __init__(self,allowed_classes:list = None,allow_background=True) -> None:
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else set([])
        self.allow_background = allow_background

    
    def __call__(self,sample:ImageSample):
        assert type(sample.label) == Detections
        # get all objects from the detections dict
        sample_existing_classes = set(sample.label.detections.keys())
        sample_allowed_and_existing_classes = set.intersection(sample_existing_classes,self.allowed_classes)

        # if there are no objects in the image
        if len(sample_allowed_and_existing_classes) == 0:
            sample.label = None
        
        else:
            if self.allow_background:
                sample_allowed_and_existing_classes.add(ChoseDetection.BACKGROUND)

            selected_class = np.random.choice(list(sample_allowed_and_existing_classes))
            # Chose a random detection from the instances of the selected class, if background was chosen return None
            random_detection_of_selected_class = np.random.choice(sample.label.detections.get(selected_class,[None]))
            sample.label = random_detection_of_selected_class
        
        return sample
    
class AddShape():
    def __call__(self,sample:ImageSample):
        metadata = sample.metadata
        metadata['H'], metadata['W'] = sample.image.shape
        return sample
    

        
class SelectCropCoordinates:
    def __init__(self,area_scale:Tuple[float,float] = None,ratio:Tuple[float,float]=None) -> None:
        self.area_scale = area_scale if area_scale is not None else (1.0,1.0)
        self.ratio = ratio if ratio is not None else (1,2)

    
    def __call__(self, sample:ImageSample):
        assert type(sample.label) == Detection or sample.label is None
        w_frame,h_frame = sample.metadata["W"], sample.metadata["H"]
        
        if sample.label is None:
            # Select random crop
            w_crop = np.random.randint(1,w_frame)
            h_crop = np.random.randint(1,h_frame)
            possible_sampling_range_x = (0,w_frame - w_crop)
            possible_sampling_range_y = (0,h_frame - h_crop)



        else:
            # Select an augmented crop round the existing detection
            detection:BoundingBox = sample.label.bbox
            w_crop, h_crop = crop_shape = self.generate_crop_dimensions(detection.area)
            x0_detection,y0_detection,w_detection,h_detection = detection.to_coco().raw_values
            crop_w_larger = w_crop>=w_detection
            crop_h_larger = h_crop>=h_detection

            possible_sampling_range_x = (x0_detection-(w_crop-w_detection),x0_detection) if crop_w_larger else (x0_detection,x0_detection+(w_detection-w_crop))
            possible_sampling_range_y = (y0_detection-(h_crop-h_detection),y0_detection) if crop_h_larger else (y0_detection,y0_detection+(h_detection-h_crop))
            
            possible_sampling_range_x = np.clip(possible_sampling_range_x,0,w_frame - w_crop)
            possible_sampling_range_y = np.clip(possible_sampling_range_y,0,h_frame - h_crop)
        
        if possible_sampling_range_x[0] == possible_sampling_range_x[1]:
            x0 = possible_sampling_range_x[0]
        
        else:
            x0 = np.random.randint(*possible_sampling_range_x)
            
            
        if possible_sampling_range_y[0] == possible_sampling_range_y[1]:
            y0 = possible_sampling_range_y[0]
        
        else:
            y0 = np.random.randint(*possible_sampling_range_y)
            

        crop = BoundingBox.from_coco(x0,y0,w_crop,h_crop)
        sample.metadata['crop_coordinates'] = crop
        return sample



    
    def generate_crop_dimensions(self,area):
        area =  area*np.random.uniform(*self.area_scale)
        ratio = np.random.uniform(*self.ratio)
        w = int(np.sqrt(area) * np.sqrt(ratio))
        h = int(np.sqrt(area) / np.sqrt(ratio))

        return w,h


class CropImage():
    def __call__(self,sample:ImageSample):
        x0,y0,x1,y1 = sample.metadata['crop_coordinates'].to_voc().raw_values
        sample.image = sample.image[x0:x1,y0:y1]
        return sample

class DetectionToClassificaton():
    def __init__(self,cls_translation_dict) -> None:

        self.cls_translation_dict = cls_translation_dict
        self.bacground_class_number = len(cls_translation_dict)
    
    def __call__(self,sample:ImageSample):
    
        if sample.label is None:
            sample.label = self.bacground_class_number
        
        else:
            assert sample.label.cls in self.cls_translation_dict, ValueError(f'class {self.sample.label.cls} not in translation dict')
            sample.label = self.cls_translation_dict[sample.label.cls]
        
        return sample



class PreapareToModel():
    def __init__(self) -> None:
        self.img_transfomrs = transforms.Compose([transforms.ToPILImage(mode='L'),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self,sample:Union[ImageSample,np.array]):
        img = sample.image if type(sample) == ImageSample else sample
        label = sample.label if  type(sample) == ImageSample else None
        transformed_img =  self.img_transfomrs(img)
        if label is not None:
            return transformed_img, label
        
        return transformed_img




        

    
        


        




    

