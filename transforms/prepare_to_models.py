from typing import Union
import torch
from torchvision import transforms
from ThermalClassifier.datasets.classes import BboxSample

class Model2Transforms:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print(f'Class {name} already exists. Will replace it')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper


@Model2Transforms.register(name='resnet18')
class PreapareToResnet18():
    def __init__(self, resize_shape: tuple = (72, 72)) -> None:
        self.resize_shape = resize_shape
        self.img_transfomrs = transforms.Compose([
            # 72, 90
            transforms.Resize(resize_shape, antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, sample: Union[BboxSample, torch.Tensor]):
        """
        This transforms is used for training and inference.
        In train time our sample is BboxSample because we need the label of the bbox
        In inference time our sample is only crop / image 
        """
        if isinstance(sample, BboxSample):
            sample.image = self.img_transfomrs(sample.image)
        else:
            sample = self.img_transfomrs(sample)
        
        return sample
    
    def get_config(self):
        # Return a dictionary capturing the configuration of the transform
        return {'resize_shape': self.resize_shape}
