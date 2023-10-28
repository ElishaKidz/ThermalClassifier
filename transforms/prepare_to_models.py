from torchvision import transforms
from datasets.classes import ImageSample

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
class PreapareToResnet():
    def __init__(self, resize_shape: tuple = (72, 72)) -> None:
        self.img_transfomrs = transforms.Compose([
            # 72, 90
            transforms.Resize(resize_shape, antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, sample: ImageSample):
        sample.image = self.img_transfomrs(sample.image)
        return sample