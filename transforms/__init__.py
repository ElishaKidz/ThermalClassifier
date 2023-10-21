from .transforms import AddShape, ToTensor, ChoseDetection, CropImage, DetectionToClassificaton, PreapareToResnet, SelectCropCoordinates
from torchvision.transforms import Compose


def hit_uav_transforms(split, class2idx):
    deterministic = False if split == 'train' else True
    return Compose([AddShape(),
                    ToTensor(),
                    ChoseDetection(class2idx, deterministic),
                    SelectCropCoordinates(area_scale=[0.5,2], ratio=[1,1.5], deterministic=deterministic),
                    CropImage(),
                    DetectionToClassificaton(),
                    PreapareToResnet()
                    ])