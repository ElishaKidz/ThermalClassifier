from .general_transforms import AddShape, ToTensor, RandomDownSampleImage, RandomHorizontalFlip, SampleBackground, CropImage, SelectCropCoordinates
from .prepare_to_models import PreapareToResnet
from torchvision.transforms import Compose


def hit_uav_transforms(split, class2idx, area_scale=[1, 2], resnet_resize=(64, 64)):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.7, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    PreapareToResnet(resnet_resize)
                    ])

def monet_transforms(split, class2idx):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.85, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale=[0.5, 2], ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    PreapareToResnet()
                    ])

def inference_transforms():
    return Compose([
        ToTensor(),
        AddShape(),
        CropImage(),
        PreapareToResnet()
    ])