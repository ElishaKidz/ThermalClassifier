from .general_transforms import AddShape, ToTensor, DownSampleImage, SampleBackground, CropImage, SelectCropCoordinates
from .prepare_to_models import PreapareToResnet
from torchvision.transforms import Compose


def hit_uav_transforms(split, class2idx):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.3),
                    #DownSampleImage(down_scale_factor=0.7),
                    AddShape(),
                    SelectCropCoordinates(area_scale=[0.8, 2], ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    PreapareToResnet()
                    ])

def monet_transforms(split, class2idx):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.3),
                    # DownSampleImage(down_scale_factor=0.7),
                    AddShape(),
                    SelectCropCoordinates(area_scale=[0.8, 2], ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    PreapareToResnet()
                    ])

def inference_transforms():
    return Compose([
        ToTensor(),
        AddShape(),
        CropImage(),
        PreapareToResnet()
    ])