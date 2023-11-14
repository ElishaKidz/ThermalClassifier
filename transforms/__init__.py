from ThermalClassifier.transforms.general_transforms import AddShape, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
                                    RandomDownSampleImage, SampleBackground, CropImage, SelectCropCoordinates
                                    
from ThermalClassifier.transforms.prepare_to_models import PreapareToResnet
from torchvision.transforms import Compose


def hit_uav_transforms(split, class2idx, area_scale=[1, 2], resnet_resize=(72, 72)):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.7, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomRotation(degrees=(0, 45)),
                    PreapareToResnet(resnet_resize)
                    ])

def monet_transforms(split, class2idx, area_scale=[0.5, 2], resnet_resize=(72, 72)):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.85, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    PreapareToResnet(resnet_resize)
                    ])

def kitti_transforms(split, class2idx, area_scale=[0.5, 2], resnet_resize=(72, 72)):
    deterministic = False if split == 'train' else True
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    PreapareToResnet(resnet_resize)
                    ])

datasets_transforms ={
    'hit_uav': hit_uav_transforms,
    'monet': monet_transforms,
    'kitti': kitti_transforms
}

def inference_transforms():
    return Compose([
        ToTensor(),
        AddShape(),
        CropImage(),
        PreapareToResnet()
    ])