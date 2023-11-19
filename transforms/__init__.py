from ThermalClassifier.transforms.general_transforms import AddShape, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
                                    RandomDownSampleImage, SampleBackground, CropImage, SelectCropCoordinates
from torchvision.transforms import Compose


def hit_uav_transforms(deterministic, class2idx, area_scale=[1, 2]):
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.7, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomRotation(degrees=(0, 45)),
                    ])

def monet_transforms(deterministic, class2idx, area_scale=[0.5, 2]):
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    # RandomDownSampleImage(down_scale_factor_range=[0.85, 1], p=0.3),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1.5], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    ])

def kitti_transforms(deterministic, class2idx, area_scale=[1, 1]):
    return Compose([ToTensor(),
                    SampleBackground(class2idx, deterministic, p=0.2),
                    AddShape(),
                    SelectCropCoordinates(class2idx, area_scale, ratio=[1, 1], deterministic=deterministic),
                    CropImage(),
                    RandomHorizontalFlip(p=0.5),
                    ])

datasets_transforms ={
    'hit-uav': hit_uav_transforms,
    'monet': monet_transforms,
    'kitti': kitti_transforms
}