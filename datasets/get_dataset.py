from .hit_uav_dataset import HitUavDataset
from .monet_dataset import MONETDataset

datasets_dict = {
    'hit-uav': HitUavDataset,
    'monet': MONETDataset
}