from torch.utils.data import Dataset
import torch
from ThermalClassifier.datasets.classes import BboxSample
from pycocotools.coco import COCO
import os.path as osp

class BboxClassificationDataset(Dataset):
    def __init__(self,
                root_dir: str,
                class2idx: dict, 
                annotation_file_name: str,
                transforms = None) -> None:
        
        self.root_dir = root_dir
        self.transforms = transforms

        try:
            data = COCO(osp.join(root_dir, annotation_file_name))
        except FileNotFoundError:
            raise Exception(f"{root_dir} does not have {annotation_file_name} !")

        self.class_mapper = self.create_class_mapper(data.cats, class2idx)

        # only anns that has wanted classes !
        self.anns_dict = {ann_id: ann_dict for ann_id, ann_dict in data.anns.items() 
                                    if ann_dict['category_id'] in self.class_mapper}
        
        self.anns_ids = list(self.anns_dict.keys())
        
        self.imgToAnns = data.imgToAnns
        self.imgs_dict = data.imgs

    def create_class_mapper(self, categories_dict, class2idx):
        old_class2idx = {cat_dict['name'].lower(): cat_id for cat_id, cat_dict in categories_dict.items()}
        
        return {old_idx: class2idx[class_name] for class_name, old_idx in old_class2idx.items() 
                                                if class_name in class2idx.keys()}


    def __len__(self):
        return len(self.anns_ids)

    def __getitem__(self, idx):
        assert idx < len(self), OverflowError(f"{idx} is out of dataset range len == {len(self)}")

        ann_id = self.anns_ids[idx]
        
        bbox = self.anns_dict[ann_id]['bbox']
        label = self.class_mapper[self.anns_dict[ann_id]['category_id']]
    
        image_id = self.anns_dict[ann_id]['image_id']
        image_path = osp.join(self.root_dir, self.imgs_dict[image_id]['file_name'])
        
        sample = BboxSample.create(image_path, bbox, label)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample.image, torch.tensor(sample.label)