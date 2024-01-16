from torch.utils.data import Dataset
import torch
from ThermalClassifier.datasets.classes import BboxSample
from pycocotools.coco import COCO
from pathlib import Path
class BboxClassificationDataset(Dataset):
    def __init__(self,
                root_dir: str,
                class2idx: dict, 
                annotation_file_name: str,
                transforms = None,subsampling=3) -> None:

        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.subsampling = subsampling
        
        try:
            data = COCO(self.root_dir/annotation_file_name)
        except FileNotFoundError:
            raise Exception(f"{root_dir} does not have {annotation_file_name} !")

        # This is a small patch to way⌊ the soi labels are currently structured, 
        # there is need to append the img dir to the relative path in order to get the relevant frame
        
        if 'img_dir' in data.dataset['info']:
            self.img_dir = data.dataset['info']['img_dir']
        
        else:
            self.img_dir = None


        self.class_mapper = self.create_class_mapper(data.cats, class2idx)

        # only anns that has wanted classes !
        self.anns_dict = {ann_id: ann_dict for ann_id, ann_dict in data.anns.items() 
                                    if ann_dict['category_id'] in self.class_mapper}
        
        # Subsample the video using the subsampling variable
        self.anns_dict = {ann_id: ann_dict for i,(ann_id, ann_dict) in enumerate(data.anns.items()) 
                                    if i %  subsampling == 0}
        
        self.anns_ids = list(self.anns_dict.keys())
        
        self.imgToAnns = data.imgToAnns
        self.imgs_dict = data.imgs

    def create_class_mapper(self, categories_dict, class2idx):
        old_2_new_idx_mapping = {}

        for _, cat_dict in categories_dict.items():
            old_id, class_name, supercatergory = cat_dict['id'] ,cat_dict['name'].lower(), cat_dict['supercategory'].lower()
            if supercatergory in class2idx:
                old_2_new_idx_mapping[old_id] = class2idx[supercatergory]
            
            elif class_name in class2idx:
                old_2_new_idx_mapping[old_id] = class2idx[class_name]
            
            else:
                continue
        
        return old_2_new_idx_mapping


    def __len__(self):
        return len(self.anns_ids)

    def __getitem__(self, idx):
        assert idx < len(self), OverflowError(f"{idx} is out of dataset range len == {len(self)}")

        ann_id = self.anns_ids[idx]
        
        bbox = self.anns_dict[ann_id]['bbox']
        label = self.class_mapper[self.anns_dict[ann_id]['category_id']]
    
        image_id = self.anns_dict[ann_id]['image_id']
        image_file_name = self.imgs_dict[image_id]['file_name']
        image_path = self.root_dir/image_file_name if self.img_dir is None else self.root_dir/self.img_dir/image_file_name 
        
        sample = BboxSample.create(image_path, bbox, label)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample.image, torch.tensor(sample.label)