import torch
import gcsfs
from ThermalClassifier.image_multiclass_trainer import BboxMultiClassClassifier
from SoiUtils.general import get_device
import pybboxes as pbx
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from SoiUtils.interfaces import Updatable, Classifier


class Predictor(Updatable,Classifier):

    def __init__(self, ckpt_path, load_from_remote=True, device='cpu'):
        self.device = get_device(device)
        self.load_from_remote = load_from_remote
        self.model = self._load_model_from_ckpt(ckpt_path).to(self.device)

    def _load_model_from_ckpt(self, ckpt_path):
        if self.load_from_remote:
            fs = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")
            return BboxMultiClassClassifier.load_from_checkpoint(fs.open(ckpt_path, "rb"), map_location='cpu')
        return BboxMultiClassClassifier.load_from_checkpoint(ckpt_path, map_location='cpu')

    def update(self, ckpt_path, **kwargs):
        self.model = self._load_model_from_ckpt(ckpt_path).to(self.device)

    @torch.inference_mode()
    def predict_frame_bboxes(self, frame:Image, frame_related_bboxes: np.array, bboxes_format: str= 'coco',
                             get_features: bool = True):

        frame_crops_according_to_bboxes = []
        frame_size = frame.size
        frame = F.to_tensor(frame)

        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            x0, y0, x1, y1 = pbx.convert_bbox(frame_related_bbox, from_type=bboxes_format, to_type="voc", image_size=frame_size)
            frame_crops_according_to_bboxes.append(self.model.model_transforms(frame[:, y0: y1, x0: x1]))

        batch = torch.stack(frame_crops_according_to_bboxes, dim=0).to(self.device)
        
        
        logits, features = self.model.predict_step(batch, get_features=get_features)

        preds = logits.argmax(axis=1).tolist()
        translated_preds = list(map(lambda x: self.model.idx2class[x], preds))
        
        return translated_preds, features


    def classify(self,*args,**kwargs):
        return self.predict_frame_bboxes(*args,**kwargs)
    
