import torch
import gcsfs
from ThermalClassifier.image_multiclass_trainer import BboxMultiClassClassifier
from SoiUtils.general import get_device
import pybboxes as pbx
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


class Predictor:
    FS = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")

    def __init__(self, ckpt_path, load_from_remote=True, device='cpu'):
        self.device = get_device(device)
        if load_from_remote:
            self.model = BboxMultiClassClassifier.load_from_checkpoint(Predictor.FS.open(ckpt_path, "rb"), map_location='cpu')
        else:
            self.model = BboxMultiClassClassifier.load_from_checkpoint(ckpt_path, map_location='cpu')


    @torch.inference_mode()
    def predict_frame_bboxes(self, frame:Image, frame_related_bboxes: np.array, bboxes_format: str= 'coco',
                             get_features: bool = False):

        frame_crops_according_to_bboxes = []
        frame_size = frame.size
        frame = F.to_tensor(frame)

        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            x0, y0, x1, y1 = pbx.convert_bbox(frame_related_bbox, from_type=bboxes_format, to_type="voc", image_size=frame_size)
            frame_crops_according_to_bboxes.append(self.model.model_transforms(frame[:, y0: y1, x0: x1]))

        batch = torch.stack(frame_crops_according_to_bboxes, dim=0)
        
        if get_features:
            logits, features = self.model.predict_step(batch, get_features=True)
            features = features.cpu()
        else:
            logits = self.model.predict_step(batch)
            features = None

        preds = logits.argmax(axis=1).tolist()
        translated_preds = list(map(lambda x: self.model.idx2class[x], preds))
        
        return translated_preds, features

