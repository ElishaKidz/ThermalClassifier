import torch
import gcsfs
from ThermalClassifier.models.resnet import ModelRepo
from SoiUtils.general import get_device
import pybboxes as pbx
import numpy as np
from PIL import Image
from torchvision import transforms 
import torchvision.transforms.functional as F
from SoiUtils.interfaces import Updatable

class ThermalPredictior(Updatable):
    # TODO Add the project name in the gcp as a class constant here, instead of using it below.
    def __init__(self,model_name,ckpt_path,load_from_remote=True,device='cpu'):
        self.device = get_device(device)
        self.load_from_remote = load_from_remote
        # TODO we need to get the transforms of the model from his ckpt somehow 
        self.model_transforms = transforms.Compose([
                                transforms.Resize((72, 72), antialias=False),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        self._load_model_from_ckpt(ckpt_path, model_name)

    def _load_model_from_ckpt(self, ckpt_path, model_name):
        if self.load_from_remote:
            fs = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")
            torch_dict = torch.load(fs.open(ckpt_path, "rb"), map_location='cpu')
        else:
            torch_dict = torch.load(ckpt_path,map_location='cpu')
        
        class2idx = torch_dict['hyper_parameters']["class2idx"]
        self.classes_to_labels_translation = {index: class_name for class_name, index in class2idx.items()}


        state_dict = {key.replace("model.", "") : value for key, value in torch_dict["state_dict"].items()}     
        self.model = ModelRepo.registry[model_name](len(class2idx))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()


    def update(self, ckpt_path, model_name, **kwargs) -> None:
        self._load_model_from_ckpt(ckpt_path, model_name)

    @torch.inference_mode()
    def predict_frame_bboxes(self, frame:Image, frame_related_bboxes: np.array, bboxes_format: str= 'coco',
                             get_features: bool = False):

        frame_crops_according_to_bboxes = []
        frame_size = frame.size
        frame = F.to_tensor(frame)

        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            x0, y0, x1, y1 = pbx.convert_bbox(frame_related_bbox, from_type=bboxes_format, to_type="voc", image_size=frame_size)
            frame_crops_according_to_bboxes.append(self.model_transforms(frame[:, y0: y1, x0: x1]))

        batch = torch.stack(frame_crops_according_to_bboxes, dim=0)
        
        if get_features:
            logits, features = self.model(batch, get_features=True)
            features = features.cpu()
        else:
            logits = self.model(batch)
            features = None

        preds = logits.argmax(axis=1).tolist()
        translated_preds = list(map(lambda x: self.classes_to_labels_translation[x], preds))
        
        return translated_preds, features

