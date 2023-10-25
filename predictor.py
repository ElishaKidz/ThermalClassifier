import torch
import gcsfs
from .models.resnet import ModelRepo
from SoiUtils.general import get_device
from .transforms import inference_transforms
import pybboxes as pbx
from pybboxes import BoundingBox
from .datasets.classes import ImageSample
import numpy as np
from PIL import Image
class ThermalPredictior:
    FS = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")

    def __init__(self,model_name,ckpt_path,load_from_remote=True,device='cpu'):
        self.device = get_device(device)
    
        if load_from_remote:
            torch_dict = torch.load(ThermalPredictior.FS.open(ckpt_path, "rb"), map_location='cpu')
        else:
            torch_dict = torch.load(ckpt_path,map_location='cpu')
        
        class2idx = torch_dict['hyper_parameters']["class2idx"]
        self.classes_to_labels_translation = {index: class_name for class_name, index in class2idx.items()}


        state_dict = {key.replace("model.", "") : value for key, value in torch_dict["state_dict"].items()}     
        self.model = ModelRepo.registry[model_name](len(class2idx))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        
        self.transforms = inference_transforms()


    def predict_frame_bboxes(self,frame:Image,frame_related_bboxes:np.array,bboxes_format:str='coco'):

        frame_crops_according_to_bboxes = []
        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            frame_related_bbox = BoundingBox.from_coco(*pbx.convert_bbox(frame_related_bbox,from_type=bboxes_format,to_type='coco'))
            x0, y0, x1, y1 = frame_related_bbox.to_voc().raw_values
            #croped_frame = frame[:, y0: y1, x0: x1].float().div(255.0)
            sample = ImageSample(image=frame, label=None)
            sample.metadata = {'crop_coordinates': [x0, y0, x1, y1]}
            frame_crops_according_to_bboxes.append(sample)
        
        crops_ready_to_model = list(map(lambda sample: self.transforms(sample).image, frame_crops_according_to_bboxes))
        batch = torch.stack(crops_ready_to_model,axis=0).to(self.device)
        preds = torch.argmax(self.model(batch), dim=1).cpu().detach().tolist()
        translated_preds = list(map(lambda x: self.classes_to_labels_translation[x],preds))
        
        
        return translated_preds


