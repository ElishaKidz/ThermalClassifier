from argparse import ArgumentParser
from pathlib import Path
from datasets.classes import ImageSample
from models.resnet import ModelRepo 
from image_multiclass_trainer import ImageMultiClassTrainer
import pandas as pd
import utils
import cv2 as cv
from pybboxes import BoundingBox
import pybboxes as pbx
from torch.utils.data import DataLoader
from transforms.prepare_to_models import Model2Transforms
import torch
import logging
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

def load_model(model_class, ckpt):
    torch_dict = torch.load(ckpt)
    state_dict = {key.replace("model.", "") : value for key, value in torch_dict["state_dict"].items()}     
    class2idx = torch_dict['hyper_parameters']["class2idx"]
    # TODO we need to save idx2class
    model = model_class(len(class2idx))
    model.load_state_dict(state_dict)
        
    return model, class2idx


parser = ArgumentParser()
parser.add_argument('--video_path',type=str)
parser.add_argument('--video_bboxes_path', type=str)
parser.add_argument('--ckpt_path',type = str,default=Path('checkpoints/VMD-classifier_debug_checkpoints_epoch=55-step=3528.ckpt'))
parser.add_argument('--model_name',default='resnet18',type=str)
parser.add_argument('--num_target_classes',type=int,default=4)

parser.add_argument('--frame_col_name',type=str,default='frame_num')
parser.add_argument('--bbox_col_names',nargs='+',default=['x','y','width','height'])
parser.add_argument('--class_col_name',type=str,default='cls')
parser.add_argument('--bbox_format',type=str,default='coco')
parser.add_argument('--frame_limit',type=int,default=500)
parser.add_argument('--device',type=int,default=None)

parser.add_argument('--bbox_save_path',type=str,default=Path('outputs/bboxes/result.csv'))
parser.add_argument('--rendered_video_save_path',type=str,default=Path('outputs/videos/result.mp4'))

logging.basicConfig(level=logging.DEBUG)
args = parser.parse_args()
video_cap = utils.create_video_capture(args.video_path)

# The following raw assumes that all models constructors accept only num of classes as input, not sure that this assumption will hold. 

# num_target_classes is the only required parameter to our models.



# themral_model = ModelRepo.registry[args.model_name](args.num_target_classes)
# trainer = ImageMultiClassTrainer.load_from_checkpoint(args.ckpt_path,model = themral_model) # will be only used for its hyperparameters 
# themral_model = trainer.model
# themral_model.eval()
# classes_to_labels_translation = trainer.idx_to_class_mapping

thermal_model, class2idx = load_model(ModelRepo.registry[args.model_name], args.ckpt_path)
classes_to_labels_translation = {index: class_name for class_name, index in class2idx.items()}


bboxes_df = pd.read_csv(args.video_bboxes_path,index_col=0)

transfrom = Model2Transforms[args.model_mame]
device = 'cpu' if args.device is None else torch.device(f'cuda:{args.device}')
predictions = []
while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        logging.debug(f'frame number {frame_num} is processed')
        success, frame = video_cap.read(0)
        if not success or frame_num>= args.frame_limit:
            break

        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = pil_to_tensor(frame)

        frame_related_bboxes = bboxes_df[bboxes_df[args.frame_col_name]==frame_num][args.bbox_col_names].values
        if len(frame_related_bboxes) == 0:
            continue

        frame_crops_according_to_bboxes = []
        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            frame_related_bbox = BoundingBox.from_coco(*pbx.convert_bbox(frame_related_bbox,from_type=args.bbox_format,to_type='coco'))
            x0,y0,x1,y1 = frame_related_bbox.to_voc().raw_values
            croped_frame = frame[:, y0: y1, x0: x1].float().div(255.0)
            sample = ImageSample(image=croped_frame, label=None)
            frame_crops_according_to_bboxes.append(sample)
        
        crops_ready_to_model = list(map(lambda sample: transfrom(sample).image, frame_crops_according_to_bboxes))
        batch = torch.stack(crops_ready_to_model,axis=0).to(device)
        preds = torch.argmax(thermal_model(batch), dim=1)
        predictions.extend(list(preds.cpu().detach().tolist()))
        
predictions_to_class_str = list(map(lambda x: classes_to_labels_translation[x],predictions))
bboxes_with_class_predicion = bboxes_df.assign(**{args.class_col_name:predictions_to_class_str})
if args.bbox_save_path is not None:
    bboxes_with_class_predicion.to_csv(args.bbox_save_path)

if args.rendered_video_save_path is not None:
    video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    utils.draw_video_from_bool_csv(video_cap,bboxes_with_class_predicion,args.bbox_col_names,args.rendered_video_save_path,
    args.class_col_name,args.bbox_format,args.frame_limit)


