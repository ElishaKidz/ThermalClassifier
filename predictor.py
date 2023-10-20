from argparse import ArgumentParser
from pathlib import Path
from models import ThermalModel
import pandas as pd
import utils
import cv2 as cv
from pybboxes import BoundingBox
import pybboxes as pbx
from torch.utils.data import DataLoader
from transforms import PreapareToModel
import torch
import logging
parser = ArgumentParser()
parser.add_argument('--video_path',type=str)
parser.add_argument('--video_bboxes_path', type=str)
parser.add_argument('--ckpt_path',type = str,default=Path('checkpoints/VMD-classifier_debug_checkpoints_epoch=62-step=3969.ckpt'))
parser.add_argument('--frame_col_name',type=str,default='frame_num')
parser.add_argument('--bbox_col_names',nargs='+',default=['x','y','width','height'])
parser.add_argument('--class_col_name',type=str,default='cls')

parser.add_argument('--bbox_format',type=str,default='coco')
parser.add_argument('--frame_limit',type=int,default=500)
parser.add_argument('--device',type=int,default=None)

parser.add_argument('--bbox_save_path',type=str,default=Path('outputs/bboxes/result.csv'))
parser.add_argument('--rendered_video_save_path',type=str,default=Path('outputs/videos/result.mp4'))

logging.basicConfig(level=logging.DEBUG)

classes_to_labels_translation = {0:'Person',1:'Car',2:'OtherVehicle',3:'Background'}
args = parser.parse_args()
video_cap = utils.create_video_capture(args.video_path)
themral_model = ThermalModel.load_from_checkpoint(args.ckpt_path,num_target_classes=len(classes_to_labels_translation))
themral_model.eval()
bboxes_df = pd.read_csv(args.video_bboxes_path,index_col=0)

transfrom = PreapareToModel()
device = 'cpu' if args.device is None else torch.device(f'cuda:{args.device}')
predictions = []
while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        logging.debug(f'frame number {frame_num} is processed')
        success, frame = video_cap.read(0)
        if not success or frame_num>= args.frame_limit:
            break

        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        frame_related_bboxes = bboxes_df[bboxes_df[args.frame_col_name]==frame_num][args.bbox_col_names].values
        if len(frame_related_bboxes) == 0:
            continue

        frame_crops_according_to_bboxes = []
        for i in range(frame_related_bboxes.shape[0]):
            frame_related_bbox = frame_related_bboxes[i,:]
            frame_related_bbox = BoundingBox.from_coco(*pbx.convert_bbox(frame_related_bbox,from_type=args.bbox_format,to_type='coco'))
            x0,y0,x1,y1 = frame_related_bbox.to_voc().raw_values
            croped_frame = frame[y0:y1,x0:x1]
            frame_crops_according_to_bboxes.append(croped_frame)
        
        crops_ready_to_model = list(map(lambda crop: transfrom(crop),frame_crops_according_to_bboxes))
        batch = torch.stack(crops_ready_to_model,axis=0).to(device)
        preds = torch.argmax(themral_model(batch),dim=1)
        predictions.extend(list(preds.cpu().detach().tolist()))
        
predictions_to_class_str = list(map(lambda x: classes_to_labels_translation[x],predictions))
bboxes_with_class_predicion = bboxes_df.assign(**{args.class_col_name:predictions_to_class_str})
if args.bbox_save_path is not None:
    bboxes_with_class_predicion.to_csv(args.bbox_save_path)

if args.rendered_video_save_path is not None:
    video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    utils.draw_video_from_bool_csv(video_cap,bboxes_with_class_predicion,args.bbox_col_names,args.rendered_video_save_path,
    args.class_col_name,args.bbox_format,args.frame_limit)


