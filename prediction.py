from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import utils
import cv2 as cv
import logging
from PIL import Image
from ThermalClassifier.predictor import Predictor

parser = ArgumentParser()
parser.add_argument('--video_path',type=str)
parser.add_argument('--video_bboxes_path', type=str)
parser.add_argument('--ckpt_path',type = str,default='gcs://soi-models/VMD-classifier/refactor/checkpoints/epoch=1-step=110-v3.ckpt')#'gcs://soi-models/VMD-classifier/debug/checkpoints/epoch=14-step=825.ckpt')
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

predictor = Predictor(args.ckpt_path, True, args.device)

bboxes_df = pd.read_csv(args.video_bboxes_path,index_col=0)
translated_predictions = []
while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        logging.debug(f'frame number {frame_num} is processed')
        success, frame = video_cap.read(0)
        if not success or frame_num>= args.frame_limit:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        frame_related_bboxes = bboxes_df[bboxes_df[args.frame_col_name]==frame_num][args.bbox_col_names].values
        if len(frame_related_bboxes) == 0:
            continue

        preds, _ = predictor.predict_frame_bboxes(frame,frame_related_bboxes,args.bbox_format) 
        translated_predictions.extend(preds)
        
bboxes_with_class_predicions = bboxes_df.assign(**{args.class_col_name:translated_predictions})

if args.bbox_save_path is not None:
    bboxes_with_class_predicions.to_csv(args.bbox_save_path)

if args.rendered_video_save_path is not None:
    video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    utils.draw_video_from_bool_csv(video_cap,bboxes_with_class_predicions,args.bbox_col_names,args.rendered_video_save_path,
    args.class_col_name,args.bbox_format,args.frame_limit)


