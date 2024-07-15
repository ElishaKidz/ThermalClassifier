import pandas as pd
import utils.utils as utils
import cv2 as cv
from pathlib import Path
from argparse import ArgumentParser
from pycocotools.coco import COCO

parser = ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--annotation_path', type=str, required=True)

parser.add_argument('--frame_col_name',type=str,default='frame_num')
parser.add_argument('--bbox_col_names',nargs='+',default=['x','y','width','height'])
parser.add_argument('--class_col_name',type=str,default='cls')
parser.add_argument('--bbox_format',type=str,default='coco')
parser.add_argument('--frame_limit',type=int,default=None)
parser.add_argument('--rendered_video_save_path',type=str,default=Path('outputs/videos/result.mp4'))
args = parser.parse_args()

annotation_df = pd.DataFrame(columns=([args.frame_col_name] + args.bbox_col_names + [args.class_col_name])) 
video_cap = utils.create_video_capture(args.video_path) 

data = COCO(args.annotation_path)
for img_id, annotations in data.imgToAnns.items():
    frame_num = img_id - 1
    for ann in annotations:
        if ann['attributes']['occluded']:
            continue
        x, y, w, h = ann['bbox']
        ann_class = data.cats[ann['category_id']]['name']
        new_row = {'x': x, 'y': y, 'width': w, 'height': h, 'frame_num': frame_num, 'cls': ann_class}
        annotation_df.loc[len(annotation_df)] = new_row

video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
utils.draw_video_from_bool_csv(video_cap, annotation_df, args.bbox_col_names,args.rendered_video_save_path,
args.class_col_name, args.bbox_format, args.frame_limit)