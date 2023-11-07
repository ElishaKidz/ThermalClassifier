from predictor import ThermalPredictior
from argparse import ArgumentParser
import gcsfs
from pycocotools.coco import COCO
import numpy as np
import cv2 as cv
from PIL import Image

fs = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")

def generate_crop_dimensions(area, area_scale=[1, 2], ratio=[1, 1.5]):
        area =  area * np.random.uniform(*area_scale)
        ratio = np.random.uniform(*ratio)
        w = int(np.sqrt(area) * np.sqrt(ratio))
        h = int(np.sqrt(area) / np.sqrt(ratio))
        return w, h

def sample_background(frame, anns):
    W, H = frame.size
    background_bboxes = []
    for ann in anns:
        w_crop, h_crop = generate_crop_dimensions(ann['area'])
        x0 = np.random.randint(0, W - w_crop + 1)
        y0 = np.random.randint(0, H - h_crop + 1)

        background_bboxes.append([x0, y0, w_crop, h_crop])

    return background_bboxes

def eval_video(video_dir, frames_bboxes_gt, data_cats):
    hits, sum_of_bboxes = 0, 0
    import time
    start_time = time.time()
    for frame_num, anns in frames_bboxes_gt.items():
        # frame num needs to be -1 because we start from 0 need to tell ariel this is bad
        frame_path = f"{video_dir}/frames/frame_{(frame_num - 1):0{6}}.PNG"
        frame = np.asarray(bytearray(fs.open(frame_path, "rb").read()), dtype="uint8")
        frame = cv.imdecode(frame, cv.IMREAD_COLOR)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        bboxes = [ann['bbox'] for ann in anns]
        labels = [data_cats[ann['category_id']]['name'] for ann in anns]
        
        bboxes = bboxes + sample_background(frame, anns)
        labels = labels + ["BACKGROUND" for _ in range(len(anns))]
        
        preds = model.predict_frame_bboxes(frame, np.array(bboxes))

        hits += sum([1 if pred == label else 0 for pred, label in zip(preds, labels)])
        sum_of_bboxes += len(bboxes)
    print(f"predictior took: {time.time() - start_time}")
    return hits, sum_of_bboxes


def main(remote_dir):
    all_hits, all_bboxes = 0, 0
    for video_dir in fs.ls(remote_dir)[1:]:
        if video_dir.split("/")[-1] in ignored_videos:
            continue 
        fs.get(f"{video_dir}/annotations.json", "annotations.json")
        data = COCO("annotations.json")
        frames_bboxes_gt = data.imgToAnns
        cats = data.cats 
        hits, number_of_bboxes = eval_video(video_dir, frames_bboxes_gt, cats)
        print(f"{video_dir} Accuracy: {hits / number_of_bboxes}")

        all_hits += hits
        all_bboxes += number_of_bboxes
    
    print(f"model Accuracy: {all_hits / all_bboxes}")

        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--remote_dir', type=str, default="soi_experiments/annotations-example/test/")
    parser.add_argument('--classifier_ckpt_path',type = str, default='gcs://soi-models/VMD-classifier/debug/checkpoints/epoch=14-step=825.ckpt')
    parser.add_argument('--classifier_model_name',default='resnet18',type=str)
    parser.add_argument('--classifier_device',type=int,default=None)
    parser.add_argument('--ignored_videos', type=str, default="")

    args = parser.parse_args()

    # model = ThermalPredictior(args.classifier_ckpt_path,)
    
    model = ThermalPredictior(args.classifier_model_name, args.classifier_ckpt_path,
                                        load_from_remote=True, device=args.classifier_device)

    ignored_videos = args.ignored_videos.split(",")
    main(args.remote_dir)