# TODO Move this functions that also appear in the vmd package to a utils package
import cv2 as cv
import pybboxes as pbx
from pybboxes import BoundingBox
def create_video_writer_from_capture(video_capture, output_video_path):
    frame_rate = video_capture.get(cv.CAP_PROP_FPS)
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width)
    new_height = int(height)
    size = (new_width, new_height)

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video_writer = cv.VideoWriter(str(output_video_path), fourcc, frame_rate, size)
    return video_writer

def create_video_capture(input_video_path):
    cap = cv.VideoCapture(input_video_path)
    assert cap.isOpened(), "Could not open video file"
    return cap

def draw_video_from_bool_csv(video, df,bbox_cols_names, output_video_path,class_col_name=None,bbox_foramt='coco',frame_limit=None):
    
    writer = create_video_writer_from_capture(video, output_video_path)
    limit_flag = False
    while True:
        frame_num = video.get(cv.CAP_PROP_POS_FRAMES)

        current_df = df[df["frame_num"] == frame_num]
        current_bboxes = current_df[bbox_cols_names]
        if class_col_name is not None:
            current_classes = current_df[class_col_name]
        
        else:
            current_classes == None



        ret, frame = video.read()
        if frame_limit is not None:
            limit_flag = frame_num>frame_limit
            
        if not ret or limit_flag:
            break

        for i, bbox in enumerate(current_bboxes.values):
            x, y, width, height = BoundingBox.from_coco(*pbx.convert_bbox(bbox,from_type=bbox_foramt,to_type='coco')).raw_values
            frame = cv.rectangle(frame, (x,y), (x+width,y+height), color=(0, 255, 0), thickness=2)
            if current_classes is not None:
                bbox_cls = current_classes.iloc[i]
                cv.putText(frame, bbox_cls, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


        writer.write(frame)

    video.release()
    writer.release()
