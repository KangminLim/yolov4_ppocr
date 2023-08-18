# app.py
from curses import tigetflag
from urllib.request import parse_keqv_list
from test_detect import *
from test_ocr import *
import darknet
import cv2
import argparse
import os
import random
import glob
import time
import numpy as np

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="./backup/yolov4-ANPR.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4-ANPR.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./data/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
    
def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def extract_frames_from_video(video_path, output_folder,target_resolution, frame_interval=60):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if target_resolution is not None:
                frame = cv2.resize(frame, target_resolution)
                
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()

def main():
    # Load YOLO network, class_names, class_colors
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    temp_image_folder = "temp_frames"
    os.makedirs(temp_image_folder, exist_ok=True)
    
    # Extract frames from the video
    target_resolution = (1920,1080)
    extract_frames_from_video(args.input, temp_image_folder, frame_interval=60,target_resolution=target_resolution)

    for frame_name in sorted(os.listdir(temp_image_folder)):
        frame_path = os.path.join(temp_image_folder, frame_name)
        prev_time = time.time()
        frame = cv2.imread(frame_path)
        plate_regions = detect_license_plates(
            frame, network, class_names, class_colors, args.thresh
        )
        for idx, plate_region in enumerate(plate_regions):
      
            # Recognize text on license plates
            result = recognize_license_plate(plate_region)
            detected_text =""
            for result_boxes in result:
                detected_text =""
                for box, (text,confidence) in result_boxes:
                    
                    detected_text += text + " "
                    print("Detected license plate:", detected_text)
                
            # Save the plate region image
            plate_image_path = os.path.splitext(frame_path)[0] + f"_lp_{idx}.jpg"
            cv2.imwrite(plate_image_path, plate_region)
                
            # If 'result' is an image, save it too
            if isinstance(result, np.ndarray):
                result_image_path = os.path.splitext(frame_path)[0] + f"_result_{idx}.jpg"
                cv2.imwrite(result_image_path, result)
            
            
if __name__ == "__main__":
    main()
