# yolov4_ppocr
yolov4_darknet + pp_ocr
About Darknet https://github.com/AlexeyAB/darknet
About ppocr https://github.com/PaddlePaddle/PaddleOCR
## Algorithm
Input Video -> Detect License plate -> Recognize License plate  

## Download pretrained weight
.weight https://github.com/Dodant/anpr-with-yolo-v4

## Requirements
yolov4_darknet, ppocr, opencv

## usage(test)
```
python app.py --input video.mp4 --dont_show
```
