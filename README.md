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

- **test image : 정지되어 있는 주차장**
![1](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/3650d62c-d10d-47ae-9fbe-93bd8ef06d84)
![2](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/abd6ebfa-aee7-481c-9ea2-680e7e95cc52)


![3](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/71367040-d3d4-49dc-a133-4aa49ef59e81)
![4](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/3e206a4f-c966-4d79-bfb6-cda2e7d54e16)
![5](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/dc0a5d64-74dc-4df8-a30c-e1126e3ab714)

## 결과물

- **출력 예시**


![6](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/9c44e8aa-887a-413b-8c99-90c0b63aba21)



- **프레임 별 이미지 (60프레임 기준)**

  
![7](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/202dbb1e-95dc-49e6-a074-c8653a087610)


- **License plate 사진 (Class: license plate가 탐지 되었을 때)**

  
![8](https://github.com/t3q-intern2023-2/task13_yolov4_ppocr/assets/87487729/a04eeb72-5055-4e98-807f-2816cc8d9ce4)



- **frame별  BBox coordinates값 및 Detected_text값이 저장된 json 파일**


```json
[
    {
        "frame_name": "frame_0000.jpg",
        "detected_text": "6202 ",
        "center_x": 618,
        "center_y": 638,
        "width": 78,
        "height": 18
    },
    {
        "frame_name": "frame_0000.jpg",
        "detected_text": "1남거 ",
        "center_x": 969,
        "center_y": 630,
        "width": 92,
        "height": 28
    },
    {
        "frame_name": "frame_0000.jpg",
        "detected_text": "1남거 9378 ",
        "center_x": 969,
        "center_y": 630,
        "width": 92,
        "height": 28
    }
	]
```
