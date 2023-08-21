from paddleocr import PaddleOCR
import darknet

ocr = PaddleOCR(lang='korean', show_log=False)
img = './01.jpg'

def pp_ocr(img):
    result = ocr.ocr(img,det=True, rec=True, cls=False)
    return result 

def main():
    result = ocr.ocr(False)
    for text_boxes in result:
        for box, text in text_boxes:
            print("Detected Text:", text[0])

if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()