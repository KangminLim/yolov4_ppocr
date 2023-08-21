# ocr.py
from pp_ocr import *

def recognize_license_plate(resized_plate_region):
    try:
        plate_text = pp_ocr(resized_plate_region)
        ocr_result = plate_text
        
    except Exception as e:
        print("Error:", e)
        ocr_result = None

    return ocr_result

