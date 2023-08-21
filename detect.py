# detect.py
import darknet
import cv2

def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    #if type(image_or_path) == "str":
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
        
    # print("Image shape:", image.shape)  
    # print("Image dtype:", image.dtype)  

        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def detect_license_plates(image_name, network, class_names, class_colors, thresh):
    image, detections = image_detection(image_name, network, class_names, class_colors, thresh)
    plate_regions = []

    original_width, original_height = image_name.shape[1], image_name.shape[0]  # Get original image dimensions
    #print("",original_width, original_height)

    for label, confidence, bbox in detections:
        if label == "license_plate":
            left_x, top_y, width, height = map(int, bbox)
            #print("",left_x,top_y,width,height)
            
            width_scale = original_width / image.shape[1]
            height_scale = original_height / image.shape[0]
            
            # Convert bbox ratios to original image coordinates
            left_x = int(left_x * width_scale)
            top_y = int(top_y *  height_scale)
            width = int(width * width_scale)
            height = int(height *  height_scale)
            #print("",left_x,top_y,width,height)
            
            nlx = max(0, left_x - width//2)
            nrx = min(original_width, left_x + width//2)
            nly = max(0, top_y - height//2)
            nhy = min(original_height, top_y + height//2)
            #print("",nlx,nrx,nly,nhy)
            
            plate_region = image_name[nly:nhy, nlx:nrx]
            plate_regions.append(plate_region)

    return plate_regions
