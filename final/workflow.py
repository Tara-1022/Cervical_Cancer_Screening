import cv2
import numpy as np
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

SEG_MODEL_PATH = r"C:\Users\taral\OneDrive\Documents\Major project\Code\final\segmentation_model.pt"
CLS_MODEL_PATH = r"C:\Users\taral\OneDrive\Documents\Major project\Code\final\classification_model.pt"

seg_model = YOLO(SEG_MODEL_PATH)
cls_model = YOLO(CLS_MODEL_PATH)

def SR_remove(image):
    orig = image
#     cv2.imwrite("original.jpg", orig)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, image = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)
    mask = image
    srs = cv2.bitwise_and(orig, orig, mask=mask)
    sub = orig - srs
    lower_val = np.array([0,0,0])
    upper_val = np.array([50,50,50])
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, lower_val, upper_val)
    kernel = np.ones((9,9), np.uint8)
    dilated = cv2.dilate(black_mask, kernel, iterations = 1)
    final = cv2.inpaint(orig, dilated, inpaintRadius=30, flags=cv2.INPAINT_TELEA)
    cv2.imshow("SR removed", final)
    # cv2.imwrite("sr_removed2.jpg", final)
    cv2.waitKey(0)
    return final


def apply_segment(image):
    results = seg_model.predict(image)
    # try:
    for result in results:
            # Get the height and width of the original image
            height, width = result.orig_img.shape[:2]

            # Create the background
            background = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Get all predicted masks
            masks = result.masks.xy

            # Get the original image
            orig_img = result.orig_img

            for mask in masks:
                    mask = mask.astype(int)

                    # Create a mask image
                    mask_img = np.zeros_like(orig_img)

                    # Fill the contour of the mask image in white
                    cv2.fillPoly(mask_img, [mask], (255, 255, 255))

                    # Extract the object from the original image using the mask
                    masked_object = cv2.bitwise_and(orig_img, mask_img)

                    # Copy the masked object to the background image
                    background[mask_img == 255] = masked_object[mask_img == 255]
    # except:
    #     print("ERROR:")
    #     return
    cv2.imshow("Segmented", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("segmented2.jpg", background)
    return background

def get_classification(img):
    results = cls_model.predict(img)
    for result in results:
          id = result.probs.top1
          name = result.names[id]
    return name

def predict(img):
     return get_classification(
          apply_segment(
               SR_remove(
                    img
               )
          )
     )

print(predict(
     cv2.imread(
        #   r"C:\Users\taral\OneDrive\Documents\Major project\Data\Compiled_Split_All_20\val\HSIL-CIN3\AAJJ0.jpg"
        # r"C:\Users\taral\OneDrive\Documents\Major project\Data\Compiled_All\HSIL-CIN3\AABP5.jpg"
        # r"C:\Users\taral\OneDrive\Documents\Major project\Data\Compiled_All\Normal\AADQ04.jpg"
        r"C:\Users\taral\OneDrive\Documents\Major project\Data\Compiled_All_ALL\AAHP1.jpg"
     )
))