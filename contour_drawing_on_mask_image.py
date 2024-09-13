
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def text_eraser_from_mask_images(source_image):
    
    image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
    # cv2.medianBlur(image_Gray,5)
    height, width = image_Gray.shape[:2]
    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)
    # cv2.imwrite("blank_image.jpg",blank_image)
    _, thresh = cv2.threshold(image_Gray, 5, 255, cv2.THRESH_BINARY & cv2.THRESH_OTSU)
    cv2.imwrite("thresh.jpg",thresh)
    # _, thresh = cv2.threshold(image_Gray, 0, 255, cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    print("Original contours length",len(contours))
    retained_contours = []
    for i,cont in enumerate(contours):
   
        contoured_mask_image = blank_image.copy()
        cnt = np.array([cont], np.int32)
        cnt = cnt.reshape((-1, 1, 2))

        cv2.fillPoly(contoured_mask_image, [cnt], (255,255,255))

        filled_area = cv2.countNonZero(contoured_mask_image)
        # total image area
        total_image_area = image_Gray.shape[0]*image_Gray.shape[1]
        # if filled_area <= 0.1 * total_image_area:
           
        retained_contours.append(cont)

    return retained_contours


def draw_intersected_bounding_box_on_mask_image(mask_image,retained_contours):

    height,width = mask_image.shape[:2]
    contoured_mask_image = np.zeros((height,width),dtype = np.uint8)
    print("retained_contours_length:",len(retained_contours))
        
    for i,cnt1 in enumerate(retained_contours):
        
        cv2.drawContours(contoured_mask_image, cnt1, -1, (255, 255, 255), 1) 
     
    return contoured_mask_image
# img_path = 'Adaptive_thresholding_results_along_with_medianblur_morph_closing_op/data_12_Sep_5/ca_emeryvile_output_mask.jpg'
# mask_image = cv2.imread(img_path)
# retained_contours = text_eraser_from_mask_images(mask_image)
# contoured_mask_image = draw_intersected_bounding_box_on_mask_image(mask_image,retained_contours)
# cv2.imwrite('ca_emeryvile.png',contoured_mask_image)




folder_path = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/canny_edge_detection/Adaptive_thresholding_results_along_with_medianblur_morph_closing_op/data_12_Sep_5/"
output_dir = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/canny_edge_detection/contour_results_after_morph_gradient_13_sep_2024/          "
all_masks = os.listdir(folder_path)

masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]

for renamed_mask in masks_renamed:
    mask_path = f"{folder_path}/{renamed_mask}.jpg"
    mask_image = cv2.imread(mask_path)

    retained_contours = text_eraser_from_mask_images(mask_image)
    contoured_mask_image = draw_intersected_bounding_box_on_mask_image(mask_image,retained_contours)

    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(folder_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
    cv2.imwrite(output_file_path,contoured_mask_image)


