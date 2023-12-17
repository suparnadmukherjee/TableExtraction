#importing required packages

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from extraction.text_extraction import paddelExtract
from directory_paths import PROJECT_ROOT_DIR
from image_processing.remove_horizontal_lines import remove_h_line2
from image_processing.dialate_vertical_space import add_v_space
from image_processing.dialate_horizontal_space import add_h_space

csv_out_path=f"{PROJECT_ROOT_DIR}data/csv/"
processed_image_path=f"{PROJECT_ROOT_DIR}data/processed_image/"

def intersection(box_1, box_2):
    '''
    Args:
        box_1:  coordinates of horizontal box
        box_2:  coordinates of vertical box

    Returns: list:[ xmin of vertical box,
                    ymin of horizontal box,
                    xmax of vertical box,
                    ymax of horizontal box,
                ]
    '''

    return ([box_2[0], box_1[1], box_2[2], box_1[3]])

def iou(box_1,box_2):
    '''
    tl=top-left coordinate
    tr=top-right coordinate
    bl=bottom-left coordinate
    br=bottom-right coordinate
    Args:
        box_1: Coordinates of Box 1 ([tl,tr,br,bl])
        box_2: Coordinates of box 2 ([tl,tr,br,bl])

    Returns:
        Area of intersecrion of box 1 and box zero if there is any common region, else returns zero.

    '''
    x_1=max(box_1[0],box_2[0])
    y_1=max(box_1[1],box_2[1])
    x_2=min(box_1[2],box_2[2])
    y_2=min(box_1[3],box_2[3])

    #intersection
    inter=abs(max((x_2-x_1,0))*max((y_2-y_1),0))
    if inter==0:
        return 0

    #union
    box_1_area=abs((box_1[2]-box_1[0])*(box_1[3]-box_1[1]))
    box_2_area=abs((box_2[2]-box_2[0])*(box_2[3]-box_2[1]))
    union=float(box_1_area+box_2_area-inter)

    #intersection over union
    iou_=inter/union
    return iou_


def restructure_table(image_path,output):
    fname = Path(image_path).stem   #getting the name of the file without extension
    '''

    Args:
        image_cv: image path str
        output: coordinates and text from paddle ocr

    Returns:
        None.
        creates csv file

    '''

    #image_path = "/content/Apple__36_cropped_margin10_1_h_lineout_v.png"
    image_cv = cv2.imread(image_path)
    image_height = image_cv.shape[0]
    image_width = image_cv.shape[1]
    im = image_cv.copy()

    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    horiz_boxes = []
    vert_boxes = []
    '''
    For every text box detected two larger boxes are being constructed: 
        1. Horizontal   :width equal to the image width and height equal to the text box height
        2. Vertical     :width equal to the box width and height equal to the text image height
    '''
    for box in boxes:
        x_h, x_v = 0, int(box[0][0])    #extending x coordinates of horizontal boxes to 0 and x of vertical box to tl x coordinate
        y_h, y_v = int(box[0][1]), 0    #extending y coordinates of horizontal boxes to y of tf and y of vertical box to 0

        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        #horizontal and vertical boxes appended in x,y,w,h format
        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        cv2.rectangle(im, (x_h, y_h), (x_h + width_h, y_h + height_h), (0, 255, 0), 1)
        cv2.rectangle(im, (x_v, y_v), (x_v + width_v, y_v + height_v), (255, 0, 0), 1)

    #drawing all vertical and horizontal boxes on the image
    cv2.imwrite(f"{processed_image_path}{fname}_hvboxes.png", im)

    #Prunes away boxes that have high intersection-over-union (IOU) overlap with previously selected horizontal boxes.
    # TODO:horiz_out iou_threshold to be added in config
    horiz_out = tf.image.non_max_suppression(horiz_boxes,
                                             probabilities,
                                             max_output_size=1000,
                                             iou_threshold=0.1,
                                             score_threshold=float('-inf'),
                                             name=None
                                             )
    horiz_lines = np.sort(np.array(horiz_out))
    #drawing the horizontal lines unsupressed
    im_nms = image_cv.copy()
    for val in horiz_lines:
        # print(horiz_boxes[val])
        cv2.rectangle(im_nms,
                      (int(horiz_boxes[val][0]), int(horiz_boxes[val][1])),
                      (int(horiz_boxes[val][2]), int(horiz_boxes[val][3])),
                      (0, 0, 255),
                      1
                      )
    cv2.imwrite(f"{processed_image_path}{fname}_hl.png", im_nms)

    # Prunes away boxes that have high intersection-over-union (IOU) overlap with previously selected horizontal boxes.
    #TODO:vert_out iou_threshold to be added in config
    vert_out = tf.image.non_max_suppression(vert_boxes, probabilities,
                                            max_output_size=1000,
                                            iou_threshold=0.05,
                                            score_threshold=float('-inf'),
                                            name=None
                                            )
    vert_lines = np.sort(np.array(vert_out))
    # drawing the vertical lines unsupressed
    for val in vert_lines:
        # print(vert_boxes[val])
        cv2.rectangle(im_nms,
                      (int(vert_boxes[val][0]), int(vert_boxes[val][1])),
                      (int(vert_boxes[val][2]), int(vert_boxes[val][3])),
                      (0, 255, 0),
                      2
                      )
    cv2.imwrite(f"{processed_image_path}{fname}_vl.png", im_nms)

    '''
    ----------------------
    The length of the horiz_out and vert_lines are the no.of rows and columns of the table detected.
    So out_array is an empty array created of dimension len(horiz_out)xlen(vert_lines)
    ----------------------
    '''
    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_out))]
    #print(np.array(out_array).shape)

    unordered_boxes = []
    for i in vert_lines:
        #print(vert_boxes[i])
        unordered_boxes.append(vert_boxes[i][0])
    #print(unordered_boxes)

    ordered_boxes = np.argsort(unordered_boxes)
    #print(ordered_boxes)

    #putting text from respective indices that have row col intersection
    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                #TODO:intersection iou threshold to be added in config
                if (iou(resultant, the_box) > 0.1):
                    out_array[i][j] = texts[b]

            #print(resultant)
    #print(out_array)

    #saving csv file
    out_file = f"{csv_out_path}nms_iou/nms_iou_{image_path.split('/')[-1][:-4]}.csv"
    pd.DataFrame(out_array).to_csv(out_file)
    logging.info(f"{out_file} saved")

def get_table(image_path):

    img_path=image_path
    logging.info(f"{img_path}going for ocr")
    image_cv = cv2.imread(img_path)
    output=paddelExtract(img_path)
    restructure_table(img_path,output)

if __name__=="__main__":
    get_table("/home/suparna/PycharmProjects/TableExtraction/data/cropped_tables/_48_cropped_margin20_1.png")