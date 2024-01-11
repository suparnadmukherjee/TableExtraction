from sklearn.cluster import AgglomerativeClustering
from pytesseract import Output
import pandas as pd

import json
from shapely.geometry import Polygon
from pdf2image import convert_from_path
from extraction.text_extraction import paddelExtract
from IDP_compressed.Unbordered.Xerox_Table.table_utils.table_recognition import recognize_table, output_to_xlsx
from IDP_compressed.Unbordered.Xerox_Table.table_utils.table_recognition import output_to_json
# import config
import logging
# logging.config.fileConfig("log_config.conf")

import cv2
import numpy as np
import matplotlib.pyplot as plt

csv_out_path="/home/suparna/PycharmProjects/TableExtraction/data/csv/"

def get_vh_lines_mask(img):
    logging.info("############################")
    logging.info(img.shape)
    img_height, img_width, channel = img.shape
    blockGryImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(blockGryImg, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_bin_inv = 255 - img_bin

    kernel_len_ver = max(20, img_height // 100)
    kernel_len_hor = max(50, img_width // 100)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))  # shape (kernel_len, 1) inverted! xD

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  # shape (1,kernel_ken) xD

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=5)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
    v_kernel = np.ones((3, 1), np.uint8)
    h_kernel = np.ones((1, 3), np.uint8)
    horizontal_lines = cv2.dilate(horizontal_lines, v_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, h_kernel, iterations=1)
    vh_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    return vh_lines, vertical_lines, horizontal_lines


def get_vh_lines(block, vh_lines_mask):
    color_vh_lines_mask = np.dstack((vh_lines_mask, vh_lines_mask, vh_lines_mask))
    vh_lines = block * (color_vh_lines_mask/255) + ~color_vh_lines_mask
    rmvh_lines = block * (~color_vh_lines_mask/255) + color_vh_lines_mask
    vh_lines = np.uint8(vh_lines)
    rmvh_lines = np.uint8(rmvh_lines)
    return vh_lines, rmvh_lines


# def get_rmvh_lines(block, vh_lines_mask):

def output_to_json(arr):
    idx = ["row_" + str(i) for i in range(arr.shape[0])]
    colms = ["col_" + str(i) for i in range(arr.shape[1])]
    dataframe = pd.DataFrame(arr, index=idx, columns=colms)
    result = dataframe.to_json(orient="index")
    return result
import logging

def get_extracted_data(table,img_path):#
    options = "--psm 6"

    print("len(table)",len(table))
    vh_lines_mask, vertical_lines_mask, horizontal_lines_mask = get_vh_lines_mask(table)
    vh_lines, rmvh_lines = get_vh_lines(table, vh_lines_mask)
    image_cv = cv2.imread(img_path)
    output = paddelExtract(img_path)
    arr = process_table_data(rmvh_lines, output)

    '''adding for paddle exp'''
    arrt = arr.T
    table_json = []
    result = output_to_json(arrt)
    print("result", result)
    print()
    parsed = json.loads(result)
    table_json.append(parsed)
    with open(f"{csv_out_path}/idp2/idp2_{img_path.split('/')[-1][:-4]}.json", 'w') as f:
        json.dump(parsed, f)
    result = output_to_xlsx(arrt.T)
    result.to_excel(f"{csv_out_path}/idp2/idp2_{img_path.split('/')[-1][:-4]}.xlsx")
    '''adding for paddle exp'''

    # logging.info(arr)
    return arr.T


def get_centroid(coords):
    x, y, w, h = coords[0], coords[1], coords[2], coords[3]
    x_center = int((x + (x + w)) / 2)
    y_center = int((y + y + h) / 2)
    return x_center, y_center


def get_col_position(col_idx_pos, data):
    idxs = []
    txt = []
    print(data)
    print(col_idx_pos)
    for x in data:
        idxs.append(np.argmin(abs(col_idx_pos - x[1][0])))
        txt.append(x[0])
    return idxs, txt


def rect_distance(rect1, rect2):
    [x1, y1, x1b, y1b] = rect1
    [x2, y2, x2b, y2b] = rect2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return np.linalg.norm(np.array([x1, y1b]) - np.array([x2b, y2])), 0
    elif left and bottom:
        return np.linalg.norm(np.array([x1, y1]) - np.array([x2b, y2b])), 0
    elif bottom and right:
        return np.linalg.norm(np.array([x1b, y1]) - np.array([x2, y2b])), 0
    elif right and top:
        return np.linalg.norm(np.array([x1b, y1b]) - np.array([x2, y2])), 0
    elif left:
        return x1 - x2b, 0
    elif right:
        return x2 - x1b, 0
    elif bottom:
        return y1 - y2b, 0
    elif top:
        return y2 - y1b, 0
    else:  # rectangles intersect
        polygon = Polygon([(x1, y1), (x1b, y1), (x1b, y1b), (x1, y1b)])
        other_polygon = Polygon([(x2, y2), (x2b, y2), (x2b, y2b), (x2, y2b)])
        intersection = polygon.intersection(other_polygon)
        min_rec = min(abs(x1-x1b)*abs(y1-y1b), abs(x2-x2b)*abs(y2-y2b))
        # print(intersection.area/min_rec)
        return 0., intersection.area/min_rec


def isclose(rect1, rect2, MERGE_THRESH):
    [x1, y1, w1, h1] = rect1[2]
    [x2, y2, w2, h2] = rect2[2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    d, intersection = rect_distance(box1, box2)
    if d < MERGE_THRESH:
        return True
    else:
        return False


def marge_bbox(rect1, rect2):
    text1 = rect1[0]
    text2 = rect2[0]
    colID1= rect1[3]
    colID2= rect2[3]
    [x1, y1, w1, h1] = rect1[2]
    [x1, y1, x1b, y1b] = [x1, y1, x1 + w1, y1 + h1]
    [x2, y2, w2, h2] = rect2[2]
    [x2, y2, x2b, y2b] = [x2, y2, x2 + w2, y2 + h2]
    new_rec = [min(x1,x2), min(y1,y2), max(x1b,x2b)-min(x1,x2), max(y1b,y2b)-min(y1,y2)]
    new_text = text1 + ' ' + text2
    new_center = get_centroid(new_rec)
    new_colID = min(colID1, colID2)
    return [new_text, new_center, new_rec, new_colID]


def marge_columns(row_data, MERGE_THRESH=30):
    for k in range(len(row_data)):
        row = row_data[k]
        need_marge = True
        while need_marge:
            i = 0
            need_marge = False
            new_row = []
            while i < len(row):
                if i < len(row) - 1 and (isclose(row[i], row[i+1], MERGE_THRESH) or (row[i][3] != -1 and row[i+1][3] != -1 and row[i][3] == row[i+1][3])):
                    new_row.append(marge_bbox(row[i], row[i+1]))
                    i += 1
                    need_marge = True
                else:
                    new_row.append(row[i])
                i += 1
            row = new_row.copy()
        row_data[k] = row
    return row_data


def interval_bining(row_data, img_shape):
    line = np.zeros(img_shape[1], np.uint8)
    for row in row_data:
        for i in range(len(row)):
            (x, y, w, h) = row[i][2]
            line[x:x+w] = 1
    NOCC, LabelOfCC, stats, centroids = cv2.connectedComponentsWithStats(line, 4, cv2.CV_32S)
    return NOCC-1, LabelOfCC


def assign_columnID(row_data, column_positions):
    try:
        for k in range(len(row_data)):
            row = row_data[k]
            new_row = []
            for i in range(len(row)):
                new_row.append([row[i][0], row[i][1], row[i][2], int(column_positions[row[i][1][0]])-1])
            row_data[k] = new_row.copy()
    except:
        print(row)
        print()
    return row_data


def process_table_data(table, output):

    min_conf = .75
    '''
    commenting for paddle exp 307-326
    '''
    coords=[]
    ocrText=[]

    for i in range(0, len(output)):
        x = int(output[i][0][0][0])
        y = int(output[i][0][0][1])
        w = int(output[i][0][2][0] - x)
        h = int(output[i][0][2][1] - y)
        text = output[i][1][0]
        coords.append((x, y, w, h))
        ocrText.append(text)
    # extract all x-coordinates from the text bounding boxes, setting the
    # y-coordinate value to zero
    '''
       commenting for paddle exp 307-326
    '''


    #     xCoords = [(c[0], 0) for c in coords]
    yCoords = [(0, c[1] + c[3]) for c in coords]
    # print("length all words ", len(xCoords))
    # apply hierarchical agglomerative clustering to the coordinates
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="manhattan",
        linkage="complete",
        distance_threshold=20)
    clustering.fit(yCoords)
    # initialize our list of sorted clusters
    sortedClusters = []
    # logging.info(clustering.labels_)
    # loop over all clusters
    for l in np.unique(clustering.labels_):
        # extract the indexes for the coordinates belonging to the
        # current cluster
        idxs = np.where(clustering.labels_ == l)[0]
        # verify that the cluster is sufficiently large
        if len(idxs) > 0:
            # compute the average x-coordinate value of the cluster and
            # update our clusters list with the current label and the
            # average x-coordinate
            avg = np.average([coords[i][1] for i in idxs])
            sortedClusters.append((l, avg))
    # sort the clusters by their average x-coordinate and initialize our
    # data frame
    sortedClusters.sort(key=lambda x: x[1])
    # print("length sorted clusters", len(sortedClusters))
    # loop over the clusters again, this time in sorted order
    row_data = []
    table_copy = table.copy()
    for (l, _) in sortedClusters:
        # extract the indexes for the coordinates belonging to the
        # current cluster
        idxs = np.where(clustering.labels_ == l)[0]
        # extract the y-coordinates from the elements in the current
        # cluster, then sort them from top-to-bottom
        yCoords = [coords[i][0] for i in idxs]
        sortedIdxs = idxs[np.argsort(yCoords)]
        # generate a random color for the cluster
        color = np.random.randint(0, 255, size=(3,), dtype="int")
        color = [int(c) for c in color]

        # loop over the sorted indexes
        for i in sortedIdxs:
            # extract the text bounding box coordinates and draw the
            # bounding box surrounding the current element
            (x, y, w, h) = coords[i]
            cv2.rectangle(table_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # cv2.imwrite('table1.png', table_copy)
        cols = [[ocrText[i].strip(), get_centroid(coords[i]), coords[i], -1] for i in sortedIdxs]
        row_data.append(cols)

    row_data = marge_columns(row_data, 10)

    num_col, column_positions = interval_bining(row_data, np.shape(table))
    row_data = assign_columnID(row_data, column_positions)
    row_data = marge_columns(row_data, 10)

    # max_cols = max([len(x) for x in row_data])
    max_rows = len(row_data)
    table_data = np.empty((max_rows, num_col), dtype="object")
    logging.info(len(row_data))
    logging.info(max_rows)
    logging.info(num_col)
    for i in range(len(row_data)):
        row = row_data[i]
        for j in range(len(row)):
            text = row[j][0]
            colID = row[j][3]
            table_data[i][colID] = text

    return table_data
