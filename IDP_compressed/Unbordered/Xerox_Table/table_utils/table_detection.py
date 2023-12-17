import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction(img, predictor):
    print("PLOT PREDICTION INPUT IMAGE SHAPE : ",img.shape)
    outputs = predictor(img)

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 3 px
    thickness = 3

    org_img = np.array(img, copy=True)
    mask = np.zeros((np.shape(img)[0],np.shape(img)[1]), dtype="uint8")
    # print("MASK SHAPE :",mask.shape)
    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        mask[int(y1):int(y2),int(x1):int(x2)] = 1

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    plt.imshow(mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    table_list = []
    table_coords = []
    for i in range(num_labels-1):
        x1 = stats[i+1, cv2.CC_STAT_LEFT]
        y1 = stats[i+1, cv2.CC_STAT_TOP] + stats[i+1, cv2.CC_STAT_HEIGHT]
        x2 = stats[i+1, cv2.CC_STAT_LEFT] + stats[i+1, cv2.CC_STAT_WIDTH]
        y2 = stats[i+1, cv2.CC_STAT_TOP]

        if np.shape(img)[1] - abs(x2-x1) < np.shape(img)[1]//10:
            x1 = 5
            x2 = np.shape(img)[1] - 5

        if np.shape(img)[0] - abs(x2-x1) < np.shape(img)[0]//10:
            y1 = 5
            y2 = np.shape(img)[0] - 5

        start_point = (x1, y1)
        end_point = (x2, y2)
        img = cv2.rectangle(np.array(img, copy=True), start_point, end_point, color, thickness)
        table_list.append(np.array(org_img[int(y2):int(y1), int(x1):int(x2)], copy=True))
        table_coords.append([int(x1), int(y1), int(x2), int(y2)])

    return img, table_list, table_coords


def make_prediction(img, predictor):
    outputs = predictor(img)

    table_list = []
    table_coords = []

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()):
        x1, y1, x2, y2 = box
        table_list.append(np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True))
        table_coords.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return table_list, table_coords
