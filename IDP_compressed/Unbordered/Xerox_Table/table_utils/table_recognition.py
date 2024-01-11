import cv2
import numpy as np
import pandas as pd
#from pcr import Pipeline
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from IDP_compressed.Unbordered.Xerox_Table.strhub.data.module import SceneTextDataModule
from IDP_compressed.Unbordered.Xerox_Table.strhub.models.utils import load_from_checkpoint, parse_model_args
import tesserocr
from PIL import Image
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#pipeline = Pipeline()
# Initialize the device
device = torch.device('cpu')
import tesserocr
kwargs = {
            "refine_iters": 3
        }

# Get the transforms and the model
model = load_from_checkpoint("/home/suparna/PycharmProjects/TableExtraction/IDP_compressed/xerox-tatras-backend/model/PCR/epoch=99"
                             "-step=134051-val_accuracy=91.1990-val_NED=96.9191.ckpt", **kwargs).eval().to(device)
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

@torch.inference_mode()
def prediction_engine(model, image):
    """ validation or evaluation """
    # Collect data from the image
    image = image.to(device)
    p = model(image).softmax(-1)
    pred, p = model.tokenizer.decode(p)
    return str(pred[0]).strip()

def sort_cluster(list_coordinates):
    '''
    Function : sort_cluster (Sort the word level bounding Boxes list).
    Input : List of unsorted Bounding Box Coordinates with every element in ['text', [x0, y0, x1, y1]] format.
    Output : List of sorted Bounding Box Coordinates with every element in ['text', [x0, y0, x1, y1]] format.
    '''
    # Placeholder
    points = []
    images_coordinates = [list_coordinates[i] for i in range(len(list_coordinates)) if
                          list_coordinates[i][0] == "image"]
    temp_list = [list_coordinates[i] for i in range(len(list_coordinates)) if list_coordinates[i][0] != "image"]
    keypoints_to_search = []
    max_x = 0
    for i in temp_list:
        x0, y0, x1, y1 = i[-1]
        keypoints_to_search.append([i[0], i[-1]])
        max_x = max(max_x, x1)

    # Loop and sort
    while len(keypoints_to_search) > 0:
        a = (sorted(keypoints_to_search, key=lambda p: p[-1][1]))
        a = (sorted(keypoints_to_search, key=lambda p: p[-1][0]))[0]
        a_list = [[i[0], i[-1][0], i[-1][1], i[-1][2], i[-1][3]] for i in keypoints_to_search]
        a_list = pd.DataFrame(a_list, columns=["label", "x0", "y0", "x1", "y1"])
        a_list = a_list.sort_values(['y0', 'x1'], ascending=[True, True]).reset_index(drop=True)
        a = list(a_list.iloc[0])
        a = [a[0], [a[1], a[2], a[3], a[4]]]
        b = [a[0], a[1], [max_x, a[-1][1]]]
        min_h = min(abs(a[1][1] - a[1][-1]), abs(b[1][1] - b[1][-1]))

        # Convert to 3-D space
        a = np.array([a[-1][0], a[-1][1]])
        b = np.array([b[-1][0], b[-1][1]])

        # Placeholder
        row_points = []
        remaining_points = []
        for k in keypoints_to_search:
            p = np.array([k[-1][0], k[-1][1]])
            d = np.linalg.norm(np.cross(np.subtract(p, a), np.subtract(b, a))) / np.linalg.norm(b - a)
            if d < min_h / 2:
                row_points.append(k)
            else:
                remaining_points.append(k)

        points.extend(sorted(row_points, key=lambda h: h[-1][0]))
        keypoints_to_search = remaining_points

    # Return the points
    points = [[points[i][0], points[i][1]] for i in range(len(points))]
    points.extend(images_coordinates)
    return points

def recognize_table(finalboxes, img):
    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append('')
            else:
                pred_str_global = ""
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                finalboxes[i][j][k][3]
                    pad = 0
                    finalimg = img[x-pad:x+h+pad , y-pad:y+w+pad]
                    if h < 7 or w < 10:
                        # inner = inner + ""
                        continue
                    #cv2.imwrite("/home/manpreet/Downloads/Xerox_Table/output/img"+str(i)+".jpg",finalimg)
                    # prediction_groups = pipeline.recognize(images=[finalimg])
                    # boxes = []
                    # for elem in prediction_groups[0]:
                    #     x0 = elem[0][0]
                    #     y0 = elem[0][1]
                    #     x1 = elem[2][0]
                    #     y1 = elem[2][1]
                    #     boxes.append(["text", [x0, y0, x1, y1]])
                    # sorted_clusters = sort_cluster(list_coordinates=boxes)


                    # for box_curr in sorted_clusters:
                    #     # Crop and collect
                    #     x0, y0, x1, y1 = map(int, box_curr[-1])
                    #     if x0 < x1 and y0 < y1:
                    # try:
                    # bb_coordinates = [x0, y0, x1, y1]
                    # Crop the image
                    cropped_image_data_ = Image.fromarray(finalimg)#.convert("L").convert("RGB")
                    #cv2.imwrite("/home/abc/Downloads/Xerox_Table/cropped/"+str(i)+"_"+str(j)+".jpg",np.array(cropped_image_data))
                    basewidth = 600
                    wpercent = (basewidth / float(cropped_image_data_.size[0]))
                    hsize = int((float(cropped_image_data_.size[1]) * float(wpercent)))
                    cropped = cropped_image_data_.resize((basewidth, hsize), Image.ANTIALIAS)
                    thresh = 200
                    fn = lambda x: 255 if x > thresh else 0
                    cropped_image_data = cropped.convert('L').point(fn, mode='1')
                    # print(type(cropped_image_data),cropped_image_data.size)
                    #cv2.imwrite("/home/manpreet/Downloads/Xerox_Table/output/cropped/" + str(i) + "_" + str(j) +
                    # ".jpg", np.array(cropped_image_data.convert('RGB')))
                    pred_str = tesserocr.image_to_text(cropped_image_data,psm=6)
                    #cropped_image_data = img_transform(cropped_image_data).unsqueeze(0)
                    # Predict the image
                    #pred_str = prediction_engine(model=model, image=cropped_image_data)
                    # pred_str = tesserocr.image_to_text(cropped_image_data)
                    pred_str_global += pred_str + " "
                    # except:
                    #     pass
                    pred_str_global = pred_str_global.replace("\n", " ").replace("\x0c", "")
                    inner = " ".join(pred_str_global.split())
                    # print("INNER : ",inner)
                outer.append(inner)
                # print("OUTER : ",outer)
    arr = np.array(outer, dtype=object)
    arr = arr.reshape(len(finalboxes), len(finalboxes[0]))
    arr1 = []

    def mostly_blank(temp_arr):
        c = 0
        if sum([1 for x in temp_arr if x == '']) >= len(temp_arr) * 0.9999:
            return True
        else:
            return False

    for i in range(arr.shape[1]):
        if not mostly_blank(arr[:, i]):
            arr1.append(arr[:, i])
    arr1 = np.array(arr1)
    print("ARR1 SHAPE",arr1.shape)
    arr2 = []
    for i in range(arr1.shape[1]):
        if not mostly_blank(arr1[:, i]):
            arr2.append(arr1[:, i])
    arr2 = np.array(arr2)
    result = arr2.T
    return result


def output_to_json(arr):
    idx = ["row_" + str(i) for i in range(arr.shape[0])]
    colms = ["col_" + str(i) for i in range(arr.shape[1])]
    dataframe = pd.DataFrame(arr, index=idx, columns=colms)
    result = dataframe.to_json(orient="index")
    return result


def output_to_xlsx(arr):
    idx = ["row_" + str(i) for i in range(arr.shape[0])]
    colms = ["col_" + str(i) for i in range(arr.shape[1])]
    dataframe = pd.DataFrame(arr, index=idx, columns=colms)
    result = dataframe.style.set_properties(align="left")
    # Converting dataframe into an excel-file
    # result.to_excel("output.xlsx")
    return result

