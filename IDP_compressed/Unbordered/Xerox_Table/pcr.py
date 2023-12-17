import os
import math
import spacy
import json
import string
import enchant
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import time
from PIL import Image
from tqdm import tqdm
import sys
import detection, tools
import torch
import torch.utils.data
import torchvision.transforms as transforms

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

import warnings
warnings.simplefilter("ignore")

# Initialize the device
device = torch.device('cpu')

# Init the dictionary for english
eng_dict = enchant.Dict("en_US")

from collections import defaultdict
import re
import locale
locale.setlocale(locale.LC_ALL, 'C')
import tesserocr
# import extract_msg
# import textract
# from email import message_from_file
# from xml.etree import cElementTree as ET
from operator import itemgetter

import efficientnet.tfkeras as efficientnet
from tensorflow import keras

os.environ["TOKENIZERS_PARALLELISM"] = "false"
path = "./msgfiles"

class args:
    checkpoint="/home/abc/Documents/xerox-tatras-backend/model/epoch=99-step=134051-val_accuracy=91.1990-val_NED=96.9191.ckpt"

class Pipeline:
    """A wrapper for detection.

    Args:
        detector: The detector to use
        scale: The scale factor to apply to input images
        max_size: The maximum single-side dimension of images for
            inference.
    """
    def __init__(self, detector=None, scale=2, max_size=2048):
        if detector is None:
            detector = detection.Detector()
        self.scale = scale
        self.detector = detector
        self.max_size = max_size

    def recognize(self, images, detection_kwargs=None):
        """Run the pipeline on one or multiples images.

        Args:
            images: The images to parse (can be a list of actual images or a list of filepaths)
            detection_kwargs: Arguments to pass to the detector call

        Returns:
            A list of lists of text bounding boxes.
        """

        # Make sure we have an image array to start with.
        if not isinstance(images, np.ndarray):
            images = [tools.read(image) for image in images]
        # This turns images into (image, scale) tuples temporarily
        images = [tools.resize_image(image, max_scale=self.scale, max_size=self.max_size) for image in images]
        max_height, max_width = np.array([image.shape[:2] for image, scale in images]).max(axis=0)
        scales = [scale for _, scale in images]
        images = np.array([tools.pad(image, width=max_width, height=max_height) for image, _ in images])
        if detection_kwargs is None:
            detection_kwargs = {}
        box_groups = self.detector.detect(images=images, **detection_kwargs)
        box_groups = [tools.adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale) if scale != 1 else boxes
                      for boxes, scale in zip(box_groups, scales)]
        return [boxes.tolist() for boxes in box_groups]


class PCR_HWR:
    def __init__(self, img_list):
        self.pipeline = Pipeline()
        self.img_data = img_list
        self.nlp = spacy.load("en_core_web_sm")

    def get_skew_angle(self, cvImage) -> float:
        '''
        Function : get_skew_angle (Calculate skew angle of an image).
        Input : Image.
        Output : Skew Angle.
        '''
        # Prep image, copy, convert to gray scale, blur, and threshold
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    # Rotate the image around its center
    def rotate_image(self, cvImage, angle: float):
        '''
        Function : rotate_image (Rotates image given a skew_angle).
        Input : Image and Angle generated using get_skew_angle function.
        Output : Rotated Image.
        '''
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    # Deskew image
    def deskew(self, image):
        '''
        Function : deskew (Deskew Image).
        Input : Unprocessed Skewed Image.
        Output : Processed deskewed Image.
        '''
        angle = self.get_skew_angle(image)
        return self.rotate_image(image, -1.0 * angle)

    # SORTING BOUNDING BOX COORDINATES
    def sort_cluster(self, list_coordinates):
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

    #  COLUMN/BLOCKS DETECTION FOR MULTI-COLUMN TEXT DOCUMENT
    def column_detection(self, images):
        '''
        Function : column_detection (detect blocks/columns of text and outputs its bounding box coordinates).
        Input : Images list.
        Output : List of Column/Block coordinates in the format [[x0,y0,x1,y1] , [x0,y0,x1,y1], ...].
        '''
        img = images[0].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # black and white, and inverted, because white pixels are treated as objects in contour detection
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

        # Used a kernel that is wide enough to connect characters
        # but not text blocks, and tall enough to connect lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 33))
        closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bb = []
        for cont in range(len(contours)):
            x_values = []
            y_values = []
            for j in contours[cont]:
                x_values.append(j[0][0])
                y_values.append(j[0][1])
            x0 = min(x_values)
            x1 = max(x_values)
            y0 = min(y_values)
            y1 = max(y_values)
            bb.append([x0, y0, x1, y1])
        return bb

    # def tup(self,point):
    #     return (int(point[0]), int(point[1]))

    # returns true if the two boxes overlap
    # def overlap(self,source, target):
    #     # unpack points
    #     tl1, br1 = source
    #     tl2, br2 = target
    #     # checks
    #     if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
    #         return False
    #     if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
    #         return False
    #     return True

    # returns all overlapping boxes
    # def getAllOverlaps(self,boxes, bounds, index):
    #     overlaps = []
    #     for a in range(len(boxes)):
    #         if a != index:
    #             if self.overlap(bounds, boxes[a]):
    #                 overlaps.append(a)
    #     return overlaps

    # def group_x_close_boxes(self,orig, boxes, merge_margin=1):
    #     # filter out excessively large boxes
    #     filtered = []
    #     max_area = 3000
    #     # for box in boxes:
    #     #     w = box[1][0] - box[0][0]
    #     #     h = box[1][1] - box[0][1]
    #     #     if w*h < max_area:
    #     #         filtered.append(box)
    #     # boxes = filtered
    #     # go through the boxes and start merging
    #     # this is gonna take a long time
    #     finished = False
    #     highlight = [[0, 0], [1, 1]]
    #     points = [[[0, 0]]]
    #     while not finished:
    #         # set end con
    #         finished = True
    #         # check progress
    #         # print("Len Boxes: " + str(len(boxes)))
    #         # draw boxes # comment this section out to run faster
    #         copy = np.copy(orig)
    #         for box in boxes:
    #             cv2.rectangle(copy, self.tup(box[0]), self.tup(box[1]), (0, 200, 0), 1)
    #         cv2.rectangle(copy, self.tup(highlight[0]), self.tup(highlight[1]), (0, 0, 255), 2)
    #         for point in points:
    #             point = point[0]
    #             cv2.circle(copy, self.tup(point), 4, (255, 0, 0), -1)
    #         # cv2_imshow(copy)
    #         # key = cv2.waitKey(1)
    #         # if key == ord('q'):
    #         #     break
    #         # loop through boxes
    #         index = 0
    #         while index < len(boxes):
    #             # grab current box
    #             curr = boxes[index]
    #             # add margin
    #             tl = curr[0][:]
    #             br = curr[1][:]
    #             tl[0] -= merge_margin
    #             # tl[1] -= merge_margin
    #             br[0] += merge_margin
    #             # br[1] += merge_margin
    #             # get matching boxes
    #             overlaps = self.getAllOverlaps(boxes, [tl, br], index)
    #             # check if empty
    #             if len(overlaps) > 0:
    #                 # combine boxes
    #                 # convert to a contour
    #                 con = []
    #                 overlaps.append(index)
    #                 for ind in overlaps:
    #                     tl, br = boxes[ind]
    #                     con.append([tl])
    #                     con.append([br])
    #                 con = np.array(con)
    #                 # get bounding rect
    #                 x, y, w, h = cv2.boundingRect(con)
    #                 # stop growing
    #                 w -= 1
    #                 h -= 1
    #                 merged = [[x, y], [x + w, y + h]]
    #                 # highlights
    #                 highlight = merged[:]
    #                 points = con
    #                 # remove boxes from list
    #                 overlaps.sort(reverse=True)
    #                 for ind in overlaps:
    #                     del boxes[ind]
    #                 boxes.append(merged)
    #
    #                 # set flag
    #                 finished = False
    #                 break
    #             # increment
    #             index += 1
    #     return boxes

    @torch.inference_mode()
    def prediction_engine(self, model, image):
        """ validation or evaluation """
        # Collect data from the image
        image = image.to(device)
        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        return str(pred[0]).strip()

    def image_to_text(self, image_data, model, img_transform):
        '''
        Function : image_to_text (Extract text after identification of word level bounding box coordinates).
        Input : Image.
        Output : Extracted raw_text, identified_words and processed_image.
        '''
        # Load the image and collect the predictions in terms of boxes
        #image_data = np.array(image_data)

        image_for_kcr = [tools.read(image_data)]
        prediction_groups = Pipeline().recognize(image_for_kcr)

        # Collect the boxes
        boxes = []
        for elem in prediction_groups[0]:
            x0 = elem[0][0]
            y0 = elem[0][1]
            x1 = elem[2][0]
            y1 = elem[2][1]
            boxes.append(["text", [x0, y0, x1, y1]])
        #merge_image = image_data.copy()


        # Sort the clusters
        sorted_clusters = self.sort_cluster(list_coordinates=boxes)
        # merge_bb = []
        # for clus in sorted_clusters:
        #     inner_list = []
        #     first = [int(clus[1][0]),int(clus[1][1])]
        #     second = [int(clus[1][2]),int(clus[1][3])]
        #     inner_list.append(first)
        #     inner_list.append(second)
        #     merge_bb.append(inner_list)
        #
        # out_boxes = self.group_x_close_boxes(orig = merge_image,boxes = merge_bb)
        # sc = []
        # for box in out_boxes:
        #     sc.append(['text',[box[0][0],box[0][1],box[1][0],box[1][1]]])
        # sc = self.sort_cluster(list_coordinates=sc)
        img_draw = image_data.copy()
        for box_ in sorted_clusters:
            cv2.rectangle(img_draw,pt1=(int(box_[1][0]), int(box_[1][1])), pt2=(int(box_[1][2]), int(box_[1][3])),
                                         color=(255, 0, 0), thickness=2)
        # Crop the image
        pred_str_global = ""
        bar = tqdm(sorted_clusters, leave=False)
        identified_words = {}
        for box_curr in bar:
            # Crop and collect
            x0, y0, x1, y1 = map(int, box_curr[-1])
            if x0 < x1 and y0 < y1:
                try:
                    bb_coordinates = [x0, y0, x1, y1]
                    # Crop the image
                    cropped_image_data = Image.fromarray(image_data[y0  : y1 , x0  : x1 ]).convert("L").convert("RGB")
                    cropped_image_data = img_transform(cropped_image_data).unsqueeze(0)
                    # Predict the image
                    pred_str = self.prediction_engine(model=model, image=cropped_image_data).replace("$","")
                    if len(pred_str) == 0:
                        pred_str = ""
                    # add to the global string
                    pred_str_global += pred_str + " "
                    bar.set_description(f"Curr pred : {pred_str}")
                    identified_words[pred_str] = bb_coordinates
                except:
                    pass
        pred_str_global = pred_str_global.replace("\n", " ").replace("\x0c", "")
        return " ".join(pred_str_global.split()), identified_words, img_draw

    def extraction_from_images_list(self, model, img_transform):
        '''
        Function : extraction_from_images_list (Processes single file from img_list and return text, pages, identified_words and processed_image).
        Input : trained_model, image_transforms
        Output : raw_text list, number_of_pages_processed, identified_words, processed_image
        '''
        data_file = self.img_data
        text = []
        identified_words = []
        img_draw = []
        for idx in tqdm(range(len(data_file)), leave=False):
            global_text_, identified_words_, img_draw_ = self.image_to_text(image_data=data_file[idx],
                                                                            model=model,
                                                                            img_transform=img_transform)
            global_text_ = global_text_.replace("\n"," ")
            text.append(global_text_)
            identified_words.append(identified_words_)
            img_draw.append(img_draw_)
        return text, len(data_file), identified_words, img_draw

    def get_ner_tags_dict(self, text_in, ner_model):
        '''
        Function : get_ner_tags_dict (Will collect the ner data and return its dict)
        Input : extracted_text , ner_model
        Output : Dictionary with ner tags information.
        '''
        doc = ner_model(text_in)
        result_db = []
        for ent in doc.ents:
            result_db.append({"Named_Entity": ent.text,
                              "Start_Char": str(ent.start_char),
                              "End_Char": str(ent.end_char),
                              "Entity_Explanation": str(spacy.explain(ent.label_)),
                              "entity_label": ent.label_})
        return result_db

    def return_json(self):
        '''
        Function : return_json
        Input : None
        Output : json response with raw_text, pages_processed, no_of_extracted_words, no_of_dictionary_words, named_entities, identified_words.
        '''
        kwargs = {
            "refine_iters": 3
        }
        # Get the transforms and the model
        model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        output, pages, identified_words, img_draw = self.extraction_from_images_list(model=model,
                                                                                          img_transform=img_transform)
        ner_tags_text = []
        dic_word_list = []
        extracted_word_list = []
        for extracted_text_per_page in output:
            ner_tags_text_curr = self.get_ner_tags_dict(text_in=extracted_text_per_page, ner_model=self.nlp)
            ner_tags_text.append(ner_tags_text_curr)
            dic_word = 0
            for word in extracted_text_per_page.lower().split():
                if eng_dict.check(word):
                    dic_word = dic_word + 1
            dic_word_list.append(dic_word)
            extracted_words_count = len(extracted_text_per_page.split(" "))
            extracted_word_list.append(extracted_words_count)
            total_named_entity = 0
            unique_named_entity = 0
            unique_ner = []
            for i in ner_tags_text:
                unique_ner.append({frozenset(item.items()): item for item in i}.values())
            for i in range(len(ner_tags_text)):
                total_named_entity += len(ner_tags_text[i])
                unique_named_entity += len(unique_ner[i])
        return {"raw_test": output,
                "pages_processed": pages,
                "no_of_extracted_words": sum(extracted_word_list),
                "no_of_dictionary_words": sum(dic_word_list),
                "named_entities": ner_tags_text,
                "total_named_entity": total_named_entity,
                "unique_named_entity": unique_named_entity,
                "identified_words": identified_words}, img_draw
