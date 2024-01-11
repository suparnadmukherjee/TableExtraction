from PIL import Image
import numpy as np
import json

import cv2
import warnings
warnings.simplefilter("ignore")
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from IDP_compressed.Unbordered.Xerox_Table.table_utils import (table_structure_recognition_all as tsra, table_detection as table_detection)
from IDP_compressed.Unbordered.Xerox_Table.table_utils.table_recognition import recognize_table, output_to_xlsx
from IDP_compressed.Unbordered.Xerox_Table.table_utils.table_recognition import output_to_json
import os
import spacy
import IDP_compressed.Unbordered.Xerox_Table.table_config as table_config
#from xerox-tatras-backend.model_utils.table_utils.table_process import get_extracted_data
from pdf2image import convert_from_path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import enchant

from IDP_compressed.Unbordered.Xerox_Table.table_process import get_extracted_data
eng_dict = enchant.Dict("en_US")

class Detect_tables:
    def __init__(self, path):
        cfg = get_cfg()
        cfg.merge_from_file(table_config.YML_PATH)
        cfg.MODEL.WEIGHTS = table_config.TABLE_DETECTION_WEIGHT  # Set path model .pth
        self.predictor = DefaultPredictor(cfg)
        self.data_dir = path
        self.nlp = spacy.load("en_core_web_sm")

    def process_single_page(self,document_img,file_path,coordinates,flag):

        filename = file_path.split('/')[-1][:-4]
        print("PLOTTING PRED")

        table_coords = coordinates
        # table_coords = [[0,150,1653,1270],[0,1270,1653,1628],[0,1628,1653,1980]]
        for num,table_coord in enumerate(table_coords):
            cv2.imwrite('IDP_compressed/Unbordered/Xerox_Table/output/Table_boundary_marked_data/' + filename + '_'+ str(num) +'.jpg',
                        document_img[table_coord[1]:table_coord[3],table_coord[0]:table_coord[2]]) # table_boundary_marked_img)
            # cv2.imwrite('output/Table_boundary_marked_data/' + filename + '_' + str(num) + '.jpg',
            #             table_boundary_marked_img)
        table_list = []
        for tables in table_coords:
            [x1,y1,x2,y2] = tables
            table_list.append(document_img[y1:y2,x1:x2])
            print("document_img[y2:y1,x1:x2].shape",document_img[y2:y1,x1:x2].shape)
            import matplotlib.pyplot as plt
            # plt.imshow(document_img[y2:y1,x1:x2])
            # plt.show()
            document_img[y2:y1,x1:x2] = 255

        list_table_boxes = []
        print("len table_list",len(table_list))
        tab_count = len(table_list)
        table_json = []
        for tab_counter,table in enumerate(table_list):
            #try:
                print("TABLE SHAPE : ",table.shape)
                # if table.shape[0]!=0 and table.shape[1]!=0:
                finalboxes, output_img, table_cell_marked = tsra.recognize_structure(table)
                print("OUTPUT_IMAGE_SHAPE : ",output_img.shape)
                cv2.imwrite('IDP_compressed/Unbordered/Xerox_Table/output/Table_Cell_Marked/' + filename + '_tab_' + str(tab_counter) + '.jpg', table_cell_marked)
                list_table_boxes.append(finalboxes)
                if flag=='a':
                    arr = recognize_table(finalboxes, table)
                elif flag=='b':
                    arr = get_extracted_data(table,file_path)
                result = output_to_json(arr)
                print("result",result)
                print()
                parsed = json.loads(result)
                table_json.append(parsed)
                with open('data/csv/idp1/'+str(filename)+ '_2' + str(tab_counter) + '.json', 'w') as f:
                    json.dump(parsed, f)
                result = output_to_xlsx(arr.T)
                result.to_excel('data/csv/idp1/' + str(filename) + '_2' + str(tab_counter) + '.xlsx')
                # tab_counter += 1
                # print("table found",tab_count)
            #except:
            #    pass
        return tab_count,table_json,document_img

    def process_document(self,file_path):
        filepath, file_extension = os.path.splitext(file_path)
        print("file_extension",file_extension)
        total_table = 0
        table_jsons = []
        if file_extension == ".pdf":
            cleaned_pages = []
            # Get the pdf images
            pdf_images_curr = convert_from_path(pdf_path=file_path)
            for page_img in pdf_images_curr:
                page_img = np.array(page_img)
                tab_count,table_json,document_image = self.process_single_page(page_img)
                total_table+=tab_count
                table_jsons.extend(table_json)
                cleaned_pages.append(document_image)
            return total_table, table_jsons, cleaned_pages, file_extension

        elif file_extension in ['.tif','.tiff']:
            img = Image.open(file_path)
            image_array_list = []
            for i in range(1000):
                try:
                    img.seek(i)
                    image = np.array(img.convert('RGB'))
                    image_array_list.append(image)
                except EOFError:
                    break
            cleaned_pages = []
            for page_img in image_array_list:
                tab_count,table_json,document_image = self.process_single_page(page_img)
                total_table+=tab_count
                table_jsons.extend(table_json)
                cleaned_pages.append(document_image)
            return total_table, table_jsons, cleaned_pages, file_extension

        elif file_extension in ['.jpg', '.png', '.jpeg', '.bmp']:
            page_img = Image.open(file_path).convert("RGB")
            page_img = np.array(page_img)
            tab_count, table_json, document_image = self.process_single_page(page_img)
            # print("TABLE JSON",table_json)
            total_table+=tab_count
            table_jsons.extend(table_json)
            return total_table, table_jsons, [document_image], file_extension

        else:
            return None, None, None, file_extension


def get_table(file_path,coords_file,flag):

    with open(coords_file,'r') as cf:
        cd=cf.readline()
        coordinates=json.loads(cd)  #converting str list to int list

    obj=Detect_tables(file_path)
    page_img = Image.open(file_path).convert("RGB")
    page_img = np.array(page_img)
    tab_count, table_json, document_image = obj.process_single_page(page_img,file_path,coordinates,flag)
    # print("TABLE JSON",table_json)
    total_table = 0
    table_jsons = []
    total_table += tab_count
    table_jsons.extend(table_json)
    #return total_table, table_jsons, [document_image]

if __name__=="__main__":
    file_path = "/home/suparna/PycharmProjects/TableExtraction/extraction/Apple_10-K-2021_48_-20.png"
    print("File:", file_path)
    # coords_file=f"{coordinates_path}{img[:-4]}.txt"
    coords_file = "/home/suparna/PycharmProjects/TableExtraction/extraction/Apple_10-K-2021_48_-20.txt"
    print("coords:", coords_file)
    flag='b'
    get_table(file_path,coords_file,flag)