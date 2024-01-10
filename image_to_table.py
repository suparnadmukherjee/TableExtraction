import sys
import os
import time
import logging
from directory_paths import PROJECT_ROOT_DIR
from image_processing import remove_horizontal_lines,dialate_horizontal_space,dialate_vertical_space
from extraction.table_detection import detect_table
from approaches import nms_iou, idp_1, idp_2

# from IDP_compressed.Unbordered.Xerox_Table import processes_document, table_utils
# from ultralyticsplus import YOLO, render_result
# from PIL import Image, ImageOps
# from utils.pdf_to_image import pdftoimage
# from utils import pdf_to_image

#pdf_file_path = "data/pdf/tesla.pdf"
png_pages_dir = "data/pngpages/"
source_image_path = 'data/testImages/'
cropped_image_path = "/home/suparna/PycharmProjects/TableExtraction/data/cropped_tables/"
coords_file_path = "data/coordinates/"
processed_image_path = "data/processed_image/"
csv_out_path = "data/csv/"
'''
Step 1: Convert PDF tp PNG Pages
Step 2: Detect tables in the pages and crop and save
Step 3: Extract Text from the tables
Step 4: Output csv files with the chosen approach
'''



def imagetotable(image_list,choice_of_approaches):

    '''
    Args:
        image_list: list of full paths to page images
        choice_of_approaches: list of integers

    Returns:None

    1. Approach 1-  Non Max Suppression
                    i/p     ->cropped image of a table
                    method  ->Non Max Suppression followed by IOU
                    o/p     ->csv

    2: Approach 2-  IDP Original Solution 
                    i/p     -> original page image,coordinates of the tables detected,
                    method  ->IDP Heuristic,
                    o/p     ->csv

    3: Approach 3-  IDP 2nd approach ,
                    i/p     ->cropped image 
                    method  ->Multicolumn HAC clustering algorithm,IDP heuristic
                    o/p     ->csv
    '''


    # image_list = os.listdir(source_image_path)
    # image_list = ["enersys_webpage_1.png", "enersys_webpage_2.png", ]

    for img_path in image_list:
        # TODO: To be put inside config file
        # img_path=remove_horizontal_lines.remove_h_line2(img_path)
        # img_path=dialate_horizontal_space.add_h_space(img_path)
        # img_path=dialate_vertical_space.add_v_space(img_path)
        detect_table(f"{img_path}")#/home/suparna/PycharmProjects/TableExtraction/

    # image_list=os.listdir(png_pages_dir)
    # image_list=["enersys_webpage_1.png","enersys_webpage_2.png","enersys_webpage_3.png"]                      #use this to get csv for a selected list of pages

        for ch in choice_of_approaches:
            if ch == 1:

                table_images = os.listdir(cropped_image_path)
                table_image = [x for x in table_images if img_path.split('/')[-1][:-4] in x]                   # getting all the cropped table images  from the original image in img_path
                for t_image in table_image:
                    t_image = f"{cropped_image_path}{t_image}"
                    logging.info(f"{t_image} under nms_iou process")
                    nms_iou.get_table(t_image)
            elif ch == 2:
                flag = 'a'
                coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"                           # getting all the coordinates text file for the page from  the coords_file_path
                logging.info(f"{img_path} under idp1 process")
                idp_1.get_table(img_path, coords_file, flag)
            elif ch == 3:
                flag = 'b'
                # coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
                table_images = os.listdir(cropped_image_path)
                table_image = [x for x in table_images if img_path.split('/')[-1][:-4] in x]
                for t_image in table_image:
                    t_image = f"{cropped_image_path}{t_image}"
                    detect_table(t_image)
                    coords_file = f"{coords_file_path}{t_image.split('/')[-1][:-4]}.txt"
                    # img_path = f"{source_image_path}{img_path}"
                    logging.info(f"{t_image} under idp2 process")
                    idp_2.get_table(t_image, coords_file, flag)


if __name__ == "__main__":
    '''
    ----------------------------------
    Arguments format from CLI
    python3 image_to_table.py ["path to image file"] <choice> <choice> <choice>
    python3 image_to_table.py data/pdf/tesla.pdf 1 2 3
    -----------------------------------
    '''


    # store starting time
    begin = time.time()

    #list_of_images =os.listdir("/home/suparna/PycharmProjects/Table Extraction Annotation")#sys.argv[1]
    list_of_images=["/home/suparna/PycharmProjects/TableDetection/all_Data/png_images-Ashis/png_images/B_A_33.png",
                    "/home/suparna/PycharmProjects/TableDetection/all_Data/png_images-Ashis/png_images/B_B_28.png",
                    "/home/suparna/PycharmProjects/TableDetection/all_Data/png_images-Ashis/png_images/R_A_84.png",
                    "/home/suparna/PycharmProjects/TableDetection/all_Data/png_images-Ashis/png_images/R_C_62.png"]
    choice_of_approaches = [1]#[int(x) for x in sys.argv[2:]]

    imagetotable(list_of_images,choice_of_approaches)  #pass pdf file path and list of approaches to be executed 1,
    # 2 or 3
    # store end time
    end = time.time()

    # total time taken
    print(f"Total runtime of the program is {end - begin}")




