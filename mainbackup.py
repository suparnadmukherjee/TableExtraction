import sys
import os
from extraction.table_detection import detect_table
from approaches import nms_iou, idp_1, idp_2, multicol_HAC

from IDP_compressed.Unbordered.Xerox_Table import processes_document, table_utils
from ultralyticsplus import YOLO, render_result
from PIL import Image, ImageOps
from utils.pdf_to_image import pdftopages
from utils import pdf_to_image

pdf_file_path = "data/pdf/tesla.pdf"
png_pages_dir = "data/pngpages/"
source_image_path = 'data/testImages/'
cropped_image_path = "data/cropped_tables/"
coords_file_path = "data/coordinates/"
processed_image_path = "data/processed_image/"
csv_out_path = "data/csv/"
'''
Step 1: Convert PDF tp PNG Pages
Step 2: Detect tables in the pages and crop and save
Step 3: Extract Text from the tables
Step 4: Output csv files with the chosen approach
'''


def pdftable(pdf_file_path, choice_of_approaches):
    '''

    Args:
        pdf_file_path:          Path to pdf file
        choice_of_approaches:   list of approches chosen

    Returns:

    '''

    # create png pages from pdf file
    pdftopages(pdf_file_path, png_pages_dir)

    '''
    1. Approach 1-  Non Max Supression
                    i/p     ->cropped image of a table
                    method  ->Non Max Supression followed by IOU
                    o/p     ->csv

    2: Approach 2-  IDP Original Solution 
                    i/p     -> original page image,coordinates of the tables detected,
                    method  ->IDP Heuristic,
                    o/p     ->csv

    3: Approach 3-  IDP 2nd approach ,
                    i/p     ->cropped image 
                    method  ->Multicolumn HAC clustering algorithm,IDP heuristic
                    o/p     ->csv 

    # 4. Approach 4- Su_heuristics
    #                 i/p     ->cropped image
    #                 method  ->Multicolumn HAC clustering algorithm,Su_heuristic 
    #                 o/p     ->csv

    '''

    image_list = os.listdir(source_image_path)

    # Detect table from all page images
    for img_path in image_list:
        detect_table(f"{source_image_path}{img_path}", )

    image_list = os.listdir(png_pages_dir)  # use this to get csv of all png pages
    # image_list=["enersys_webpage_1.png",
    #             "enersys_webpage_2.png",
    #             "enersys_webpage_3.png"]  #use this to get csv for a selected list of pages
    for ch in choice_of_approaches():
        if ch == 1:
            table_images = os.listdir(cropped_image_path)
            table_image = [x for x in table_images if img_path[:-4] in x]
            for t_image in table_image:
                t_image = f"{cropped_image_path}{t_image}"
                nms_iou.get_table(t_image)
        elif ch == 2:
            flag = 'a'
            coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
            img_path = f"{source_image_path}{img_path}"
            idp_1.get_table(img_path, coords_file, flag)
        elif ch == 3:
            flag = 'b'
            # coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
            table_images = os.listdir(cropped_image_path)
            table_image = [x for x in table_images if img_path[:-4] in x]
            for t_image in table_image:
                t_image = f"{cropped_image_path}{t_image}"
                detect_table(t_image)
                coords_file = f"{coords_file_path}{t_image.split('/')[-1][:-4]}.txt"
                # img_path = f"{source_image_path}{img_path}"
                idp_2.get_table(t_image, coords_file,
                                flag)  # elif ch==4:  #     table_images = os.listdir(cropped_image_path)  #     table_image = [x for x in table_images if img_path[:-4] in x]  #     for t_image in table_image:  #         t_image = f"{cropped_image_path}{t_image}"  #         multicol_HAC.get_table(t_image)


def singlepage_to_table():
    print("welcome to converting odf to table")

    # create png pages from pdf file
    # 1pdftopages(pdf_file_path, png_pages_dir)

    '''
    1. Approach 1-  Non Max Supression
                    i/p     ->cropped image of a table
                    method  ->Non Max Supression followed by IOU
                    o/p     ->csv

    2: Approach 2-  IDP Original Solution 
                    i/p     -> original page image,coordinates of the tables detected,
                    method  ->IDP Heuristic,
                    o/p     ->csv

    3: Approach 3-  IDP 2nd approach ,
                    i/p     ->cropped image 
                    method  ->Multicolumn HAC clustering algorithm,IDP heuristic
                    o/p     ->csv 

    # 4. Approach 4- Su_heuristics
    #                 i/p     ->cropped image
    #                 method  ->Multicolumn HAC clustering algorithm,Su_heuristic 
    #                 o/p     ->csv

    '''

    # image_list = os.listdir(source_image_path)
    image_list = ["enersys_webpage_1.png", "enersys_webpage_2.png", ]

    # test images in the source image path creating csv by any one of the options described above
    ch = int(input("Enter your choice"))
    for img_path in image_list:
        # Detect table from page images
        detect_table(f"{source_image_path}{img_path}", )

        # image_list=os.listdir(png_pages_dir)  #use this to get csv of all png pages

        if ch == 1:
            table_images = os.listdir(cropped_image_path)
            table_image = [x for x in table_images if img_path[:-4] in x]
            for t_image in table_image:
                t_image = f"{cropped_image_path}{t_image}"
                nms_iou.get_table(t_image)
        elif ch == 2:
            flag = 'a'
            coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
            img_path = f"{source_image_path}{img_path}"
            idp_1.get_table(img_path, coords_file, flag)
        elif ch == 3:
            flag = 'b'
            # coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
            table_images = os.listdir(cropped_image_path)
            table_image = [x for x in table_images if img_path[:-4] in x]
            for t_image in table_image:
                t_image = f"{cropped_image_path}{t_image}"
                detect_table(t_image)
                coords_file = f"{coords_file_path}{t_image.split('/')[-1][:-4]}.txt"
                # img_path = f"{source_image_path}{img_path}"
                idp_2.get_table(t_image, coords_file,
                                flag)  # elif ch==4:  #     table_images = os.listdir(cropped_image_path)  #     table_image = [x for x in table_images if img_path[:-4] in x]  #     for t_image in table_image:  #         t_image = f"{cropped_image_path}{t_image}"  #         multicol_HAC.get_table(t_image)


if __name__ == "__main__":
    '''
    ----------------------------------
    Arguments format from CLI
    python3 pdf_to_table.py <path to pdf file> <choice> <choice> <choice>
    python3 pdf_to_table.py data/pdf/tesla.pdf 1 2 3
    -----------------------------------
    '''
    pdf_file_path = sys.argv[1]
    choice_of_approaches = [int(x) for x in sys.argv[2:]]
    print("welcome to converting odf to table")
    print(
        f"you have chosen to detect tables from {pdf_file_path} with {choice_of_approaches} approaches")  # pdf_to_table(pdf_file_path,choice_of_approaches)  #pass pdf file path and list of approaches to be executed 1,  # 2 or 3




