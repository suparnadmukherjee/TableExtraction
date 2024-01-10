import sys
import os
from pathlib import Path
import time
import logging

from directory_paths import PROJECT_ROOT_DIR
from image_to_table import imagetotable
from utils.pdf_to_image import pdftoimage


#pdf_file_path="data/pdf/tesla.pdf"
png_pages_dir="data/pngpages/"
source_image_path = 'data/testImages/'
cropped_image_path="data/cropped_tables/"
coords_file_path="data/coordinates/"
processed_image_path="data/processed_image/"
csv_out_path="data/csv/"
'''
Step 1: Convert PDF tp PNG Pages
Step 2: Detect tables in the pages and crop and save
Step 3: Extract Text from the tables
Step 4: Output csv files with the chosen approach
'''


def pdf2table(pdf_file_path,choice_of_approaches):
    logging.basicConfig(filename=f'{time.time()}_pdf_to_table.log', level=logging.INFO)
    #converting pdf to table
    pdffilename=Path(pdf_file_path).stem    #getting file name without extension.
    company_dir=f"{png_pages_dir}{pdffilename}"
    if not os.path.exists(company_dir):
        os.makedirs(company_dir)
    logging.info('Vendor Dir created')
    page_image_dir=f"{company_dir}/"
    '''
    Args:
        pdf_file_path:          Path to pdf file
        choice_of_approaches:   list of approches chosen

    Returns:None. Creates a csv file.
    
    1. Approach 1-  Non Max Suppression_IntersectionOverUnion
                    i/p     ->cropped image of a table
                    method  ->Non Max Suppression followed by IOU
                    o/p     ->csv
                                          
    2: Approach 2-  IDP Original Solution 
                    i/p     -> original page image,coordinates of the tables detected,
                    method  ->IDP Heuristic,
                    o/p     ->csv
                    
    3: Approach 3-  IDP 2nd approach with PaddleOCR,
                    i/p     ->cropped image 
                    method  ->Multicolumn HAC clustering algorithm,IDP heuristic
                    o/p     ->csv                    
    '''

    # create png pages from pdf file
    pdftoimage(pdf_file_path, page_image_dir)
    image_list = os.listdir(f"{page_image_dir}")
    image_list=[f"{page_image_dir}{f}" for f in image_list]
    imagetotable(image_list,choice_of_approaches)


    ''''
    # image_list=os.listdir(source_image_path) #use this to get csv of images in testimage folder

    # Detect table from all page images
    for img_path in image_list:
        detect_table(f"{img_path}",)

    #image_list=os.listdir(png_pages_dir)
    # image_list=["enersys_webpage_1.png","enersys_webpage_2.png","enersys_webpage_3.png"]      #use this to get csv for a selected list of pages

    for ch in choice_of_approaches():
        if ch==1:
            table_images=os.listdir(cropped_image_path)
            table_image=[x for x in table_images if img_path[:-4] in x ]    #getting all the cropped table images  from the original image in img_path
            for t_image in table_image:
                t_image=f"{cropped_image_path}{t_image}"
                nms_iou.get_table(t_image)
        elif ch==2:
            flag='a'
            coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"   #getting all the coordinates text file for the page from  the coords_file_path
            idp_1.get_table(img_path,coords_file,flag)
        elif ch==3:
            flag='b'
            #coords_file = f"{coords_file_path}{img_path.split('/')[-1][:-4]}.txt"
            table_images = os.listdir(cropped_image_path)
            table_image = [x for x in table_images if img_path[:-4] in x]
            for t_image in table_image:
                t_image = f"{cropped_image_path}{t_image}"
                detect_table(t_image)
                coords_file=f"{coords_file_path}{t_image.split('/')[-1][:-4]}.txt"
                #img_path = f"{source_image_path}{img_path}"
                idp_2.get_table(t_image,coords_file,flag)
    '''


if __name__=="__main__":

    '''
    ----------------------------------
    Arguments format from CLI
    python3 pdf_to_table.py <path to pdf file> <choice> <choice> <choice>
    python3 pdf_to_table.py data/pdf/tesla.pdf 1 2 3
    -----------------------------------
    '''
    start=time.time()
    pdf_file_path = f"{PROJECT_ROOT_DIR}data/pdf/Nvidia.pdf"#sys.argv[1]
    choice_of_approaches =[1,2,3]#[int(x) for x in sys.argv[2:]]
    # print("welcome to converting odf to table")
    # print(f"you have chosen to detect tables from {pdf_file_path} with {choice_of_approaches} approaches")
    pdf2table(pdf_file_path,choice_of_approaches)  #pass pdf file path and list of approaches to be executed 1,
    end=time.time()
    print("total time elapsed=",(end-start))
    # 2 or 3




