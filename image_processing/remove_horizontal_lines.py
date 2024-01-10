
import cv2
import numpy as np
import argparse

processed_image_path="/home/suparna/PycharmProjects/TableExtraction/data/processed_image/"#"/home/suparna/PycharmProjects/TableExtraction/data/processed_image/"
def remove_h_line(img):#not in use

    image = cv2.imread(img)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('detected_lines', detected_lines)
    #cv2.imshow('image', image)
    #cv2.imshow('result', result)
    cv2.imwrite("no_hline.png",result)
    #cv2.waitKey()

def remove_h_line2(img_path):


    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to the image file")
    # args = vars(ap.parse_args())

    # load the image and convert it to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get binary inverted image
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -2)

    # fill the broken lines
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.dilate(thresh, kernal, iterations=2)
    thresh = cv2.erode(thresh, kernal, iterations=1)

    # Make a copy for horizontal and vertical
    horz = np.copy(thresh)
    vert = np.copy(thresh)

    # Set the kernel
    (_, horzcol) = np.shape(horz)
    horzSize = int(horzcol / 26)
    horzStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horzSize, 1))

    # perform a series of erosions and dilations
    horz = cv2.erode(horz, horzStructure, iterations=1)
    horz = cv2.dilate(horz, horzStructure, iterations=1)

    # Get the output image after inversion
    outimage = cv2.bitwise_and(~gray, ~horz)
    outimage = ~outimage
    outfname=f"{processed_image_path}{img_path.split('/')[-1][:-4]}_noL.png"
    cv2.imwrite(outfname, outimage)
    '''
    edges = cv2.Canny(outimage,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    #print lines
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(outimage,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('houghlines5.png',outimage)
    '''
    return outfname
if __name__=="__main__":
    img="/home/suparna/PycharmProjects/TableExtraction/data/cropped_tables/_48_cropped_margin20_1.png"
    remove_h_line2(img)