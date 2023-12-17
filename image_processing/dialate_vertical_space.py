#increase blank lines in between text verticall
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

#image_dir="testImages/"
processed_image_path="data/processed_image/"
# Load image, grayscale, Otsu's threshold
def add_v_space(image_path):
    #image_path=f"{image_dir}{fname}"
    #image_path="processed_image/v_space_processed/Apple__36_cropped_margin10_1_h_lineout_v.png"
    fname=image_path
    file_name=fname.split('/')[-1][:-4]

    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Sum white pixels in each row
    # Create blank space array and and final image
    hpixels = np.sum(thresh, axis=0).tolist()
    hspace = np.ones((h, 1), dtype=np.uint8) * 255
    hresult = np.zeros((h, 0), dtype=np.uint8)

    # Iterate through each row and add space if entire row is empty
    # otherwise add original section of image to final image
    for index, value in enumerate(hpixels):
        if value == 0:
            hresult = np.concatenate((hspace,hresult), axis=1)
        col = gray[0:h,index:index+1]
        hresult = np.concatenate((hresult,col), axis=1)

    # Uncomment for plot visualization
    '''
    x = range(len(pixels))[::-1]
    df = pd.DataFrame({'y': x, 'x': pixels})
    df.plot(x='x', y='y', xlim=(-2000,max(pixels) + 2000), legend=None, color='teal')
    '''
    #cv2.imshow('result', hresult)
    #cv2.imshow('thresh', thresh)
    #cv2.imwrite(f"{processed_image_path}{file_name}_v.png",hresult)
    spacedfname=f"{processed_image_path}{file_name}_v.png"
    cv2.imwrite(spacedfname,hresult)
    # print(f"{file_name}.png has been saved")
    # plt.show()
    #cv2.waitKey()
    return spacedfname

# if __name__=="__main__":
#     imglist=os.listdir(image_dir)
#     for img in imglist:
#         add_v_space(img)
if __name__=="__main__":
    image_path = "/home/suparna/PycharmProjects/TableExtraction/data/processed_image/_48_cropped_margin20_1_no_hLine_h.png"
    imgpath=add_v_space(image_path)
    imgpath=add_v_space(imgpath)
    print(imgpath)