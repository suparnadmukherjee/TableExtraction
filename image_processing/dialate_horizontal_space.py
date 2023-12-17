#increase blank lines in between text horizontal
import cv2
import numpy as np

processed_image_path="data/processed_image/"

def add_h_space(image_path) :
    '''
    ------------------------------------------------------------------------------
    Args:
        image_path: str-> image path

    Returns:
        results:ndarray format of the input image after dialating horizontal space
    ------------------------------------------------------------------------------
    '''

    # Load image, grayscale, Otsu's threshold
    fname=image_path.split('/')[-1][:-4]
    #image_path = f"{image_dir}{fname}"

    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Sum white pixels in each row
    # Create blank space array and and final image
    pixels = np.sum(thresh, axis=1).tolist()
    space = np.ones((1, w), dtype=np.uint8) * 255
    result = np.zeros((0, w), dtype=np.uint8)

    # Iterate through each row and add space if entire row is empty
    # otherwise add original section of image to final image
    for index, value in enumerate(pixels):
        if value == 0:
            result = np.concatenate((result, space), axis=0)
        row = gray[index:index+1, 0:w]
        result = np.concatenate((result, row), axis=0)


    #use next 2 lines for testing horizontal spacing
    spacedfname=f"{processed_image_path}h_{fname}.png"
    cv2.imwrite(spacedfname,result)
    #plt.show()
    return spacedfname



if __name__=="__main__":
    image_path="/home/suparna/PycharmProjects/TableExtraction/data/cropped_tables/_48_cropped_margin20_1_no_hLine.png"
    r=add_h_space(image_path)