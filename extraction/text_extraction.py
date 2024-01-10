#importing required packages
from paddleocr import PaddleOCR,draw_ocr
import cv2
import os


outfilepath="/home/suparna/PycharmProjects/TableExtraction/data/box_text_marked/"
def paddelExtract(image_path):
    #downloading Paddle OCR model for English
    ocr=PaddleOCR(lang='en')

    filename=os.path.basename(image_path)

    #reading the image
    image_cv=cv2.imread(image_path)
    image_height=image_cv.shape[0]
    image_width=image_cv.shape[1]

    #print(image_height,image_width)************

    #Extracting the text from image
    output=ocr.ocr(image_path)

    output=output[0]
    #print(output)**********

    boxes=[line[0] for line in output]
    texts=[line[1][0] for line in output]
    probabilities=[line[1][1] for line in output]
    # for box in texts:
    #   print(box)*****************

    image_boxes=image_cv.copy()
    image_text_boxes=image_cv.copy()

    #font=(cv2.FONT_HERSHEY_SIMPLEX,1,1,0,1)

    #Drawing Rectangles and text on image
    for box,text in zip(boxes,texts):
      cv2.rectangle(image_boxes,(int(box[0][0]),int(box[0][1])),(int(box[2][0]),int(box[2][1])),(0,0,255),1)
      cv2.putText(image_text_boxes,text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

    cv2.imwrite(f"{outfilepath}{filename}.png",image_boxes)
    #cv2.imwrite("boxed_image.png",image_boxes)
    print("textand bb marked, ocr complete")
    return output

if __name__=="__main__":
    op=paddelExtract("/home/suparna/PycharmProjects/TableExtraction/extraction/Apple_10-K-2021_48_-20.png")