from ultralytics import YOLO
from PIL import Image, ImageOps
import os
# from image_processing.remove_horizontal_lines import remove_h_line
# from image_processing.dialate_vertical_space import add_v_space
# from image_processing.dialate_horizontal_space import add_h_space

model_path="/home/suparna/PycharmProjects/TableExtraction/model/beauty_yolo_ft3_150/weights/best.pt"
markedtable_path="/home/suparna/PycharmProjects/TableExtraction/data/tables_marked/"
cropped_image_path="/home/suparna/PycharmProjects/TableExtraction/data/cropped_tables/"
coords_txt_path="/home/suparna/PycharmProjects/TableExtraction/data/coordinates/"
# load model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(model_path)

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def detect_table(img:str)-> None:
    '''
    Args:
        img: Complete page image
    Returns:None
    steps:
    1. detects the table/tables present in the image
    2. saves the coordinates in coords_txt_path in .txt file with the same as that of the img.
    3. also invokes crop table to crop the image and save in cropped_image_path
    '''

    # 1. detects the table/tables present in the image
    image=img
    results = model.predict(image)
    #render = render_result(model=model, image=image, result=results[0])
    #render.show()

    #2. saves the coordinates in coords_txt_path in .txt file with the same as that of the img.
    file_name = image.split('/')[-1]
    #render = render.save(f'{markedtable_path}{file_name}')
    #render.show()
    # print(results[0].boxes.xyxy,results[0].boxes.xyxyn,results[0].boxes.xywh,results[0].boxes.xywhn)

    #3. also invokes crop table to crop the image and save in cropped_image_path
    crop_table(results,image)


def crop_table(results,image:str) -> None:
    '''    
    Args:
        results:list of tensors as detected by the prediction model
        image: path to image

    Returns:None

    '''
    all_boxes = results[0].boxes.data.cpu().numpy()
    box_coord=[]
    for i, box in enumerate(all_boxes):
        im2 = Image.open(image)
        input_image = ImageOps.grayscale(im2)

        #cropping the tables with an added margin of 20 pixels
        x1, y1, x2, y2, conf, cls = box
        tables = input_image.crop((x1 - 20, y1 - 20, x2 +20, y2 +20))

        table_filename = (f'{cropped_image_path}{image.split("/")[-1][:-4]}_{i + 1}.png')
        box_coord.append([int(x1),int(y1),int(x2),int(y2)])

        tables.save(table_filename)
        #print(box_coord)

    coordsfname=f"{coords_txt_path}{image.split('/')[-1][:-4]}.txt"
    with open(coordsfname, 'w') as f:
        c=str(box_coord)
        f.write(c)


if __name__=="__main__":

    img="/home/suparna/PycharmProjects/TableExtraction/data/pngpages/tesla_0.png" #only image name
    detect_table(img)
