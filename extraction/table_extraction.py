from ultralyticsplus import YOLO, render_result
from PIL import Image, ImageOps
import os
from image_processing.remove_horizontal_lines import remove_h_line
from image_processing.dialate_vertical_space import add_v_space
from image_processing.dialate_horizontal_space import add_h_space

# source_image_path = ('data/testImages/')
# cropped_image_path="data/cropped_tables/"
# coords_file_path="data/coordinates"
#company="Enersys"
#model_path="/home/suparna/PycharmProjects/TableDetection/YOLO_FT_tables/best_model/keremberke8m_10_50/weights/best.pt"

model_path="model/keremberke8m_50/weights/best.pt"
# load model
model = YOLO(model_path)

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def detect_crop_save_tables(img,cropped_image_path):

    # set image
    # image = "/home/suparna/PycharmProjects/TableDetection/all_Data/FinQA_tables/combined_tables-056.png"

    # perform inference
    #image=source_image_path + img
    image=img
    results = model.predict(image)
    print(results)
    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image, result=results[0])
    #render.show()
    file_name = image


    #render = render.save(f'{file_name}')
    # print(results[0].boxes.xyxy)
    # print(results[0].boxes.xyxyn)
    # print(results[0].boxes.xywh)
    # print(results[0].boxes.xywhn)

    all_boxes = results[0].boxes.data.cpu().numpy()
    box_coord=[]
    for i, box in enumerate(all_boxes):
        im2 = Image.open(image)
        input_image = ImageOps.grayscale(im2)

        x1, y1, x2, y2, conf, cls = box
        tables = im2.crop((x1 - 20, y1 - 20, x2 +20, y2 +20))

        #to do add remove hline,dialate horizontal,dialte vertical

        table_filename = (f'{cropped_image_path}'
                          f'{image[-7:-4]}_cropped_margin20_{i + 1}.png')
        box_coord.append([int(x1),int(y1),int(x2),int(y2)])

        tables.save(table_filename)
    with open(f"{img[:-4]}_-20.txt", 'w') as f:#{coords_file_path}
        c=str(box_coord)
        f.write(c)

#detect_crop_save_tables("/home/suparna/Downloads/out.jpg")
# images = os.listdir(source_image_path)
# for img in images:
#     print(img)
#     detect_crop_save_tables(img)