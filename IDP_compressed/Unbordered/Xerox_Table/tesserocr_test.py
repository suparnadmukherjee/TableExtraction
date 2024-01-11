import tesserocr
from tqdm import tqdm
import cv2
import os
from PIL import Image
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.module import SceneTextDataModule
import torch.utils.data
import matplotlib.pyplot as plt

def get_text(image):
    img = cv2.imread(image,0)
    # .png_cropped_margin10_1_.png",0)
    device = torch.device('cpu')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
    #print(thresh.shape)
    plt.imshow(thresh)
    img = Image.fromarray(img).convert("L").convert("RGB")
    text = tesserocr.image_to_text(img)
    # print("TEXT : ",text)
    # print("----------------------------------------------------------------")
    return text


if __name__=="__main__":
    dir_path="/home/suparna/PycharmProjects/TableDetection/Approaches_results/TransDigm/cropped_tables/"
    txt_dir="/home/suparna/PycharmProjects/TableDetection/Approaches_results/TransDigm/tableTxt/"
    files=os.listdir(dir_path)
    for f in tqdm(files):
        #print(f)
        text=get_text(dir_path+f)
        with open(f'{txt_dir}{f[:-4]}.txt','w') as txtf:
            txtf.write(text)


#get_text()
# kwargs = {
#             "refine_iters": 3
#         }
# @torch.inference_mode()
# def prediction_engine(model, image):
#     """ validation or evaluation """
#     # Collect data from the image
#     image = image.to(device)
#     p = model(image).softmax(-1)
#     pred, p = model.tokenizer.decode(p)
#     return str(pred[0]).strip()
# # Get the transforms and the model
# model = load_from_checkpoint("/home/abc/Downloads/Handwritten_Code/model_weight/epoch=99-step=134051-val_accuracy=91.1990-val_NED=96.9191.ckpt", **kwargs).eval().to(device)
# img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
# cropped_image_data = img_transform(img).unsqueeze(0)
# # Predict the image
# pred_str = prediction_engine(model=model, image=cropped_image_data)
# print(pred_str)