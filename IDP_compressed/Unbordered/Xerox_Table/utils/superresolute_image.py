from PIL import Image
import matplotlib as plt

def get_srImage(img):

    img = img.convert("RGBA")

    new_img = Image.blend(img, img, 0.5)
    new_img.save("new.png1","PNG")
    new_img.show()
    plt.imshow(new_img)