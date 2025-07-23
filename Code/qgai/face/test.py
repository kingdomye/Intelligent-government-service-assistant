import io
import base64
from PIL import Image


def img_to_bin(image):
    img_bin = io.BytesIO()
    image.save(img_bin, format='JPEG')
    image_bytes = img_bin.getvalue()

    return image_bytes


def bin_to_img(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))

    return img


if __name__ == '__main__':
    img = Image.open('facedata/test.jpg')
    img_bin = img_to_bin(img)
    img = bin_to_img(img_bin)
    img.show()
