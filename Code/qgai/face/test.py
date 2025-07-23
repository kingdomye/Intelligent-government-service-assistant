import io
import base64
from PIL import Image
import numpy as np

img = Image.open("./facedata/12_1075.jpg")
img = np.array(img)
print(img)
