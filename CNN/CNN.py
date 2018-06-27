import numpy
import matplotlib
from PIL import Image
import opencv


def loadImage(imagepath):
    img = Image.open(r"C:\Users\wenp\Pictures\\" + imagepath)
    img_matrix = numpy.array(img)
    return img_matrix


def img_process(img_matrix, process_matrix):
