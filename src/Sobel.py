import cv2 as cv
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import time
import llf


class Sobel:
    def _Sobel(self):
        # Here we read the image and bring it as an array

        im = Image.open('../images/shape.jpg')
        im.save('../images/shape.png')
        original_image = imread("../images/shape.png")

        # Next we apply the Sobel filter in the x and y directions to then calculate the output image
        dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
        sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
        sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step

        plt.subplot(121), plt.imshow(original_image), plt.title('Original')
        plt.imshow
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(sobel_filtered_image), plt.title('Sobel Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def _sobel(self):
        print()

