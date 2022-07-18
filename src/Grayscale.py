from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

input_image = imread("../images/2021-03-14_orig.jpg")

r = input_image[:,:,0]
g = input_image[:,:,1]
b = input_image[:,:,2]

# gamma = 1.04
gamma = 1

r_const, g_const, b_const = 0.2126, 0.7152, 0.0722

grayscale_image = r_const * r ** gamma + g_const * g ** gamma + b_const * b ** gamma

fig = plt.figure(1)

img1, img2 = fig.add_subplot(121), fig.add_subplot(122)

img1.imshow(input_image)
img2.imshow(grayscale_image, cmap = plt.cm.get_cmap('hot'))

# fig.show()
plt.show()