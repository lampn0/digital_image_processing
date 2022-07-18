import sys
import cv2
import numpy as np


def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha * img + beta, dtype=int)  # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new


if __name__ == "__main__":
    alpha = 1.0          #raise
    beta = 35
    # alpha = 0.5        #reduce
    # beta = 10
    if len(sys.argv) == 3:
        alpha = float(sys.argv[1])
        beta = int(sys.argv[2])
    img = cv2.imread('../images/2021-03-14_orig.jpg')  # [height, width, channel]

    # change image brightness g(x,y) = alpha*f(x,y) + beta
    img_new = change_brightness(img, alpha, beta)

    cv2.imwrite('../results/2021-03-14_orig.jpg', img_new)