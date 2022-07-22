import cv2
import numpy as np



def histogram_equalization(img_in):
    # Write histogram equalization here

    # Histogram equalization result

    blue, green, red = cv2.split(img_in)  #split the image into r,g,b channels
    hist_blue = cv2.calcHist(
        [blue], [0], None, [256],
        [0, 256])  #calculating the histogram and CDF for each histogram
    cdf_blue = np.cumsum(hist_blue)

    hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
    cdf_green = np.cumsum(hist_green)

    hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])
    cdf_red = np.cumsum(hist_red)

    blue1 = np.around(np.subtract(cdf_blue, np.amin(cdf_blue)))
    cv2.divide(blue1, blue.size, blue1)
    cv2.multiply(blue1, 255, blue1)

    green1 = np.around(np.subtract(cdf_green, np.amin(cdf_green)))
    cv2.divide(green1, green.size, green1)
    cv2.multiply(green1, 255, green1)

    red1 = np.around(np.subtract(cdf_red, np.amin(cdf_red)))
    cv2.divide(red1, red.size, red1)
    cv2.multiply(red1, 255, red1)

    new_blue = blue1[blue.ravel()].reshape(blue.shape)
    new_green = green1[green.ravel()].reshape(green.shape)
    new_red = red1[red.ravel()].reshape(red.shape)

    img = cv2.merge([new_blue, new_green, new_red])  #Merging all channels
    img_out = img
    return True, img_out


# Read in input images
input_image = cv2.imread("../images/2021-03-14_orig.jpg", cv2.IMREAD_COLOR)

# Histogram equalization
succeed, output_image = histogram_equalization(input_image)

# Write out the result
output_name = "1.jpg"
cv2.imwrite(output_name, output_image)
