import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import time
import llf


class Filter:
    def Gaussian(self):
        # Load and blur image
        img = cv.imread("../images/2021-03-14_orig.jpg")
        img2 = cv.imread("../images/2021-03-14_orig.jpg")
        blur = cv.GaussianBlur(img, (5, 5), 0)
        blur2 = cv.GaussianBlur(img2, (5, 5), 0)

        # Convert color from bgr (OpenCV default) to rgb
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        blur_rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
        img_rgb2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        blur_rgb2 = cv.cvtColor(blur2, cv.COLOR_BGR2RGB)

        # Display
        plt.subplot(221), plt.imshow(img_rgb), plt.title('Gauss Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(blur_rgb), plt.title('Gauss Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(img_rgb2), plt.title('Salt&Pepper Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(blur_rgb2), plt.title('Salt&Pepper Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
    def TrungVi(self):
        # Load and blur image
        img = cv.imread('25.png')
        img2 = cv.imread('25.png')
        blur = cv.medianBlur(img, 5)
        blur2 = cv.medianBlur(img2, 5)

        # Convert color from bgr (OpenCV default) to rgb
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        blur_rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
        img_rgb2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        blur_rgb2 = cv.cvtColor(blur2, cv.COLOR_BGR2RGB)

        # Display
        plt.subplot(221), plt.imshow(img_rgb), plt.title('Gauss Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(blur_rgb), plt.title('Gauss Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(img_rgb2), plt.title('Salt&Pepper Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(blur_rgb2), plt.title('Salt&Pepper Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
    def TrungBinh(self):
        # Load and blur image
        img = cv.imread("../images/2021-03-14_orig.jpg")
        img2 = cv.imread("../images/2021-03-14_orig.jpg")
        blur = cv.blur(img, (5, 5))
        blur2 = cv.blur(img2, (5, 5))

        # Convert color from bgr (OpenCV default) to rgb
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        blur_rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
        img_rgb2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        blur_rgb2 = cv.cvtColor(blur2, cv.COLOR_BGR2RGB)

        # Display
        plt.subplot(221), plt.imshow(img_rgb), plt.title('Gauss Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(blur_rgb), plt.title('Gauss Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(img_rgb2), plt.title('Salt&Pepper Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(blur_rgb2), plt.title('Salt&Pepper Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
    def Bilateral(self):
        # Load and blur image
        img = cv.imread("../images/2021-03-14_orig.jpg")
        img2 = cv.imread("../images/2021-03-14_orig.jpg")
        blur = cv.bilateralFilter(img, 9, 75, 75)
        blur2 = cv.bilateralFilter(img, 9, 75, 75)

        # Convert color from bgr (OpenCV default) to rgb
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        blur_rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
        img_rgb2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        blur_rgb2 = cv.cvtColor(blur2, cv.COLOR_BGR2RGB)

        # Display
        plt.subplot(221), plt.imshow(img_rgb), plt.title('Gauss Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(blur_rgb), plt.title('Gauss Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(img_rgb2), plt.title('Salt&Pepper Noise')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(blur_rgb2), plt.title('Salt&Pepper Noise - Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Sobel(self):
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



    def LaplacianFilter(self):
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)

        # im1 = cv.imread('../images/LennaBL.tif')
        # im2 = cv.imread('../images/LennaBW.tif')
        im1 = cv.imread('../images/2021-03-14_blur.jpg')
        # im2 = cv.imread('../images/2021-03-14_dark.jpg')
        I1 = np.float32(cv.cvtColor(im1, cv.COLOR_BGR2GRAY) / 255)
        # I2 = np.float32(cv.cvtColor(im2, cv.COLOR_BGR2GRAY) / 255)

        sigma = 0.1
        N = 4
        fact = -0.75
        t = time.time()

        # Filtering
        print(I1.shape)
        # print(I2.shape)
        # im_e = llf.xllf(I1, I2, sigma, fact, N)
        im_e = llf.xllf(I1, sigma, fact, N)
        im_ergb = llf.repeat(im_e)
        elapsed = time.time() - t
        print(elapsed)

        # plot the image
        # im_con = np.concatenate((im1 / 255, im2 / 255, im_ergb), axis=1)
        im_con = np.concatenate((im1 / 255, im_ergb), axis=1)
        imgplt = plt.imshow(im_con)
        plt.show()

ans=True
test = Filter()

while ans:
    print("""
    1.Bo loc Gaussian
    2.Bo loc Trung Vi
    3.Bo loc Trung Binh
    4.Bo loc Bilateral
    5.Bo loc Sobel
    6.Bo loc Laplacian
    7.Exit/Quit
    """)
    ans= input("Input your choice: ")
    if ans=="1":
        test.Gaussian()
    elif ans=="2":
        test.TrungVi()
    elif ans=="3":
        test.TrungBinh()
    elif ans=="4":
        test.Bilateral()
    elif ans=="5":
        test.Sobel()
    elif ans=="6":
        test.LaplacianFilter()
    elif ans=="7":
        print("Exit!")
        ans = None
    else:
        print("\n Not Valid Choice Try again")


