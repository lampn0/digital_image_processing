import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Filter:
    def Gaussian(self):
        # Load and blur image
        img = cv.imread('25.png')
        img2 = cv.imread('25.png')
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
        img = cv.imread('25.png')
        img2 = cv.imread('25.png')
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
        img = cv.imread('25.png')
        img2 = cv.imread('25.png')
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

ans=True
test = Filter()
while ans:
    print("""
    1.Bo loc Gaussian
    2.Bo loc Trung Vi
    3.Bo loc Trung Binh
    4.Bo loc Bilateral
    5.Exit/Quit
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
        print("Bye")
        ans = None
    else:
        print("\n Not Valid Choice Try again")

