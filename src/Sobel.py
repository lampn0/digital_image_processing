import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import array as arr
import math


def convolute(image, kernel):
    #lật ma trận kernel
    r,c = kernel.shape
    flip_kernel = np.zeros((r,c), np.float32)
    h = 0
    if ((r==c) and (r%2)):
        h = int((r-1)/2)   #h=1
        for i in range(-h, h+1):    #i -1 -> 1
            for j in range(-h, h+1):   #j -1 -> 1
                flip_kernel[i+h, j+h] = kernel[-i+h, -j+h]

    #Thêm padding cho ảnh đầu vào
    m, n = image.shape
    padding_image = np.zeros((m+2*h, n+2*h), np.float32)
    padding_image[h:-h, h:-h] = image

    #Ảnh sau khi tích chập
    convolute_image = np.zeros((m, n), np.float32)
    for i in range(m):
        for j in range(n):
            convolute_image[i, j] = (flip_kernel * padding_image[i: i+r, j: j+r]).sum()
    return convolute_image

def createGaussianKernel(kernel_size,sigma):
    h=kernel_size//2     #h = int(kernel_size/2)
    s = 2.0 * sigma * sigma
    # sum = 0
    kernel=np.zeros((kernel_size,kernel_size),np.float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            r = np.sqrt(np.square(i-h) +np.square(j-h))
            kernel[i,j] = (np.exp(-(r * r) / s)) / (math.pi * s)
    kernel = kernel/kernel.sum()
    return kernel


def SobelFilter(image, gauss_ksize=5, dx=1, dy=1, threshold=60):
    # GaussKernel = createGaussianKernel(gauss_ksize,1)
    X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # kernel_x = convolute(GaussKernel,X)
    sobel_x = convolute(image, X)
    # kernel_y = convolute(GaussKernel,Y)
    sobel_y = convolute(image, Y)

    if dx == 1 and dy == 0:
        return sobel_x
    if dx == 0 and dy == 1:
        return sobel_y

    # abs_sobel_x = np.absolute(sobel_x)
    # img_sobel_x = np.uint8(abs_sobel_x)
    # abs_sobel_y = np.absolute(sobel_y)
    # img_sobel_y = np.uint8(abs_sobel_y)

    # sobelxy = img_sobel_x + img_sobel_y

    # sobelxy = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    abs_grad_x = cv.convertScaleAbs(sobel_x)  # |src(I)∗alpha+beta|
    abs_grad_y = cv.convertScaleAbs(sobel_y)

    # sobel_xy = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    alpha = 0.5
    beta = 0.5
    gamma = 0
    sobel_xy = abs_grad_x * alpha + abs_grad_y * beta + gamma

    # sobel = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    # [src1, alpha, src2, beta, gamma[, dst[, dtype]]
    # dst = src1*alpha + src2*beta + gamma
    # void cv::addWeighted	(	InputArray 	src1, double 	alpha, InputArray 	src2, double 	beta,double 	gamma,OutputArray 	dst,int 	dtype = -1)

    # sobel = sobel_x + sobel_y
    # sobel = image + sobel

    # Loại bỏ những pixel yếu để tăng độ sắc nét của cạnh
    # sobelxy = np.float64(sobel_xy)
    # for i in range(sobelxy.shape[0]):
    #     for j in range(sobelxy.shape[1]):
    #         if sobelxy[i][j] < threshold:
    #             sobelxy[i][j] = 0
    #         else:
    #             sobelxy[i][j] = 255
    return sobel_xy

# Test bộ lọc Sobel
#1:img -> img_gray
img1 = cv.imread("../images/2021-03-14_orig.jpg")
img_show = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#2: img_gray-> img_gauss
kernel_gauss = createGaussianKernel(3,1)
img = convolute(img_show,kernel_gauss)

#img_gauss -> sobelx
SobelFilterX = SobelFilter(img,5,1,0)
#img_gauss -> sobely
SobelFilterY = SobelFilter(img,5,0,1)
#img_gauss -> sobelxy
SobelFilterXY = SobelFilter(img,5,1,1)

#img_gauss -> sobelx use openCV
Sobelx = cv.Sobel(img,cv.CV_64F,1,0)
#img_gauss -> sobely use openCV
Sobely = cv.Sobel(img,cv.CV_64F,0,1)
#img_gauss -> sobelxy use openCV
Sobelxy = cv.Sobel(img,cv.CV_64F,1,1)

plt.figure(figsize=(30,30))
plt.subplot(421), plt.title('Original'), plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(423), plt.title('Sobel x'), plt.imshow(Sobelx,cmap='gray',vmin=0,vmax=255)
plt.subplot(424), plt.title('SobelFilterX'), plt.imshow(SobelFilterX,cmap='gray',vmin=0,vmax=255)
plt.subplot(425), plt.title('Sobel y'), plt.imshow(Sobely,cmap='gray',vmin=0,vmax=255)
plt.subplot(426), plt.title('SobelFilterY'), plt.imshow(SobelFilterY,cmap='gray',vmin=0,vmax=255)
plt.subplot(427), plt.title('Sobel xy'), plt.imshow(Sobelxy,cmap='gray',vmin=0,vmax=255)
plt.subplot(428), plt.title('SobelFilterXY'), plt.imshow(SobelFilterXY,cmap='gray',vmin=0,vmax=255)
plt.show()