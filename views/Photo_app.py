import sys, os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtGui import QResizeEvent

import cv2
import numpy as np
from matplotlib.image import imread
from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageDraw, ImageColor, ImageFilter
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
    kernel=np.zeros((kernel_size,kernel_size), np.float)
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

    abs_grad_x = cv2.convertScaleAbs(sobel_x)  # |src(I)∗alpha+beta|
    abs_grad_y = cv2.convertScaleAbs(sobel_y)

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


# from giaodienmahoa import Ui_Dialog
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("main_window.ui", self)
        self.actionOpen.triggered.connect(self.openFile)
        self.actionExit.triggered.connect(self.exit)
        self.btn_edit.clicked.connect(self.editImage)
        self.btn_rotate.clicked.connect(self.rotate)
        # self.btn_zin.clicked.connect(self.zoom_in)
        # self.btn_zout.clicked.connect(self.btn_zout)

    def openFile(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*.png *.jpg)')
        global filename
        filename = os.path.basename(fname[0])
        global path
        path = os.path.dirname(fname[0])
        global name, type
        name, type = os.path.splitext(filename)
        self.show_image()

    def show_image(self):
        self.display.setPixmap(QPixmap(fname[0]).scaled(1341, 601, QtCore.Qt.KeepAspectRatio))
        self.label_name.setText(fname[0])
        self.btn_edit.setEnabled(True)
        self.btn_del.setEnabled(True)
        self.btn_zin.setEnabled(True)
        self.btn_full.setEnabled(True)
        self.btn_zout.setEnabled(True)
        self.btn_rotate.setEnabled(True)

    def editImage(self):
        editimage = EditWindow()
        widget.addWidget(editimage)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def exit(self):
        widget.close()

    def rotate(self):
        Image.Image.rotate(90)

    def zoom_in(self):
        print()

    def zoom_out(self):
        print()


class Label(QLabel):
    def __init__(self):
        super(Label, self).__init__()
        self.pixmap_width: int = 1
        self.pixmapHeight: int = 1

    def setPixmapNew(self, pm: QPixmap) -> None:
        self.pixmap_width = pm.width()
        self.pixmapHeight = pm.height()

        self.updateMargins()
        super(Label, self).setPixmap(pm)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.updateMargins()
        super(Label, self).resizeEvent(a0)

    def updateMargins(self):
        if self.pixmap() is None:
            return
        pixmapWidth = self.pixmap().width()
        pixmapHeight = self.pixmap().height()
        if pixmapWidth <= 0 or pixmapHeight <= 0:
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        if w * pixmapHeight > h * pixmapWidth:
            m = int((w - (pixmapWidth * h / pixmapHeight)) / 2)
            self.setContentsMargins(m, 0, m, 0)
        else:
            m = int((h - (pixmapHeight * w / pixmapWidth)) / 2)
            self.setContentsMargins(0, m, 0, m)


def cacel():
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle("Cancel")
    msg.setText("Please Enter OK to Cancel!!!")
    msg.setIcon(QtWidgets.QMessageBox.Warning)
    msg.exec_()

class EditWindow(QDialog):
    def __init__(self):
        super(EditWindow, self).__init__()
        loadUi("edit_window.ui", self)
        self.show_img()
        self.btn_back.clicked.connect(self.back)
        self.Original.clicked.connect(self.original)
        self.Sobel.clicked.connect(self.sobel)
        self.Invert.clicked.connect(self.invert)
        self.exposure.valueChanged.connect(self.exposure_gfc)
        self.btn_cancel.clicked.connect(self.cancel)

    def back(self):
        back = MainWindow()
        widget.addWidget(back)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        back.show_image()

    def show_img(self):
        self.display.setPixmap(QPixmap(fname[0]).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))
        self.label_name.setText(fname[0])

    def cancel(self):
        cacel()
        self.back()

    def show_value(self):
        new_value = str(self.exposure.value())
        self.display.setText(new_value)

    def original(self):
        self.show_img()

    def invert(self):
        im = Image.open(fname[0])
        new_path = path + "/" + name + ".png"
        im.save(new_path)
        img = cv2.imread(new_path, cv2.IMREAD_UNCHANGED)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # plt.figure(figsize=(10, 10))
        # plt.subplot(121)
        # plt.imshow(img1, cmap='gray', norm=NoNorm())
        # plt.subplot(222)
        # plt.hist(img1.ravel(), 256, [0, 256])

        # Create zeros array to store the stretched image
        new_img = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                new_img[i, j] = 255 - img1[i, j]
        # plt.subplot(122)
        plt.imshow(new_img, cmap='gray', norm=NoNorm())
        plt.xticks([]), plt.yticks([])
        # plt.subplot(224)
        # plt.hist(new_img.ravel(), 256, [0, 256])
        # plt.show()

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def sobel(self):
        # Here we read the image and bring it as an array
        # im = Image.open(fname[0])
        # _path = path + name + ".png"
        # im.save(_path)
        # original_image = imread(_path)

        # Next we apply the Sobel filter in the x and y directions to then calculate the output image
        # dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
        # sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
        # sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step

        # Test bộ lọc Sobel
        # 1:img -> img_gray
        img_show = cv2.imread(fname[0])
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)

        # 2: img_gray-> img_gauss
        kernel_gauss = createGaussianKernel(3, 1)
        img = convolute(img_show, kernel_gauss)

        # img_gauss -> sobelx
        SobelFilterX = SobelFilter(img, 5, 1, 0)
        # img_gauss -> sobely
        SobelFilterY = SobelFilter(img, 5, 0, 1)
        # img_gauss -> sobelxy
        SobelFilterXY = SobelFilter(img, 5, 1, 1)

        # img_gauss -> sobelx use openCV
        Sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        # img_gauss -> sobely use openCV
        Sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        # img_gauss -> sobelxy use openCV
        Sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1)

        plt.imshow(SobelFilterXY, cmap='gray', vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def exposure_gfc(self):
        # Read the image
        img = cv2.imread(fname[0], cv2.IMREAD_UNCHANGED)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        k = self.exposure.value()
        # Create zeros array to store the stretched image
        new_img = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                v = img1[i, j] + k
                if v > 255:
                    new_img[i, j] = 255
                elif v < 0:
                    new_img[i, j] = 0
                else:
                    new_img[i, j] = v

        plt.imshow(new_img, cmap='gray', norm=NoNorm())
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))


app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1366)
widget.setFixedHeight(768)
widget.show()
sys.exit(app.exec())
