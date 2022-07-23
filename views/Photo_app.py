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
from scipy.interpolate import UnivariateSpline
from PIL import Image, ImageEnhance, ImageDraw, ImageColor, ImageFilter
import math


def _create_LUT_8UC1(x, y):
  spl = UnivariateSpline(x, y)
  return spl(range(256))

class WarmingFilter:
    def __init__(self):
        x = [0, 64, 128, 192, 256]
        y1 = [0, 70, 140, 210, 256]
        y2 = [0, 30,  80, 120, 192]
        self.incr_ch_lut = _create_LUT_8UC1(x, y1)
        self.decr_ch_lut = _create_LUT_8UC1(x, y2)

    def render(self, img_rgb):
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb,
                                               cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)),
                            cv2.COLOR_HSV2RGB)


class CoolingFilter:
    def __init__(self):
        self.incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
                                            [0, 70, 140, 210, 256])
        self.decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
                                            [0, 30,  80, 120, 192])
    def render(self, img_rgb):
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

def convolution(image, kernel):
    # Lật ma trận mặt nạ
    # lấy kích thước ma trận mặt nạ
    r, c = kernel.shape

    # Tạo mảng 2 chiều toàn 0 có kích thước rxc
    flip_kernel = np.zeros((r, c), np.float32)
    h = 0

    # Kích thước ma trận lẻ trả về True
    if (r == c) and (r % 2):
        h = int((r - 1) / 2)  # h=1
        for i in range(-h, h + 1):  # i -1 -> 1
            for j in range(-h, h + 1):  # j -1 -> 1
                flip_kernel[i + h, j + h] = kernel[-i + h, -j + h]

    # Thêm padding cho ảnh đầu vào
    m, n = image.shape
    padding_image = np.zeros((m + 2 * h, n + 2 * h), np.float32)
    padding_image[h:-h, h:-h] = image

    # Ảnh sau khi tích chập
    convolute_image = np.zeros((m, n), np.float32)
    for i in range(m):
        for j in range(n):
            convolute_image[i, j] = (flip_kernel * padding_image[i: i + r, j: j + r]).sum()
    return convolute_image


def createGaussianKernel(kernel_size, sigma):
    h = kernel_size // 2  # h = int(kernel_size/2)
    s = 2.0 * sigma * sigma
    # sum = 0
    kernel = np.zeros((kernel_size, kernel_size), np.float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            r = np.sqrt(np.square(i - h) + np.square(j - h))
            kernel[i, j] = (np.exp(-(r * r) / s)) / (math.pi * s)
    kernel = kernel / kernel.sum()
    return kernel


def SobelFilter(image, dx=1, dy=1):
    X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    sobel_x = convolution(image, X)
    sobel_y = convolution(image, Y)

    if dx == 1 and dy == 0:
        return sobel_x
    if dx == 0 and dy == 1:
        return sobel_y

    abs_grad_x = cv2.convertScaleAbs(sobel_x)  # |src(I)∗alpha+beta|
    abs_grad_y = cv2.convertScaleAbs(sobel_y)

    sobel_xy = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel_xy


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


def controller(img, brightness=255, contrast=127):
    brightness = int(brightness)
    contrast = int(contrast)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        print(Alpha, Gamma)
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
    return cal


class EditWindow(QDialog):
    def __init__(self):
        super(EditWindow, self).__init__()
        loadUi("edit_window.ui", self)
        self.show_img(fname[0])
        global img_edit
        img_edit = cv2.imread(fname[0])
        self.btn_back.clicked.connect(self.back)
        self.Original.clicked.connect(self.original)
        self.Sobel.clicked.connect(self.sobel)
        self.Invert.clicked.connect(self.invert)
        self.contrast.valueChanged.connect(self.Contrast)
        self.exposure.valueChanged.connect(self.Exposure)
        self.highlights.valueChanged.connect(self.Highlights)
        self.shadows.valueChanged.connect(self.Shadows)
        self.tint.valueChanged.connect(self.Tint)
        self.warmth.valueChanged.connect(self.Warmth)
        self.clarity.valueChanged.connect(self.Clarity)
        self.vignette.valueChanged.connect(self.Vignette)
        self.btn_cancel.clicked.connect(self.cancel)
        self.Cool.clicked.connect(self.cool)
        self.Warm.clicked.connect(self.warm)

    def back(self):
        back = MainWindow()
        widget.addWidget(back)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        back.show_image()

    def show_img(self, img_path):
        self.display.setPixmap(QPixmap(img_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))
        self.label_name.setText(fname[0])

    def cancel(self):
        cacel()
        self.back()

    def original(self):
        self.show_img(fname[0])

    def invert(self):
        im = Image.open(fname[0])
        new_path = path + "/" + name + ".png"
        im.save(new_path)
        img = cv2.imread(new_path, cv2.IMREAD_UNCHANGED)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create zeros array to store the stretched image
        new_img = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                new_img[i, j] = 255 - img1[i, j]

        plt.imshow(new_img, cmap='gray', norm=NoNorm())
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def sobel(self):
        # Here we read the image and bring it as an array
        im = Image.open(fname[0])
        _path = path + "/" + name + ".png"
        im.save(_path)
        original_image = imread(_path)

        # Next we apply the Sobel filter in the x and y directions to then calculate the output image
        dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
        sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
        sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step

        # Test bộ lọc Sobel
        # 1:img -> img_gray
        img_show = cv2.imread(fname[0])
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)

        # 2: img_gray-> img_gauss
        kernel_gauss = createGaussianKernel(3, 1)
        img = convolution(img_show, kernel_gauss)

        # img_gauss -> sobelx
        SobelFilterX = SobelFilter(img, 1, 0)
        # img_gauss -> sobely
        SobelFilterY = SobelFilter(img, 0, 1)
        # img_gauss -> sobelxy
        SobelFilterXY = SobelFilter(img, 1, 1)

        # img_gauss -> sobelx use openCV
        Sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        # img_gauss -> sobely use openCV
        Sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        # img_gauss -> sobelxy use openCV
        Sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1)


        plt.subplot(121), plt.imshow(sobel_filtered_image), plt.title('Thư viện có sẵn')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(SobelFilterXY, cmap='gray', vmin=0, vmax=255), plt.title('Tự tạo hàm')
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

    def warm(self):
        img_result = WarmingFilter()
        img_result = img_result.render(img_edit)
        plt.imshow(img_result, norm=NoNorm())
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def cool(self):
        img_result = CoolingFilter()
        img_result = img_result.render(img_edit)
        plt.imshow(img_result, norm=NoNorm())
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def Contrast(self):
        value = str(self.contrast.value())
        self.Contrast_value.setText(value)
        effect = controller(img_edit, self.exposure.value(), self.contrast.value())
        cv2.imshow('Effect', effect)
        # plt.imshow(effect, norm=NoNorm())
        # plt.xticks([]), plt.yticks([])
        #
        # new_path = path + "/" + name + "_tmp.png"
        # plt.savefig(new_path)
        # self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def Exposure(self):
        value = str(self.exposure.value())
        self.Exposure_value.setText(value)
        effect = controller(img_edit, self.exposure.value(), self.contrast.value())
        cv2.imshow('Effect', effect)

    def Highlights(self):
        value = str(self.highlights.value())
        self.Highlights_value.setText(value)

    def Shadows(self):
        value = str(self.shadows.value())
        self.Shadows_value.setText(value)

    def Tint(self):
        value = str(self.tint.value())
        self.Tint_value.setText(value)

    def Warmth(self):
        value = str(self.warmth.value())
        self.Warmth_value.setText(value)

    def Clarity(self):
        value = str(self.clarity.value())
        self.Clarity_value.setText(value)

    def Vignette(self):
        value = str(self.vignette.value())
        self.Vignette_value.setText(value)


app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1366)
widget.setFixedHeight(768)
widget.show()
sys.exit(app.exec())
