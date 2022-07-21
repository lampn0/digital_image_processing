import sys, os

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
# import Sobel

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


def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha * img + beta, dtype=int)  # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new


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
        img = cv2.imread(fname[0], cv2.IMREAD_UNCHANGED)
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
        im = Image.open(fname[0])
        _path = path + name + ".png"
        im.save(_path)
        original_image = imread(_path)

        # Next we apply the Sobel filter in the x and y directions to then calculate the output image
        dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
        sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
        sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step

        # plt.subplot(121), plt.imshow(original_image)
        # plt.imshow
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(122),
        plt.imshow(sobel_filtered_image)
        plt.xticks([]), plt.yticks([])

        new_path = path + "/" + name + "_tmp.png"
        plt.savefig(new_path)
        self.display.setPixmap(QPixmap(new_path).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))

    def exposure_gfc(self):
        # Read the image
        img = cv2.imread(fname[0], cv2.IMREAD_UNCHANGED)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # plt.figure(figsize=(10, 10))
        # plt.subplot(121)
        # plt.imshow(img1, cmap='gray', norm=NoNorm())
        # plt.subplot(222)
        # plt.hist(img1.ravel(), 256, [0, 256])

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

        # plt.subplot(122)
        plt.imshow(new_img, cmap='gray', norm=NoNorm())
        plt.xticks([]), plt.yticks([])
        # plt.subplot(224)
        # plt.hist(new_img.ravel(), 256, [0, 256])

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
