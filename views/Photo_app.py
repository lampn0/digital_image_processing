import sys, os

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
# import Sobel

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi

from PyQt5.QtGui import QResizeEvent
import cv2 as cv
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image


def Sobel():
    # Here we read the image and bring it as an array
    filename = os.path.basename(fname[0])
    path = os.path.dirname(fname[0])
    name, type = os.path.splitext(filename)
    im = Image.open(fname[0])
    new_path = path + name + ".png"
    im.save(new_path)
    original_image = imread(new_path)

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

    #     self.browser.clicked.connect(self.browserfiles)
    #     self.khoangaunhien.clicked.connect(self.khoaNgauNhien)
    #     self.tuychinhkhoa.clicked.connect(self.tuyChinhKhoa)
    #     self.lammoikhoa.clicked.connect(self.lamMoiKhoa)
    #     self.mahoa.clicked.connect(self.encrypt_img)
    #     self.giaima.clicked.connect(self.decrypt_img)
    #     self.refresh.clicked.connect(self.refresh_bt)


class EditWindow(QDialog):
    def __init__(self):
        super(EditWindow, self).__init__()
        loadUi("edit_window.ui", self)
        self.show_img()
        self.btn_back.clicked.connect(self.back)
        self.Sobel.clicked.connect(Sobel)

    def back(self):
        back = MainWindow()
        widget.addWidget(back)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        back.show_image()

    def show_img(self):
        self.display.setPixmap(QPixmap(fname[0]).scaled(1041, 721, QtCore.Qt.KeepAspectRatio))
        self.label_name.setText(fname[0])

    #
    # def browserfiles(self):
    #     fname = QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*.png)')
    #     self.anhgoc.setPixmap(QPixmap(fname[0]))
    #     self.txtLink.setText(fname[0])
    #
    # def khoaNgauNhien(self):
    #     p = random_key()
    #     self.p.setText(str(p))
    #     e1, e2, d = key_gen(p)
    #     self.e1.setText(str(e1))
    #     self.e2.setText(str(e2))
    #     self.d.setText(str(d))
    #
    # def tuyChinhKhoa(self):
    #     p = self.p.toPlainText()
    #     p = int(p)
    #     check = check_prime_number(p)
    #     msg = QtWidgets.QMessageBox()
    #     msg.setWindowTitle("Kiểm tra số nguyên tố")
    #     if check:
    #         e1, e2, d = key_gen(p)
    #         self.e1.setText(str(e1))
    #         self.e2.setText(str(e2))
    #         self.d.setText(str(d))
    #         msg.setText("Bạn đã chọn p đúng ^-^")
    #         msg.setIcon(QtWidgets.QMessageBox.Information)
    #         msg.exec_()
    #     else:
    #         msg.setText("p không phải là số nguyên tố. Bạn hãy nhập lại !!!")
    #         msg.setIcon(QtWidgets.QMessageBox.Information)
    #         self.lamMoiKhoa()
    #         msg.exec_()
    #
    # def lamMoiKhoa(self):
    #     self.p.clear()
    #     self.e1.clear()
    #     self.e2.clear()
    #     self.d.clear()
    #     self.r.clear()
    #     self.c1.clear()
    #
    # def encrypt_img(self):
    #     p = int(self.p.toPlainText())
    #     e1 = int(self.e1.toPlainText())
    #     e2 = int(self.e2.toPlainText())
    #     link = self.txtLink.text()
    #     img = load_image(str(link))
    #     print("Load ảnh thành công")
    #     c1, r, encrypt = encryption(img, p, e1, e2)
    #     self.r.setText(str(r))
    #     self.c1.setText(str(c1))
    #     self.anhmahoa.setPixmap(
    #         QPixmap('D:/workspace/lampn182628/ly_thuyet_mat_ma/project/elgamal_image_encryption/encrypt.png'))
    #     msg = QtWidgets.QMessageBox()
    #     msg.setWindowTitle("Mã hóa")
    #     msg.setText("Mã hóa thành công")
    #     msg.setIcon(QtWidgets.QMessageBox.Information)
    #     msg.exec_()
    #
    # def decrypt_img(self):
    #     p = int(self.p.toPlainText())
    #     d = int(self.d.toPlainText())
    #     c1 = int(self.c1.toPlainText())
    #     decryption(c1, p, d)
    #     self.anhgiaima.setPixmap(
    #         QPixmap('D:/workspace/lampn182628/ly_thuyet_mat_ma/project/elgamal_image_encryption/decrypt.png'))
    #     msg = QtWidgets.QMessageBox()
    #     msg.setWindowTitle("Giải mã")
    #     msg.setText("Giải mã thành công")
    #     msg.setIcon(QtWidgets.QMessageBox.Information)
    #     msg.exec_()
    #
    # def refresh_bt(self):
    #     self.lamMoiKhoa()
    #     self.anhgoc.clear()
    #     self.anhmahoa.clear()
    #     self.anhgiaima.clear()



app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1366)
widget.setFixedHeight(768)
widget.show()
sys.exit(app.exec())
