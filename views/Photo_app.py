import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi

from PyQt5.QtGui import QResizeEvent

import pickle, argparse, re, sys
import array
import random
import cv2 as cv
import numpy



# from giaodienmahoa import Ui_Dialog
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("main_window.ui", self)
        self.actionOpen.triggered.connect(self.openFile)
        self.btn_edit.clicked.connect(self.editImage)

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*.png *.jpg)')
        # self.display.setPixmap(QPixmap(fname[0]))
        # self.label_name.setText(fname[0])
        # self.display.setPixmap(QPixmap(fname[0]))
        self.label_name.setText(fname[0])
        self.display = Label()
        self.display.setPixmap(QPixmap(fname[0]))

    def editImage(self):
        editimage = EditWindow()
        widget.addWidget(editimage)
        widget.setCurrentIndex(widget.currentIndex()+1)

class Label(QLabel):
    def __init__(self):
        super(Label, self).__init__()
        self.pixmap_width: int = 1
        self.pixmapHeight: int = 1

    def setPixmap(self, pm: QPixmap) -> None:
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
        self.btn_back.clicked.connect(self.back)

    def back(self):
        back = MainWindow()
        widget.addWidget(back)
        widget.setCurrentIndex(widget.currentIndex() + 1)

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
