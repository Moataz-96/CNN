import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from copy import deepcopy
from threading import Thread
from PyQt5 import QtWidgets, QtCore, QtGui
import time
from random import randint
import cv2
from matplotlib import pyplot as plt

rand = ""
class gui(QDialog):
    def __init__(self):
        super(gui, self).__init__()
        self.images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg",
                       "9.jpg", "10.jpg", "11.jpg"]
        loadUi('Lab1.ui',self)
        self.startlineedit.clicked.connect(self.startThread)
        self.loadimagelineedit.clicked.connect(self.loadRandomImage)

    def loadRandomImage(self):
        self.statuslineedit.setText('')
        global rand
        rand = randint(0, 10)
        self.loadimagelineedit.setEnabled(False)
        self.imagelineedit.setPixmap(QtGui.QPixmap(self.images[rand]))
        if rand == 0 or rand == 10:
            self.imagelineedit.setScaledContents(False)
        else:
            self.imagelineedit.setScaledContents(True)
        time.sleep(0.5)
        self.loadimagelineedit.setEnabled(True)

    def setImage(self, image,type):
        if type == 'grayscale':
            qformat = QtGui.QImage.Format_Grayscale8
        elif type == 'rgb':
            qformat = QtGui.QImage.Format_RGB888
        elif type == 'bgr':
            qformat = QtGui.QImage.Format_BGR30
        elif type == 'bw':
            qformat = QtGui.QImage.Format_Indexed8
        newImage = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             qformat).rgbSwapped()
        self.imagelineedit.setPixmap(QtGui.QPixmap.fromImage(newImage))

    def grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def medianBlur(self, image):
        blur = cv2.medianBlur(image, 7)
        return blur

    def laplacianFilter(self, image):
        laplacian = cv2.Laplacian(image, ddepth=-1, ksize=5, borderType=cv2.BORDER_DEFAULT)
        return laplacian

    def threshold(self, image):
        coloredImage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        ret, thresh = cv2.threshold(coloredImage,125, 255, type=cv2.THRESH_BINARY_INV)
        return thresh
    
    #def cartonize(self,image):
        
    def bilateral(self,image, thresholdImage):
        bilateral = cv2.bilateralFilter(image,9 , 9,7)
        for _  in range(6):
            bilateral = cv2.bilateralFilter(bilateral,9 , 9,7)
        cartoon = cv2.bitwise_and(bilateral, thresholdImage)   
        return cartoon

    def setStatus(self, text):
        font = QFont("Verdana", 12)
        self.statuslineedit.setText(text)
        self.statuslineedit.setFont(font)

    def startThread(self):
        self.start_thread = Thread(target=self.start)
        self.start_thread.start()

    def start(self):
        global rand
        image = cv2.imread(self.images[rand])

        grayscaledImage = self.grayscale(image)
        self.setImage(grayscaledImage,'grayscale')
        self.setStatus('Converting to gray scale')

        time.sleep(1)

        blurredImage = self.medianBlur(grayscaledImage)
        self.setImage(blurredImage,'grayscale')
        self.setStatus('Smoothing the gray-scaled image')

        time.sleep(1)

        laplacianImage = self.laplacianFilter(blurredImage)
        self.setImage(laplacianImage,'bw')
        self.setStatus('Applying Laplacian filter')

        time.sleep(1)

        thresholdImage = self.threshold(laplacianImage)
        self.setImage(thresholdImage, 'rgb')
        self.setStatus('Applying threshold to the image')

        time.sleep(1)

        bilateralImage = self.bilateral(image,thresholdImage)
        self.setImage(bilateralImage, 'rgb')
        self.setStatus('Applying bilateral filter to the image')

        time.sleep(1)










def main():
    main_app = QApplication(sys.argv)
    window = gui()
    window.setWindowTitle('Image Cartoonifier')
    window.show()
    sys.exit(main_app.exec_())
if __name__ == '__main__':
    main()
