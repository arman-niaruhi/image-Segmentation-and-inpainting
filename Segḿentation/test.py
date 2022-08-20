from PySide2.QtWidgets import QLabel
from PySide2 import QtGui, QtCore
from PySide2.QtGui import  QPainter
from PySide2.QtCore import Qt
from PySide2.QtCore import Qt, QMimeData
import os
from PySide2.QtGui import  QPainter, QPixmap
from PySide2.QtCore import Qt,QSize,QRect
from PySide2.QtWidgets import QApplication,QLineEdit,QCheckBox, QComboBox ,QMainWindow ,QPushButton, QFileDialog ,QWidget ,QHBoxLayout,QVBoxLayout , QGroupBox , QToolBar, QStatusBar, QGroupBox
from PySide2.QtWidgets import QLabel
from PySide2 import QtGui, QtCore
from PySide2.QtGui import  QPainter
from PySide2.QtCore import Qt, QMimeData
from PySide2.QtGui import QDrag

class Merge(QLabel):
    def __init__(self,path):
        super().__init__()
        self.path = path
        self.width, self.height = 700, 500
        self.setFixedSize(self.width, self.height)
        self.image_small = None
        self.widthImage , self.heightImage = self.width//3 , self.height//3
        self.button = None
        self.initUI()
        
    def initUI(self):
        self.setAcceptDrops(True)
        folder_pic = QPixmap(self.path)
        folder_size = folder_pic.scaled(self.width, self.height)
        self.label = QLabel(self)
        self.label.setPixmap(folder_size)
          

    def merge_completly(self, img):
        self.button = Label()
        print (img)
            

    def dragEnterEvent(self, e):
        if(self.button is not None):
            e.accept()

    def dropEvent(self, e):
        if(self.button is not None):
            position = e.pos()
            self.button.move(position)
            e.setDropAction(Qt.MoveAction)
            e.accept()
        
    def mouseMoveEvent(self, e):

        if e.buttons() != Qt.LeftButton:
            return

        mimeData = QMimeData(self)

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(e.pos() - self.rect().topLeft())

        dropAction = drag.start(Qt.MoveAction)
    
class Label(QLabel):
    def __init__(self):
        super(Label, self).__init__()
        # self.img = img 
        self.setup()


    def setup(self):
        folder_pic = QPixmap("pexels-pascal-renet-1089306.jpg")
        folder_size = folder_pic.scaled(300, 250)
        self.label = QLabel(self)
        self.label.setPixmap(folder_size)
