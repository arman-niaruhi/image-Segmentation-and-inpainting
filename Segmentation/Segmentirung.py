from PySide2.QtWidgets import QApplication,QLineEdit,QCheckBox, QComboBox ,QMainWindow ,QPushButton, QFileDialog ,QWidget, QHBoxLayout,QVBoxLayout , QGroupBox , QToolBar, QStatusBar, QGroupBox
import sys
from PySide2.QtCore import Qt
from PySide2 import QtGui, QtCore
from Scrible import Scrible 
from Merge import Merge 
from PIL import Image


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: grey;")
        self.lbl2 = None
        self.lbl1 = None
        self.fieldEdit = None
        self.create_UI()                    #Set Window Size
        self.create_Component()
        self.theme = "grey"
        self.css()

    def create_Component(self):

        # Toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)



        #combiBox to select Fore- and Background 
        self.themesel = QComboBox()
        self.themesel.addItems(['grey', 'red', 'white','blue', 'green'])
        self.themesel.setStatusTip("Change the color of window!")
        self.toolbar.addWidget(self.themesel)
        qbtnt = QPushButton("Theme change",self)
        qbtnt.clicked.connect(self.themechange)
        self.toolbar.addWidget(qbtnt)


        #saveImage
        saveimg = QPushButton("Save",self)
        saveimg.clicked.connect(self.saveImage)
        self.toolbar.addWidget(saveimg)

        #combiBox to select Fore- and Background 
        self.combibox_draw = QComboBox()
        self.combibox_draw.addItems(['Foreground(Red)', 'Background(Blue)'])
        self.combibox_draw.setStatusTip("Please click on Draw to apply!")
        self.toolbar.addWidget(self.combibox_draw)

        #select the size of the draw pan
        self.fieldEdit = QLineEdit(self)
        self.fieldEdit.setFixedWidth(45)
        self.fieldEdit.textChanged.connect(self.onChangeSize)
        self.toolbar.addWidget(self.fieldEdit)

        #Draw button to apply selection of Fore- and Background 
        qbtn = QPushButton("Draw",self)
        qbtn.clicked.connect(self.farbe)
        self.toolbar.addWidget(qbtn)

        #Eraser Button
        erase_btn = QCheckBox("Erase", self)
        erase_btn.setMaximumSize(60,30)
        erase_btn.stateChanged.connect(self.erase)
        self.toolbar.addWidget(erase_btn)

        #select the size of the eraser
        self.fieldEdit1 = QLineEdit(self)
        self.fieldEdit1.setFixedWidth(45)
        self.fieldEdit1.textChanged.connect(self.onClickErase)
        self.toolbar.addWidget(self.fieldEdit1)

        #Cut and transport Button
        cutBtn = QPushButton("Autom. Fill Mask",self)
        cutBtn.clicked.connect(self.transfer)
        self.toolbar.addWidget(cutBtn)
        merge_btn = QPushButton("Merge",self)
        merge_btn.clicked.connect(self.merge_img)
        self.toolbar.addWidget(merge_btn)
      

        #labels
        self.path1, _ = QFileDialog.getOpenFileName(self, 'Insert image',".","Images (*.png *.jpg *.jpeg *bmp)") 
        self.lbl1 = Scrible(self.path1)
        self.setStyleSheet("background-color: grey;")
        self.path2, _ = QFileDialog.getOpenFileName(self, 'Insert image',".","Images (*.png *.jpg *.jpeg *bmp)")

        # self.lbl1.transfer() 
        self.lbl2 = Merge(self.path2)

        #Layout
        self.central_widget = QWidget()
        groupBox = QGroupBox("Selected Pictures")
        self.HBOX = QHBoxLayout()
        self.vbox = QVBoxLayout()
        self.HBOX.addWidget(self.lbl1)
        self.HBOX.addWidget(self.lbl2)
        groupBox.setLayout(self.HBOX)
        self.vbox.addWidget(groupBox)
        self.central_widget.setLayout(self.vbox)
        self.setCentralWidget(self.central_widget)
    
    def css(self):
        a = self.theme
        self.setStyleSheet(
            f"background-color: {a};"
            """
            QLabel{
                border-style: solid;
                border-color: rgba(55, 55, 55, 255);
                border-width: 2px;
            }
            """)

    def farbe(self):
        if (self.combibox_draw.currentText() == "Foreground(Red)"):
            self.lbl1.brushColor = QtGui.QColor(QtCore.Qt.red)
        elif (self.combibox_draw.currentText() == "Background(Blue)"):
            self.lbl1.brushColor = QtGui.QColor(QtCore.Qt.blue)

    def themechange(self):
        self.theme = self.themesel.currentText()
        self.css()

    def erase(self,state):
        if state is not Qt.Checked:
            self.lbl1.changeColour()
    
    def onChangeSize(self):
        self.lbl1.brushSize = int(self.fieldEdit.text())
        self.update()
        self.setStyleSheet("background-color: grey;")
    
    def onClickErase(self):
        self.lbl1._clear_size = int(self.fieldEdit1.text())
        self.update()
    
    def saveImage(self):
        self.lbl1.saveImage(self.path2)
        self.update()
        self.setStyleSheet("background-color: grey;")
    
    def merge_img (self):
        if(self.img2 is not None):
            self.lbl2.mergePic(self.path1)
            self.update()
            self.setStyleSheet("background-color: grey;")
    
    def inpainting (self):
        if(self.img2 is not None):
            self.lbl2.inpainting()
            self.update()
            self.setStyleSheet("background-color: grey;")
         
    def transfer(self):
        self.img1, self.img2 = self.lbl1.trasform()
        self.lbl1.image = self.img1
        self.lbl2.mask = self.img2
        self.lbl1.finished = True
        self.update()
        self.setStyleSheet("background-color: grey;")

    def create_UI(self):
        width = 1800
        height = 800
        self.setWindowTitle("Segmentierung")
        self.setGeometry(300,200,width,height)
        self.setFixedWidth(width)
        self.setFixedHeight(height)
        self.setStatusBar(QStatusBar(self))
        self.setStyleSheet("background-color: grey;")
        self.show()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())

