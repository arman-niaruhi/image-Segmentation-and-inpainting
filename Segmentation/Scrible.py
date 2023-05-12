from PySide2.QtWidgets import QLabel
from PySide2 import QtGui, QtCore
from PySide2.QtGui import  QPainter
from PySide2.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import qimage2ndarray
from Network import Net
import torch.nn as nn
import torch
import cv2
import time

start = None
class Scrible(QLabel):
    def __init__(self,path):
        super().__init__()
        width, height = 700, 500
        self.setFixedSize(width, height)
        self.path = path
        self.change = False
        self.image = QtGui.QImage(self.path)
        self.color = QtGui.QColor(QtCore.Qt.red)
        self.brushColor = self.color
        self.imageDraw = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.finished = False
        self.drawing = False
        self.brushSize = 15
        self._clear_size = 20
        self.lastPoint = QtCore.QPoint()
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
       
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image,self.image.rect())
        if(self.finished == False):
            canvasPainter.drawImage(self.rect(), self.imageDraw, self.imageDraw.rect())

    def mouseMoveEvent(self, event):  
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            painter = QtGui.QPainter(self.imageDraw)
            painter.setPen(QtGui.QPen(self.brushColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            if self.change:
                r = QtCore.QRect(QtCore.QPoint(), self._clear_size*QtCore.QSize())
                r.moveCenter(event.pos())
                painter.save()
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
                painter.eraseRect(r)
                painter.restore()
            else :
                painter.drawLine(self.lastPoint, event.pos())
                self.imgForeground = painter
            painter.end()
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False
    
    def changeColour(self):
        self.change = not self.change
        if self.change:
            pixmap = QtGui.QPixmap(QtCore.QSize(1, 1)*self._clear_size)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.drawRect(pixmap.rect())
            painter.end()

    def saveImage(self,path):
        self.image = QtGui.QImage(path)
        self.update()

    def trasform(self):
        rgb_im = Image.open(self.path).convert('RGB')
        imageOrg = np.array(rgb_im.resize([700,500])) / 255
        # rgb_im1 = Image.open(".Background1.png").convert('RGB')
        imageOrg1 = np.ones([500,700,1])
        
        mask = qimage2ndarray.recarray_view(self.imageDraw)
        y , x = mask.shape
        # x , y = mask.shape
        self.imgForeground = np.zeros((y,x)) 
        self.imgBackground = np.zeros((y,x)) 

        for i in range (y):
            for j in range (x):
                a = (mask[i,j])
                if(a[2] == 255):
                    self.imgForeground[i,j] = 1
                if(a[0] == 255):
                    self.imgBackground[i,j]= 1
        self.imgForeground = self.imgForeground.astype(int) 
        self.imgBackground = self.imgBackground.astype(int) 

        #get Pixel data of scribles in image
        PixelData_of_Background = self.getPixelData(self.imgBackground,imageOrg)
        PixelData_of_Foreground = self.getPixelData(self.imgForeground,imageOrg)
        allpixels = np.concatenate((PixelData_of_Foreground,PixelData_of_Background),axis = 0)

        labels = np.zeros ((allpixels.shape[0]))
        labels[0:PixelData_of_Foreground.shape[0]]=1
        labels = labels[:,np.newaxis]
        
        allPixelRGB = np.ones([allpixels.shape[0],5])
        allPixelRGB[:,0] = allpixels[:,0]
        allPixelRGB[:,1] = allpixels[:,1]
        allPixelRGB[:,2] = allpixels[:,2]
        allPixelRGB[:,3] = allpixels[:,3]/y
        allPixelRGB[:,4] = allpixels[:,4]/x
               
#######################  Training  #########################

        # Random Permutation
        # rand_p = torch.randperm(len(labels)-1)
        # allPixelRGB = torch.permute(allPixelRGB, rand_p)
        # labels = torch.permute(labels, rand_p)


        #all pixel (R,G,B) use for input 
        #labels compare to Loss function   
        trained_network = self.train(allPixelRGB, labels)
        
        ny = imageOrg.shape[0]
        nx = imageOrg.shape[1]
        # x = np.linspace(0, nx/max(ny,nx), nx) # Todo: 0-1
        # y = np.linspace(0, ny/max(ny,nx), ny) # Todo: 0-1

        x = np.linspace(0, 1, nx) # Todo: 0-1
        y = np.linspace(0, 1, ny) # Todo: 0-1
        xv, yv = np.meshgrid(x, y)
        coordinates = np.concatenate((yv.reshape(nx*ny,1),xv.reshape(nx*ny,1)),axis=1)

        imageMasknew = np.ones((ny*nx,1))
        imageMask_tensor = torch.from_numpy(imageMasknew)
        imageMask_tensor = imageMask_tensor.float()
        inferenceData = np.concatenate((np.reshape(imageOrg,(ny*nx,3)),coordinates),axis=1)
        
        inferenceData = torch.from_numpy(inferenceData)
        inferenceData= inferenceData.float()

        # Alle Pixel in das Netzwerk einfügen
        imageMask_tensor[:] = trained_network(inferenceData[:])
        imageMask_tensor[:] = torch.nn.functional.sigmoid(imageMask_tensor[:])

        imageMaskFinal = imageMask_tensor.detach().numpy()
        imageMaskFinal1 = (imageMaskFinal < 0.5)
        imageMaskFinal2 = (imageMaskFinal >= 0.5)
        imageMaskFinal1 = np.reshape(imageMaskFinal1, (ny,nx,1))
        imageMaskFinal2 = np.reshape(imageMaskFinal2, (ny,nx,1))


        # save the image
        cv2.imwrite("back.png", (imageMaskFinal2 * 1)*255)




        manipulatedImg = imageOrg1.copy()
        manipulatedImg = imageOrg*imageMaskFinal1 + (1-imageMaskFinal1)*imageOrg1
        image_masked1 = qimage2ndarray.array2qimage(manipulatedImg,normalize = True)
        return image_masked1, imageMaskFinal2
    

######################  Pixel extrahieren ###################   
    def getPixelData (self,mask, img):
        ny,nx,nc = img.shape
        y ,x = np.where (mask==1)

        number_of_scribbled_pixel = np.sum(mask)
        pixelData = np.zeros([number_of_scribbled_pixel, nc+2])
        pixelData[:,3] = y
        pixelData[:,4] = x

        for i in range (nc):
            pixelData[:,i] = img[y,x,i]
        
        return pixelData


    def train (self,x_train, y_train): # Batchsize??
        d_in = x_train.shape[1]
        d_out = y_train.shape[1] 
        d_hidden1 = 30#40
        d_hidden2 = 20#50
        d_hidden3 = 30

        net = Net(d_in, d_hidden1, d_hidden2, d_hidden3, d_out)

        x_train_torch = torch.from_numpy(x_train).float()
        y_train_torch = torch.from_numpy(y_train).float()

        lossFunction = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr= 1e-2)
        # optimizer = torch.optim.SGD(net.parameters(), lr= 1, momentum=0.9)
    

        maxIter = 10000
        tol = 1e-4
        Losses = []
        i = 0
        start = time.time()
        for i in range (maxIter):
            # Batch GD
            random_idx = torch.randint(0,x_train_torch.shape[0],(50,))
            prediction = net(x_train_torch[random_idx])
            loss = lossFunction(prediction, y_train_torch[random_idx])

            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            Losses.append(np.round(loss.detach().numpy(),4))
            if np.mod(i,100)== 0:
                print(i,loss.item())
                
            if loss < tol:
                break
        print(time.time() - start)
        plt.plot(range(1, i+2), Losses)
        plt.ylabel('Error')
        plt.xlabel('number of epochs')
        plt.savefig('fig.png')
        return net
