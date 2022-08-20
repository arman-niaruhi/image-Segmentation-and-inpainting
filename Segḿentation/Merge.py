from PySide2.QtWidgets import QLabel
from PySide2 import QtGui, QtCore
from PySide2.QtGui import  QPainter
from PySide2.QtCore import Qt
import numpy as np
from PIL import Image
import qimage2ndarray
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import linalg, kron, identity, vstack
from scipy.ndimage.morphology import binary_dilation 



class Merge(QLabel):
    def __init__(self,path):
        super().__init__()
        width, height = 700, 500
        self.setFixedSize(width, height)
        self.mask = np.array((500,700))
        self.path = path
        self.drawing = False
        self.image = QtGui.QImage(self.path)
        # self.vordergrund = QtGui.QImage( "pexels-pascal-renet-1089306.jpg")
        # self.vordergrund.fill(QtCore.Qt.transparent)
        self.lastPoint = QtCore.QPoint()
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
       
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image,self.image.rect())
    #     # if(self.change == False):
    #     #     canvasPainter.drawImage(self.rect(), self.vordergrund, self.vordergrund.rect())

    def mouseMoveEvent(self, event):  
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            # painter = QtGui.QPainter(self.image)
        #     if self.change:
        #         painter.setTransform(QtGui.QTransform())
        #         # r = QtCore.QRect(QtCore.QPoint(), self._clear_size*QtCore.QSize())
        #         # r.moveCenter(event.pos())
        #         # painter.save()
        #         # painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        #         # painter.eraseRect(r)
        #         # painter.restore()
            # painter.save()
            # painter.end()
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False
    
    def mergePic(self,path_mask_img):
        rgb_im = Image.open(self.path).convert('RGB')
        self.image_Background = np.array(rgb_im.resize([700,500])) / 255
        rgb_Background = Image.open(path_mask_img).convert('RGB')
        self.image_mask = np.array(rgb_Background.resize([700,500])) / 255
        self.final_array = self.image_mask * self.mask + ( 1 - self.mask ) * self.image_Background
        final = qimage2ndarray.array2qimage(self.final_array,normalize = True)
        self.changed = True
        self.image = final
    
    def inpainting (self):
        ny,nx,nc= self.final_array.shape
        ## img Vektorisieren 
        img_vec = np.reshape(self.final_array, (ny*nx,3), order='F')
        ## mask Vektorisieren
        self.mask_vec = np.reshape(self.mask, (ny*nx,1), order='F')
        ## Differenzen in y-Richtung
        Dy = kron(identity(nx), self.generate_D_matrix(ny))
        ## Differenzen in x-Richtung
        Dx = kron(self.generate_D_matrix(nx),identity(ny))

        self.D_hut = vstack([Dy, Dx])
        # helper = self.image_Background.copy()
        # dilated_mask = binary_dilation(self.mask, np.ones((3,3)))
        # helper[dilated_mask] = self.image_mask[dilated_mask]
        # manipulated =  self.solve_one_D_Problem(helper, img_vec,nx*ny)
        manipulated =  self.solve_one_D_Problem( img_vec,nx*ny)
        manipulated = np.reshape(manipulated,(ny,nx,3))*255
        image_masked = qimage2ndarray.array2qimage(manipulated,normalize = True)
        self.image = image_masked
        
    def generate_D_matrix(self,n):
        data = np.ones((2,n))
        data[1,:] *=-1 
        diags = np.array([0, 1])
        D = spdiags(data, diags, n-1, n)
        return D
    
    def b_addition(self,helper):
        ny,nx,nc= self.final_array.shape
        extra_b = np.array((3,ny*nx))
        for i in range (3):
            extra_b[i] = self.D_hut@helper[:,:,i].reshape(ny*nx,order='F')
        return extra_b
    # def solve_one_D_Problem(self,helper,f,n):    
    def solve_one_D_Problem(self,f,n):
        # b = -self.D_hut@((1-self.mask_vec)*f) + self.b_addition(helper)
        b = -self.D_hut@((1-self.mask_vec)*f) 
        I = np.where(self.mask_vec)[0]
        m = np.sum(self.mask_vec)

        #Hilfe von coo_matrix: A[i[k], j[k]] = data[k]
        calig_1 = coo_matrix((np.ones(m), (I, np.array(range(m)))), shape=(n, m))

        u = np.zeros((n,3,1))
        A= self.D_hut@calig_1
        for i in range (3):
            x = linalg.lsqr(A,b[:,i])[0]
            u[:,i][self.mask_vec] = x
            u[:,i][~self.mask_vec] = f[:,i,np.newaxis][~self.mask_vec]
        return u
    