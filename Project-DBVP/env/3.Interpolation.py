import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from PIL import Image

def hut(x):
    b = np.array([[0 , -x[2], x[1]],
              [x[2], 0, -x[0]],
              [-x[1], x[0],0]])
    return b
    
def konstruiere_bi(a,x):
    temp = hut(x)
    return np.append(np.append(a[0]*temp, a[1]*temp, axis=1), a[2]*temp, axis=1)

def konstruiere_b(A,X):
    B1 = konstruiere_bi(np.append(A[:,0],[1]),np.append(X[:,0],[1]))
    B2 = konstruiere_bi(np.append(A[:,1],[1]),np.append(X[:,1],[1]))
    B3 = konstruiere_bi(np.append(A[:,2],[1]),np.append(X[:,2],[1]))
    B4 = konstruiere_bi(np.append(A[:,3],[1]),np.append(X[:,3],[1]))
    return np.vstack((B1,B2,B3,B4))
    

def kleinster_ev(B):
    w, v = LA.eig(B)
    ind = np.argmin(w)
    return v[:,ind]
    
def interpolateBilin(img,coord):

    # change the coordinates to the floor numbers
    i = np.int(np.floor(coord[0]))
    j = np.int(np.floor(coord[1]))
    
    # control i and j to be in range of our bild coordinate and do not be negativ numbers
    if i > img.shape[0]-2:
        return 0.0
    if j > img.shape[1]-2:
        return 0.0
    if i < 0:
        return 0.0
    if j < 0:
        return 0.0
    
    #calculate the new changed of i and j with bilinear interpolation 
    x_coord1 = ((coord[0]-i) * img[i+1, j] +(i+1-coord[0])*img[i,j]) * (j+1 - coord[1]) 
    x_coord2= ((coord[0]-i) * img[i+1, j+1] +(i+1-coord[0])*img[i,j+1]) * (coord[1]-j)
    return x_coord2 + x_coord1
    
def main():

    # coordinates
    A = np.array([[538,850,380,673],[325,333,1117,1117]], dtype=float)
    X = np.array([[500,800,500,800],[200,200,1000,1000]], dtype=float)

    # create transformation matrix
    B = konstruiere_b(A,X)
    BtB = np.matmul(B.T,B)
    H = np.linalg.inv(kleinster_ev(BtB).reshape(3,3).T)

    # load image
    img_grey_pil = Image.open("box.jpg").convert('L')
    img = np.array(img_grey_pil) / 255.0


    # perform bilinear interpolation to correct the perspective distortion
    shiftedImg = np.copy(img)
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        x = H @ np.append(it.multi_index, [1])
        x = x / x[2]
        shiftedImg[it.multi_index] = interpolateBilin(img, x)
        it.iternext()

    # visualize the result
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.imshow(img_grey_pil,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(shiftedImg,cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    main()