import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.__config__ import show
import scipy.ndimage.morphology as morph
from skimage import measure



def erosion(img, kernel):
    img_padding =np.pad(img, 1, "wrap")
    x_axe , y_axe = img.shape
    out = np.zeros_like(img)
    for i in range (1,x_axe-1):
        for j in range (1,y_axe-1):
            pad = img_padding[i-1:i+2, j-1:j+2]
            summe = np.sum(pad*kernel) 
            summe_kern = np.sum(kernel)
            if summe == summe_kern:
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out

def dilatation(img, kernel):
    img_padding =np.pad(img, 1, "wrap")
    x_axe , y_axe = img.shape
    out = np.zeros_like(img)
    for i in range (1,x_axe-1):
        for j in range (1,y_axe-1):
            pad = img_padding[i-1:i+2, j-1:j+2]
            summe = np.sum(pad*kernel) 
            if summe >= 1:
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out


def closing(img, kernel):
    return(erosion(dilatation(img, kernel), kernel))

def opening(img, kernel):
    return(dilatation(erosion(img, kernel), kernel))

def getLargetCC(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0)
    lartgestCC = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    return lartgestCC

def main():
    
    # Load Image
    img_dir = "box.jpg"
    img_pil = Image.open(img_dir).convert('L')
    img_pil = img_pil.resize((img_pil.size[0]//10,img_pil.size[1]//10))
    img = np.array(img_pil)/255
    org_img = img.copy()
    
    ### Your Code ###

     #vertikaler und horizontaler Linie als Strukturelement

    kernel_zeros = np.zeros((3,3))
    kernel_vertical = kernel_zeros.copy()
    kernel_vertical[1,:] = 1
    kernel_horizontal = kernel_zeros.copy()
    kernel_horizontal[:,1] = 1  
    kernel_quadrat = np.ones_like(kernel_zeros)


    #binaerbild mit Schwellwertverfahren
    thresholder = 0.75
    binaryImg = img > thresholder

    # img[img<thresholder] = 1
    # img[img>=thresholder] = 0   

    #Closing entlang jeder Achse
    closeImage_vertical = closing(binaryImg, kernel_vertical)
    closeImage = closing(closeImage_vertical, kernel_horizontal)

    #final binary Image
    binaryImage = morph.binary_fill_holes(closeImage).astype(int)

    #opening und give labels
    res_opening = opening(binaryImage, kernel_quadrat)

    result = getLargetCC(res_opening) 
    print(result)
    


    # Plot 
    plt.figure(figsize=(10,15))
    sp1 = plt.subplot(121)
    sp1.imshow(result, 'gray')
    sp2 = plt.subplot(122)
    sp2.imshow(org_img, 'gray')
    plt.show()
    



if __name__ == "__main__":
    main()




        








