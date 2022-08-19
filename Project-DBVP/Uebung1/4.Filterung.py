import numpy as np
import matplotlib.pylab as plt
from PIL import Image


def plot_results(img, new_img):
        
    plt.subplot(221)
    plt.imshow(img,'gray')
    plt.axis('off')
    plt.title("Image")

    plt.subplot(222)
    plt.imshow(new_img,'gray')
    plt.axis('off')
    plt.title("Sharpened")

    plt.subplot(223)
    plt.imshow(img[100:300,400:600],'gray')
    plt.axis('off')
    plt.title("Image-Zoomed In")

    plt.subplot(224)
    plt.imshow(new_img[100:300,400:600],'gray')
    plt.axis('off')
    plt.title("Sharpened-Zoomed In")

    plt.show()

def apply_conv3x3(kernel, img):
    ### Your Code ###
    img_padding =np.pad(img, 1, "wrap")
    out = np.zeros_like(img)
    kernel = np.rot90(kernel,2,axes=(0,1))
    x_axe , y_axe = img.shape
    
    for i in range (1,x_axe+1):
        for j in range (1,y_axe+1):
            faltung = img_padding[ i -1 : i +2 , j -1: j +2]
            res= faltung * kernel
            out[i-1,j-1] =np.sum(res)
    return out
    
#####alternativ 1 ##################################################    
    # img_padding =np.pad(img, 1, "wrap")
    # out = np.copy(img)
    # y_axe , x_axe = img.shape
    # kernel = np.rot90(kernel,2,axes=(0,1))
    # for i in range (1,y_axe+1):
    #     for j in range (1,x_axe+1):
    #         faltung = img_padding[ i -1 : i +2 , j -1: j +2]
    #         out[i-1,j-1] =np.sum(faltung*kernel)
    # return out

#####alternativ 2 #############################################
    # patchsize = kernel.shape[0]

    # nx,ny = img.shape
    # img_pad = np.pad(img,patchsize//2)
    # img_res = np.copy(img)
    # kernel = np.rot90(kernel,2,axes=(0,1))
    # # Loop over pixel
    # for i in range(nx):
    #     for j in range(ny):
    #         img_res[i,j] = np.sum(img_pad[i:(i+patchsize), j:(j+patchsize)]*kernel)

    # return img_res
##################################################

def main():

    # Load Image
    img_dir="giraffe.jpg"
    img_pil=Image.open(img_dir).convert('L')
    img=np.array(img_pil)/255
    # Run Convolution
    alpha = 2
    kernel = np.array([[0, -alpha, 0], [-alpha, 4+alpha*4, -alpha], [0, -alpha, 0]])/4
    new_img = apply_conv3x3(kernel, img)
    new_img[new_img>1]=1
    new_img[new_img<0]=0

    plot_results(img, new_img)
    

   
if __name__ == "__main__": 
    main()