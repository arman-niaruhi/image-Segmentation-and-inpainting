import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from patchify import patchify

def bilateralFilter(img_noisy, w, sigma_space, sigma_int):
    ### Your Code ###
    y, x = img_noisy.shape

    image_padded = np.pad(img_noisy,w, "wrap")
    denoisedBilateral = np.copy(img_noisy)
    
#distanz von der mitte in jedem Pixel mit Gausskernel
    gausKernel = np.exp(-sigma_space*np.arange(-w,w+1)**2)
    gausKernel2d = np.outer(gausKernel,gausKernel)
   
    #alternatev##################################################
    # size = gausKernel.shape[0]
    # gausKernel2d = np.zeros((size,size))
    # for a in range (gausKernel.shape[0]):
    #     for b in range (gausKernel.shape[0]):
    #         gausKernel2d[a,b] = gausKernel[a]*gausKernel[b] 
    # plt.imshow(gausKernel2d)
    # plt.show()
    #############################################################
    for i in range(k, ny + k):
        for j in range(k, nx + k):

            # Kernel für jedes Pixel berechnen
            kern_2 = np.copy(gausKernel2d)
            for m in range( 0, 2*k+1 ):
                for l in range(  0, 2*k +1):
                    spacial_filter[l , m]= np.exp(-sigma_int * (image_padded[i,j]-image_padded[l, m])**2)
            #####################################

            ####Filter aufbauen und normieren#####
            filterKernel = gausKernel2d * kern_2 
            filterKernel = filterKernel/ np.sum(filterKernel)     
            ######################################
            
            denoisedBilateral[i-w,j-w] = np.sum(filterKernel * image_padded[i-w:i+w+1,j-w:j+w+1])       
    
    return denoisedBilateral    

def main():
    # load image
    img_dir = "statue.jpg"
    img_pil = Image.open(img_dir).convert('L')
    img = np.array(img_pil) / 255
    
    #add noisy
    img_noisy = img + np.random.normal(0,0.1,img.shape)
    img_noisy = img_noisy[0:200,0:200]

    # apply filter
    w = 5
    sigma_space = .05
    sigma_int = 5
    new_img = bilateralFilter(img_noisy, w, sigma_space, sigma_int)

    # plot result
    plt.subplot(121)
    plt.imshow(img_noisy, 'gray')
    plt.subplot(122)
    plt.imshow(new_img, 'gray')
    plt.show()

if __name__ == "__main__": 
    main()
