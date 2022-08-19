import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from patchify import patchify, unpatchify


def transform_colorspace(img):
    y , x ,z = img.shape 
    vec_img = img.reshape((y*x,z)).T
    tras = np.array([
        [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]
        ,[1/np.sqrt(2),0,-1/np.sqrt(2)]
        ,[1/np.sqrt(6),-2/np.sqrt(6),1/np.sqrt(6)]])
    new_vec = tras @ vec_img
    new_img = new_vec.T.reshape((y,x,z))


def transform_colorspace_back(img):
    sy , sx ,sz = img.shape
    vec_img = img.reshape((sy*sx,sz)).T
    tras = np.array([
        [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]
        ,[1/np.sqrt(2),0,-1/np.sqrt(2)]
        ,1/np.sqrt(6),-2/np.sqrt(6),1/np.sqrt(6)])
    new_vec = tras.T @ vec_img
    new_img = new_vec.T.reshape((sy,sx,sz))

def DCT_denoise_Farbig(threshold, img, img_trans):
  
    new_img_smooth = np.copy(img_trans)
    old_img_smooth =np.copy(img)

    for i in range (img.shape[2]):
        new_img_smooth[:,:,i] = denoise_DCT(threshold, img_trans[:,:,i])
        old_img_smooth[:,:,i] = denoise_DCT(threshold, img[:,:,i])
    
    transform_colorchannel_back_and_plot(new_img_smooth,old_img_smooth,img,img_trans)


def sharpening_farbig(alpha, img, img_trans):
    kernel = np.array ([[0,-alpha,0],[-alpha,-alpha*4,-alpha], [0,-alpha,0]])/4
    new_img_sharp = np.copy(img_trans)
    old_img_sharp = np.copy(img)
    for i in range (img.shape[2]):
        new_img_sharp[:,:,i] = apply_conv3x3(kernel, img_trans[:,:,i])
        old_img_sharp[:,:,i] = apply_conv3x3(kernel, img[:,:,i])
    transform_colorchannel_back_and_plot(new_img_sharp,old_img_sharp,img,img_trans)


def bilateral_smoothing(img, img_trans):
    new_img_bilat = np.copy(img_trans)
    old_img_bilat = np.copy(img)
    for i in range (img.shape[2]):
        new_img_bilat[:,:,i] = bilateralFilter(img_trans[:,:,i],5,0.05,5)
        old_img_bilat[:,:,i] = bilateralFilter(img[:,:,i],5,0.05,5)
    transform_colorchannel_back_and_plot(new_img_bilat,old_img_bilat,img,img_trans)

def transform_colorchannel_back_and_plot(new_img,old_img,original_im,img_trans):
        
    img_uncorr = transform_colorspace_back(new_img)
    img_uncorr[img_uncorr>1]= 1
    img_uncorr[img_uncorr<0] = 0
    old_img[old_img>1] =1
    old_img[old_img<0] =0

    img_R = np.copy(img_trans)
    img_G = np.copy(img_trans)
    img_B = np.copy(img_trans)

    img_R[:,:,0] = new_img[:, :, 0]
    img_G[:,:,1] = new_img[:, :, 1]
    img_B[:,:,2] = new_img[:, :, 2]

    img_R = transform_colorspace_back(img_R)
    img_G = transform_colorspace_back(img_G)
    img_B = transform_colorspace_back(img_B)

    img_R[img_R>1] = 1
    img_R[img_R<0] = 0
    img_G[img_G>1] = 1
    img_G[img_G<0] = 0
    img_B[img_B>1] = 1
    img_B[img_B<0] = 0


    plt.figure(figsize = (40,40))
    spl = plt.subplot(231)
    spl.imshow(original_im) 
    plt.title =("Original")
    spl = plt.subplot(232)
    spl.imshow(img_uncorr) 
    plt.title =("Uncorraled")
    spl = plt.subplot(233)
    spl.imshow(old_img) 
    plt.title =("RGB")
    
    spl = plt.subplot(234)
    spl.imshow(img_R) 
    plt.title =("Uncorraled _ channel1")
    spl = plt.subplot(235)
    spl.imshow(img_G) 
    plt.title =("Uncorraled _ channel2")
    spl = plt.subplot(236)
    spl.imshow(img_B) 
    plt.title =("Uncorraled _ channel3")


def bilateralFilter(img_noisy, w, sigma_space, sigma_int):
    x, y = img_noisy.shape

    image_padded = np.pad(img_noisy,w, "wrap")
    denoisedBilateral = np.zeros_like(img_noisy)
    
    temp0 = np.exp(-sigma_space*(np.arange(-w,w+1))**2)
    size = temp0.shape[0]
    temp1 = np.zeros((size,size))
    
    for a in range (temp0.shape[0]):
        for b in range (temp0.shape[0]):
            temp1[a,b] = temp0[a]*temp0[b] 
    
    for i in range(w, x +w):
        for j in range(w, y +w):
            kern_2 = np.zeros([2*w+1,2*w+1])


            for m in range(-w,w+1):
                for l in range(-w,w+1):
                    kern_2 [m+w,l+w]= np.exp(-sigma_int*(image_padded[i,j]-image_padded[i+m,j+l])**2)


            filter = temp1 * kern_2       
            denoisedBilateral[i-w,j-w] = np.sum(filter/np.sum(filter)*image_padded[i-w:i+w+1,j-w:j+w+1])       
    
    return denoisedBilateral    


def denoise_DCT(threshold, img):
  
    img_patch = patchify(img, [16,16])
    ny_block , nx_block , y_size_block, x_size_block = img_patch.shape


    img1 = 0 * np.copy(img_patch)
    for i in range (ny_block):
        for j in range(nx_block):
            img1[i,j]=dct2(img_patch[i,j])
    img1[abs(img1)< threshold] = 0



    idct_result = 0 * np.copy(img_patch)
    for i in range (ny_block):
        for j in range(nx_block):
            idct_result[i,j]= idct2(img1[i,j])

    img_denoised = unpatchify(idct_result, img.shape)
    return img_denoised

def apply_conv3x3(kernel, img):
    img_padding =np.pad(img, 1, "wrap")
    kernel = np.rot90(kernel,2,axes=(0,1))
    x_axe , y_axe = img.shape
    out = np.zeros_like(img)
    for i in range (1,x_axe+1):
        for j in range (1,y_axe+1):
            faltung = img_padding[ i -1 : i +2 , j -1: j +2]
            res = np.zeros_like(faltung)
            for e in range(faltung.shape[0]):
                for k in range(kernel.shape[0]):
                        res[e][k] = faltung[e][k] * kernel[e][k]
            out[i-1,j-1] =np.sum(res)
    return out

def dct(f):
    n = f.size
    c = np.zeros(n)
    c[0] = 1/np.sqrt(n)*np.sum(f)
    for k in range(1,n):
        c[k] = np.sqrt(2/n)*np.sum(f*np.cos(np.pi*k*(np.arange(n)+0.5)/n))
    return c

def idct(c):
    n = c.size
    f = np.zeros(n)
    for l in range(0,n):
        f[l] = np.sqrt(1/n)*c[0] + np.sqrt(2/n)*np.sum(c[1:n]*np.cos(np.pi*np.arange(1,n)*(l+0.5)/n))
    return f

def dct2(f):
    ny,nx = f.shape
    c = np.zeros([ny,nx])
    for s in range(ny):
        c[s,:] = dct(f[s,:])
    for t in range(nx):
        c[:,t] = dct(c[:,t])
    return c

def idct2(c):
    ny,nx = c.shape
    f = np.zeros([ny,nx])
    for s in range(ny):
        f[s,:] = idct(c[s,:])
    for t in range(nx):
        f[:,t] = idct(f[:,t])
    return f

def main():

    # load image
    img_dir = "statue.jpg"
    rgb_im = Image.open(img_dir).convert('RGB')
    print (rgb_im)
    img = np.array(rgb_im.resize([640//3,426//3])) / 255
    c = np.random.rand(3,6)
    print (c[2,2])
    img_noisy = img + np.random.normal(0,0.1,img.shape)

    new_img_noisy = transform_colorspace(img_noisy)
    new_img = transform_colorspace(img)

    threshold = 0.05 * 3
    DCT_denoise_Farbig(threshold,img_noisy,new_img_noisy)
    plt.savefig('DCT_Denosing.png', bbox_inches = "tight")

    sharpening_farbig(1,img,new_img)
    plt.savefig('sharpening.png', bbox_inches = "tight")

    bilateral_smoothing(img_noisy,new_img_noisy)
    plt.savefig('bilateral.png', bbox_inches = "tight")

    
    

if __name__ == "__main__": 
    main()