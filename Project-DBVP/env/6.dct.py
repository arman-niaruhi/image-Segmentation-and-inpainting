import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify, unpatchify


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


def denoise_DCT(threshold, img):
    
    # Create Patches of size 16 x 16
    img_patch = patchify(img, [16,16])
    ny_block , nx_block , y_size_block, x_size_block = img_patch.shape
    print(ny_block,nx_block , y_size_block, x_size_block )

    img1 = 0 * np.copy(img_patch)
    for i in range (ny_block):
        for j in range(nx_block):
            img1[i,j]=dct2(img_patch[i,j])
    img1[abs(img1)< threshold] = 0



    idct_result = 0 * np.copy(img_patch)
    for i in range (ny_block):
        for j in range(nx_block):
            idct_result[i,j]= idct2(img1[i,j])


    # Recreate Image from Patches
    img_denoised = unpatchify(idct_result, img.shape)

    return img_denoised

def main():
    
    # Load Image
    img_dir = "giraffe.jpg"
    img_pil = Image.open(img_dir).convert('L')
    img_nonoise = np.array(img_pil.resize([256,256]))/255
    img_nonoise = img_nonoise[50:210,50:210]

    # Add Noise
    st = 0.05
    gaussian = np.random.normal(0, st, (img_nonoise.shape[0],img_nonoise.shape[1])) 
    img = img_nonoise + gaussian
    
    # Plot Image With and Without Noise
    # plt.figure(figsize=(10,15))
    # sp1 = plt.subplot(121)
    # sp1.imshow(img_nonoise, 'gray')
    # sp2 = plt.subplot(122)
    # sp2.imshow(img, 'gray')
    # plt.show()
    
    threshold = st * 3
    img_denoised = denoise_DCT(threshold, img)
    
    # Plot
    plt.figure(figsize=(10,15))
    sp1 = plt.subplot(121)
    sp1.imshow(img_denoised, 'gray')
    sp1 = plt.subplot(122)
    sp1.imshow(img, 'gray')
    plt.show()


if __name__ == "__main__":
    main()




        








