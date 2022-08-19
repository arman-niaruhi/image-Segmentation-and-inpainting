import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from patchify import patchify



img = np.array([[[1,2,3],[10,14,13],[5,6,8]],[[111,2,3],[1110,1114,1113],[115,611,118]],[[2221,2222,3],[2210,2214,2213],[225,226,228]]])
print(img[:,0,0])
w =4
image_padded = np.pad(img, w,'wrap')
img_patch = patchify(image_padded,w)
print(img_patch.shape)
print(img_patch[6,0])
img_dir = "img.jpg"
img_pil = Image.open (img_dir).convert('L')
image = np.array(img_pil)/255

st= 0.15
gUAAIn = np.random.normal(0,st,(image.shape[0],image.shape[1]))
image = np.clip(image,cmap ="gray")

patchsize = 5
p_h = patchsize /2
ny, nx = image.shape
img_pad = np.pad(image,p_h)
img_res= np.zeros_like(image)

for i in range (ny):
    for j in range (nx):
        patch = img_pad[i:i+2*p_h+1,j:j+2*p_h+1]
        img_res[i,j]= np.median(patch.flatten())



#Medianfilter Fast
patchsize = 15
p_h = patchify/2
patch = patchify(image,patchsize)
no_x, n_y, w, h = patch.shape

ein_patch_median = np.median(patch[0,0,:,:])
patch_median = np.median(patch[:,:,:,:], axis=[2,3])
print(patch_median.shape)
plt.imshow(patch_median,cmap="gray")
plt.show()