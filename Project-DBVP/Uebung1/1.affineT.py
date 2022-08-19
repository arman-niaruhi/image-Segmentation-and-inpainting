import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import numpy as np


def getAffineTranf(X, A):
    
# Schritt 1 => Vekor X Spaltenweise zu schreiben

    x_ravel = X.ravel()
 #######################################################   
    A_long = np.append(A, np.ones((3,1)), axis =1)
    A_neu = np.empty((0,6))
    A_1 = np.kron(np.eye(2), A_long[0,:])
    A_2 = np.kron(np.eye(2), A_long[1,:])
    A_3 = np.kron(np.eye(2), A_long[2,:])
    
    A_neu = np.vstack((A_1,A_2,A_3))
    print(A_neu)

    H_ravel = np.linalg.solve(A_neu, x_ravel)

    H = np.reshape(H_ravel, (2,3))
# invert and apply affine transformation
    H = np.vstack((H, np.zeros((1,3))))
    H[2,2] = 1
    H = np.linalg.inv(H)  
    return H

##################################################
   
#Alternativ
    #Kron + Eye
#     x_ravel = X.flatten()
#     A_long = np.append(A, np.ones((3,1)), axis =1)
#     A_neu = np.empty((0,6))
#     for i in range(A.shape[0]):
#         A_1 = np.kron(np.eye(2), A_long[i,:])
#         A_neu = np.append (A_neu, A_1 , axis = 0)
#     print(A_neu)
# # Schritt 3 => H finden!
# # Ax = b 
#     H_ravel = np.linalg.solve(A_neu, x_ravel)

# # Schritt 4 => H in Form 2*3 Matrix umformen

#     H = np.reshape(H_ravel, (2,3))
# # invert and apply affine transformation
#     H = np.concatenate((H, np.zeros((1,3))),axis=0)
# # Alternative# matrix_trans = np.vstack((matrix_trans, np.zeros((1,3))))
#     H[2,2] = 1
#     H = np.linalg.inv(H)  
#     return H


def main():
    
    # load image
    img = Image.open('img.png').convert('L')
    # coordinates
    a1 = [114,160]
    a2 = [777,42]
    a3 = [148,412]
    
    A = np.float32([a1, a2, a3])

    # transform to
    x1 = [100,150]
    x2 = [777,150]
    x3 = [100,400]
    
    X = np.float32([x1, x2, x3])
    # get affine transformation matrix
    H = getAffineTranf(X, A)
    
    transformed_img = ndimage.affine_transform(img, H)

    # plot image 
    plt.imshow(transformed_img, cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == "__main__":
    main()