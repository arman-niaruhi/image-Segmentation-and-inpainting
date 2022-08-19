import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import cv2
from PIL import Image

def hut(x):
    mat = np.array([ [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0] 
            ])
    return mat
#B =(a1*x , a2*x , a3*x)
def konstruiere_bi(a,x):
    return np.hstack(( a[0]*hut(x), a[1]*hut(x), a[2]*hut(x)))

#B(B1,B2,B3,B4)T
def konstruiere_B(A,X):
    B1 = konstruiere_bi(np.append(A[:,0],[1]), np.append(X[:,0],[1]))
    B2 = konstruiere_bi(np.append(A[:,1],[1]), np.append(X[:,1],[1]))
    B3 = konstruiere_bi(np.append(A[:,2],[1]), np.append(X[:,2],[1]))
    B4 = konstruiere_bi(np.append(A[:,3],[1]), np.append(X[:,3],[1]))
    return np.vstack((B1, B2, B3, B4))
    
def kleinster_ev(B):
    w, v = LA.eig(B)
    minimale_EW = np.argmin(w)
    return v[:,minimale_EW]
    
def main():

    # Load Image
    img = Image.open('giraffe.jpg').convert('L')
    img = np.array(img)   
    #oder
    # img = np.asarray(img)
    # # Coordinates
    A = np.array([[100,100,300,300], \
                  [100,300,100,300]], dtype=float)
    X = np.array([[100,100,290,300],
                  [120,300,100,300]], dtype=float)
    print(A.shape)
    B = konstruiere_B(A,X)
    BtB = np.matmul(B.T,B)
    H = kleinster_ev(BtB).reshape(3,3).T
    
    
    # Apply Transformation
    wraped = cv2.warpPerspective(img, H, (img.shape[1],img.shape[0]))

    # plot image
    plt.imshow(wraped, cmap="gray")
    plt.show()

    
if __name__ == "__main__":
    main()