### K-Means Visualization ###

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from numpy import concatenate, linalg as LA
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, binary_fill_holes,generate_binary_structure
from scipy.ndimage import gaussian_filter
from skimage import measure

def kMeans(testimg, k, maxIter):   
    
    # This is not a good initialization: Sometimes we get empty clusters!
    centroids = np.random.rand(testimg.shape[1],k)

    for i in range(maxIter):
        # step 1: update mask
        diff = np.zeros((testimg.shape[0],k))


        for l in range(k):
            diff[:,l] = np.sum((centroids[:,l]- testimg)**2,axis = 1)

        #kleinste Distanz
        indi = np.argmin(diff, axis=1)


        plot_points = np.concatenate((testimg,centroids.T))
        #k, k+1, and k +2 sind controids oder 3 unterschiedliche Farben
        colors = np.concatenate((indi,[k,k+1,k+2])) 
        plt.scatter(plot_points[:,0],plot_points[:,1],c=colors)
        plt.show()

        # stept 2: update colors
        for actClass in range(k):
            #ml ist punkte mit wert actClass
            ml = indi==actClass
            #hoher Boolean wert in test img form umformen
            count = np.max([np.sum(ml),1])
            ml_3d = np.repeat(ml[:,np.newaxis], testimg.shape[1], axis=1)
            centroids[:,actClass] = 1/count*(ml_3d*testimg).sum(axis=0)

 
def main():

    
    
    # Create 2D Datapoints
    N = 50
    x = np.random.rand(N,2) * 0.3
    y = np.random.rand(N,2) * 0.2 + 0.5
    z = np.random.rand(N,2) * 0.2 + [0.5, 0]
    points = np.concatenate((x, y, z))
    # Plot Points
    plt.scatter(points[:,0],points[:,1])
    plt.show()

    # Kmeans
    nrClasses = 3
    maxIter = 5
    kMeans(points, nrClasses, maxIter)


    
if __name__ == "__main__": 
    main()