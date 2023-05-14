import numpy as np
from matplotlib import pyplot as plt
    
#Singular Value Decomposition class
class SVD:
    def __init__(self):
        # Initialize instance variables
        pass


    def to_greyscale(self, im):
        return 1 - np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
    

    def compare_images(self, A, A_):
        #Helper function to compare two images

        fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

        axarr[0].imshow(A, cmap = "Greys")
        axarr[0].axis("off")
        axarr[0].set(title = "original image")

        axarr[1].imshow(A_, cmap = "Greys")
        axarr[1].axis("off")
        axarr[1].set(title = "reconstructed image")


    def svd_reconstruct(self, img, k):
        # Reconstructs an image from its singular value decomposition

        img = self.to_greyscale(img)
        U, sigma, V = np.linalg.svd(img)

        # Create the D matrix in the SVD
        D = np.zeros_like(img,dtype=float) # matrix of zeros of same shape as img
        D[:min(img.shape),:min(img.shape)] = np.diag(sigma) # singular values on the main diagonal
        
        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]
        A_ = U_ @ D_ @ V_

        return A_
    

    def svd_experiment(self, img):
        # Perform several experiments with different values of k

        fig, axarr = plt.subplots(8, 1, figsize = (7, 3*8))
        current = 0

        for k in [2, 5, 10, 25, 50, 75, 100, 150]:
            img_grey_ = self.svd_reconstruct(img, k)

            # Calculate the amount of storage
            m,n = img_grey_.shape
            percent = 100 * k * (m + n + 1) / (m * n)

            # Use the grayscale colormap for the image
            axarr[current].imshow(img_grey_, cmap="Greys")
            axarr[current].axis("off")
            axarr[current].set(title = f"{k} components, % storage = {percent:.2f}")
            current += 1

