import numpy as np

def helmert(k):
    # Helmert sub-matrix: (n-1) x n
    # Dryden, I. L., & Mardia, K. V. (2016). 
    # Statistical shape analysis: with applications in R, page 49
    H = np.zeros((k-1,k))
    for j in range(1, k):
        hj = -1/np.sqrt(j*(j+1))
        H[j-1] = np.concatenate((np.ones(j)*hj, -j*hj, np.zeros(k-j-1)) ,axis = None)
    return H

def shape_transform(X):
    # X : k x d x n array; n samples and each sample consists of k landmarks in R^d
    # output: an n x (k-1) x 1 complex array
    k, d, n = X.shape
    X = np.transpose(X, (2,0,1))
    H = helmert(k)
    Z = np.matmul(H, X) # n x (k-1) x d
    Z = Z/np.expand_dims(np.linalg.norm(Z, axis = (1,2)),axis=(1,2))
    Z = Z[:,:,0] + 1j*Z[:,:,1]
    Z = np.expand_dims(Z, axis = 2)
    return Z
