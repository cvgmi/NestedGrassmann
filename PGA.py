from compute_centroid import *
import numpy as np
from sklearn.decomposition import PCA
from sPCA import *
from Complex_PCA import *

def PGA(X, m, man):
    """
    tangent PCA
    """
    n = X.shape[0]
    FM = compute_centroid(man, X)
    logX = np.zeros(X.shape).reshape((n, -1))
    for i in range(n):
        logX[i] = man.log(FM, X[i]).reshape(-1)

    pca = PCA(n_components=m)
    logX_trans = pca.fit_transform(logX)
    logX_trans = pca.inverse_transform(logX_trans)
    logX_trans = logX_trans.reshape(X.shape)

    X_trans = np.zeros(X.shape)
    for i in range(n):
        X_trans[i] = man.exp(FM, logX_trans[i])

    #var_ratio = np.sum(pca.explained_variance_ratio_)
    #return var_ratio, X_trans
    return pca

def sPGA(X, y, m, man):
    """
    tangent supervised PCA
    """
    n = X.shape[0]
    FM = compute_centroid(man, X)
    logX = np.zeros(X.shape).reshape((n, -1))
    for i in range(n):
        logX[i] = man.log(FM, X[i]).reshape(-1)

    spca = SPCA(n_components = m, metric = delta)
    logX_trans = spca.fit_transform(logX, y)
    logX_trans = spca.inverse_transform(logX_trans)
    logX_trans = logX_trans.reshape(X.shape)

    X_trans = np.zeros(X.shape)
    for i in range(n):
        X_trans[i] = man.exp(FM, logX_trans[i])

    var_ratio = np.sum(spca.explained_variance_ratio_)
    return var_ratio, X_trans

def Complex_PGA(X, m, man):
    """
    tangent PCA for complex manifolds
    """
    n = X.shape[0]
    FM = compute_centroid(man, X)
    logX = np.zeros(X.shape, dtype = X.dtype)
    logX = logX.reshape((n, -1))
    for i in range(n):
        logX[i] = man.log(FM, X[i]).reshape(-1)
        
    cpca = Complex_PCA(n_components=m)
    cpca.fit(logX)
    #logX_trans = cpca.inverse_transform(logX_trans)
    #logX_trans = logX_trans.reshape(X.shape)
    
    #X_trans = np.zeros(X.shape, dtype = X.dtype)
    #for i in range(n):
    #    X_trans[i] = man.exp(FM, logX_trans[i])
    
    #var_ratio = np.sum(stds[0:m]**2)/np.sum(stds**2)
    #return var_ratio, X_trans
    return cpca

def sPGA_complex(X, y, kernel_y, m, man):
    """
    tangent supervised PCA for complex manifold
    """
    n = X.shape[0]
    FM = compute_centroid(man, X)
    logX = np.zeros(X.shape, dtype = X.dtype)
    logX = logX.reshape((n, -1))
    for i in range(n):
        logX[i] = man.log(FM, X[i]).reshape(-1)
        
    kernel_y = pairwise_kernels(y, metric = delta)

    logX_mean = X.mean(axis=0).reshape(-1)
    logX_center = logX
    for i in range(n):
        logX_center[i] = logX[i] - logX_mean
        
    Q = (logX_center.conj().T).dot(kernel_y).dot(logX_center)/np.sqrt(n)
    lambdas, pcs = np.linalg.eigh(Q.real)
    
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    lambdas = lambdas[lambdas > 0]
    #print(np.trace(Q))
    #print(lambdas[0:m])
    
    pcs = pcs[:, indices]

    
    logX_trans = np.matmul(np.matmul(logX, pcs[0:m].T), pcs[0:m])
    logX_trans = logX_trans.reshape(X.shape)
    
    X_trans = np.zeros(X.shape, dtype = X.dtype)
    for i in range(n):
        X_trans[i] = man.exp(FM, logX_trans[i])
    
    var_ratio = np.sum(lambdas[0:m])/np.sum(lambdas)
    return var_ratio, X_trans