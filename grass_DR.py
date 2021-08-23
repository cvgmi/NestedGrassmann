"""
Nested Grassmanns for Dimensionality Reduction
"""
from pymanopt.manifolds import Grassmann, ComplexGrassmann, Stiefel, Product, Euclidean
import numpy as np
from numpy.linalg import inv, det, qr, norm, svd
from scipy.linalg import logm, expm, sqrtm
from numpy import log, sqrt
from pymanopt import Problem
from pymanopt.tools.multi import multiprod, multitransp, multihconj
from compute_centroid import *
from pymanopt.solvers import SteepestDescent, ConjugateGradient
import torch

def var(man, X, M):
    d = 0
    N = X.shape[0]
    for i in range(N):
        d += man.dist(X[i], M)**2
    return d

def dist_proj(X, Y):
    Px = torch.matmul(X, torch.matmul(torch.inverse(torch.matmul(X.t(), X)), X.t()))
    Py = torch.matmul(Y, torch.matmul(torch.inverse(torch.matmul(Y.t(), Y)), Y.t()))
    return torch.norm(Px - Py)/np.sqrt(2)

def DR_proj(X, m, verbosity=0, *args, **kwargs):
    """
    X: array of N points on Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, W \in St(n, m)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
    
    gr = Grassmann(n, p)
    FM_all = compute_centroid(gr, X)
    v = var(gr, X, FM_all)/N
    
    gr = Grassmann(m, p)
    gr_map = Grassmann(n, m)
    man = Product([Grassmann(n, m), Euclidean(n, p)])
    X_ = torch.from_numpy(X)
    
    @pymanopt.function.PyTorch
    def cost(A, B):
        AAT = torch.matmul(A, A.t()) # n x n
        IAATB = torch.matmul(torch.eye(n) - AAT, B) # n x p
        d2 = 0
        for i in range(N):
            d2 = d2 + dist_proj(X_[i], torch.matmul(AAT, X_[i]) + IAATB)**2/N
        return d2

    solver = ConjugateGradient()
    problem = Problem(manifold=man, cost=cost, verbosity=verbosity)
    theta = solver.solve(problem)
    A = theta[0]
    B = theta[1]

    tmp1 = np.array([A.T for i in range(N)])
    X_low = multiprod(tmp1, X)
    X_low = np.array([qr(X_low[i])[0] for i in range(N)])

    M_hat = compute_centroid(gr, X_low)
    v_hat = var(gr, X_low, M_hat)/N
    var_ratio = v_hat/v
    return var_ratio, X_low, A, B

def DR_proj_complex(X, m, verbosity=0, *args, **kwargs):
    """
    X: array of N points on complex Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = A^T X_i - A^T B, A \in St(n, m), and B \in R^(n x p)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
        
    Cgr = ComplexGrassmann(n, p)
    FM_all = compute_centroid(Cgr, X)
    v = var(Cgr, X, FM_all)/N
    
    Cgr = ComplexGrassmann(m, p)
    Cgr_map = ComplexGrassmann(n, m)
    man = Product([ComplexGrassmann(n, m), Euclidean(n, p, 2)])
    X_ = torch.from_numpy(X)
    
    @pymanopt.function.PyTorch
    def cost(A, B):
        AAT = torch.matmul(A, A.conj().t()) # n x n
        B_cpx = B[:,:,0] + B[:,:,1]*1j
        IAATB = torch.matmul(torch.eye(n, dtype=torch.complex128) - AAT, B_cpx) # n x p
        d2 = 0
        for i in range(N):
            d2 = d2 + dist_proj(X_[i], torch.matmul(AAT, X_[i]) + IAATB)**2/N
        return d2
    

    solver = ConjugateGradient()
    problem = Problem(manifold=man, cost=cost, verbosity=verbosity)
    theta = solver.solve(problem)
    A = theta[0]
    B = theta[1]
    B_cpx = B[:,:,0] + B[:,:,1]*1j
    
    tmp = np.array([multihconj(A) for i in range(N)]) # N x m x n
    X_low = multiprod(tmp, X) # N x m x p
    #X_low = X_low/np.expand_dims(np.linalg.norm(X_low, axis=1), axis = 2)
    X_low = np.array([qr(X_low[i])[0] for i in range(N)])

    M_hat = compute_centroid(Cgr, X_low)
    v_hat = var(Cgr, X_low, M_hat)/N
    var_ratio = v_hat/v
    return var_ratio, X_low, A, B_cpx

def DR_supervised_proj_complex(X, m, verbosity=0, *args, **kwargs):
    """
    X: array of N points on complex Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, W \in St(n, m)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
        
    Cgr = ComplexGrassmann(n, p)
    FM_all = compute_centroid(Cgr, X)
    v = var(Cgr, X, FM_all)/N
    
    Cgr = ComplexGrassmann(m, p)
    Cgr_map = ComplexGrassmann(n, m)
    XXT = multiprod(X, multihconj(X)) # N x n x n
    meanXXT = np.mean(XXT, axis = 0) # n x n
    
    @pymanopt.function.Callable
    def cost(Q):
        QQ = np.matmul(Q, multihconj(Q)) # n x n
        d2 = p - np.trace(np.matmul(meanXXT, QQ))
        return d2.real
    
    @pymanopt.function.Callable
    def egrad(Q):
        eg = -2  * np.matmul(meanXXT, Q)
        return eg

    # solver = ConjugateGradient()
    solver = SteepestDescent(*args, **kwargs)
    problem = Problem(manifold=Cgr_map, cost=cost, egrad=egrad, verbosity=verbosity)
    Q_proj = solver.solve(problem) # n x m

    tmp = np.array([multihconj(Q_proj) for i in range(N)]) # N x m x n
    X_low = multiprod(tmp, X) # N x m x p
    X_low = X_low/np.expand_dims(np.linalg.norm(X_low, axis=1), axis = 2)

    M_hat = compute_centroid(Cgr, X_low)
    v_hat = var(Cgr, X_low, M_hat)/N
    var_ratio = v_hat/v
    return var_ratio, X_low, Q_proj



def DR_geod(X, m, verbosity=0):
    """ 
    X: array of N points on Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, W \in St(n, m)
    minimizing the projection error (using geodesic distance)
    """
    N, n, p = X.shape
    
    gr = Grassmann(n, p)
    FM_all = compute_centroid(gr, X)
    v = var(gr, X, FM_all)/N
    
    gr = Grassmann(n, p, N)
    gr_low = Grassmann(m, p)
    gr_map = Grassmann(n, m) # n x m
    XXT = multiprod(X, multitransp(X)) # N x n x n
    
    @pymanopt.function.Callable
    def cost(Q):
        tmp = np.array([np.matmul(Q, Q.T) for i in range(N)]) # N x n x n
        new_X = multiprod(tmp, X) # N x n x p
        q = np.array([qr(new_X[i])[0] for i in range(N)])
        d2 = gr.dist(X, q)**2
        return d2/N
    
    @pymanopt.function.Callable
    def egrad(Q):
        """
        need to be fixed
        """
        QQ = np.matmul(Q, Q.T)
        tmp = np.array([QQ for i in range(N)])
        XQQX = multiprod(multiprod(multitransp(X), tmp), X)
        lam, V = np.linalg.eigh(XQQX)
        theta = np.arccos(np.minimum(np.sqrt(lam), 1-1e-5))
        d = -2*theta/(np.cos(theta)*np.sin(theta))
        Sig = np.array([np.diag(dd) for dd in d])
        XV = multiprod(X,V)
        eg = multiprod(XV, multiprod(Sig, multitransp(XV)))
        eg = np.mean(eg, axis = 0)
        eg = np.matmul(eg, Q)
        return eg

    def egrad_num(R, eps = 1e-8):
        """
        compute egrad numerically
        """
        g = np.zeros(R.shape)
        for i in range(n):
            for j in range(m):
                R1 = R.copy()
                R2 = R.copy()
                R1[i,j] += eps
                R2[i,j] -= eps
                g[i,j] = (cost(R1) - cost(R2))/(2*eps)
        return g

    # solver = ConjugateGradient()
    solver = SteepestDescent()
    problem = Problem(manifold=gr_map, cost=cost, egrad=egrad, verbosity=verbosity)
    Q_proj = solver.solve(problem)

    tmp = np.array([Q_proj.T for i in range(N)])
    X_low = multiprod(tmp, X)
    X_low = X_low/np.expand_dims(np.linalg.norm(X_low, axis=1), axis = 2)

    M_hat = compute_centroid(gr_low, X_low)
    v_hat = var(gr_low, X_low, M_hat)/N
    var_ratio = v_hat/v
    return var_ratio, X_low, Q_proj


def DR_geod_complex(X, m, verbosity=0):
    """ 
    X: array of N points on Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, W \in St(n, m)
    minimizing the projection error (using geodesic distance)
    """
    N, n, p = X.shape
    Cgr = ComplexGrassmann(n, p, N)
    Cgr_low = Grassmann(m, p)
    Cgr_map = ComplexGrassmann(n, m) # n x m
    XXT = multiprod(X, multihconj(X))
    
    @pymanopt.function.Callable
    def cost(Q):
        tmp = np.array([np.matmul(Q, Q.T) for i in range(N)]) # N x n x n
        new_X = multiprod(tmp, X) # N x n x p
        q = np.array([qr(new_X[i])[0] for i in range(N)])
        d2 = Cgr.dist(X, q)**2
        return d2/N
    
    @pymanopt.function.Callable
    def egrad(Q):
        """
        need to be fixed
        """
        QQ = np.matmul(Q, multihconj(Q))
        tmp = np.array([QQ for i in range(N)])
        XQQX = multiprod(multiprod(multihconj(X), tmp), X)
        lam, V = np.linalg.eigh(XQQX)
        theta = np.arccos(np.sqrt(lam))
        d = -2*theta/(np.cos(theta)*np.sin(theta))
        Sig = np.array([np.diag(dd) for dd in d])
        XV = multiprod(X,V)
        eg = multiprod(XV, multiprod(Sig, multitransp(XV.conj())))
        eg = np.mean(eg, axis = 0)
        eg = np.matmul(eg, Q)
        return eg

    def egrad_num(R, eps = 1e-8+1e-8j):
        """
        compute egrad numerically
        """
        g = np.zeros(R.shape, dtype=np.complex128)
        for i in range(n):
            for j in range(m):
                R1 = R.copy()
                R2 = R.copy()
                R1[i,j] += eps
                R2[i,j] -= eps
                g[i,j] = (cost(R1) - cost(R2))/(2*eps)
        return g

    # solver = ConjugateGradient()
    solver = SteepestDescent()
    problem = Problem(manifold=Cst, cost=cost, egrad=egrad, verbosity=verbosity)
    Q_proj = solver.solve(problem)

    tmp = np.array([multihconj(Q_proj) for i in range(N)])
    X_low = multiprod(tmp, X)
    X_low = X_low/np.expand_dims(np.linalg.norm(X_low, axis=1), axis = 2)

    M_hat = compute_centroid(Cgr_low, X_low)
    v_hat = var(Cgr_low, X_low, M_hat)/N
    var_ratio = v_hat/v
    return var_ratio, X_low, Q_proj


def pairmean(self, X, Y):
    return self.exp(X, self.log(X, Y)/2)

Grassmann.pairmean = pairmean
ComplexGrassmann.pairmean = pairmean
    



if __name__ == '__main__':
    N = 100
    n = 4
    m = 2
    p = 1
    sig = 0.01
    gr = Grassmann(n, p)
    W, X = data_gen(N, n, m, p, sig)
    # P_W = np.matmul(W, W.T)
    M, v = compute_centroid(gr, X)
    Q_proj = DR_proj(X, m)
    # P_proj = np.matmul(Q_proj, Q_proj.T)
    Q_geod = DR_geod(X, m)
    # P_geod = np.matmul(Q_geod, Q_geod.T)
    tmp = np.array([Q_proj.T for i in range(N)])
    X_hat = multiprod(tmp, X)

    gr_hat = Grassmann(m, p)
    M_hat, v_hat = compute_centroid(gr_hat, X_hat)
    gr_map = Grassmann(n, m)
    print(gr_map.dist(W, Q_proj))
    print(gr_map.dist(W, Q_geod))

    
