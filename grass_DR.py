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
    Px = torch.matmul(X, torch.matmul(torch.inverse(torch.matmul(X.conj().t(), X)), X.conj().t()))
    Py = torch.matmul(Y, torch.matmul(torch.inverse(torch.matmul(Y.conj().t(), Y)), Y.conj().t()))
    if torch.is_complex(X) or torch.is_complex(Y):
        P = Px - Py
        return torch.sqrt(torch.sum(torch.matmul(P,P.conj().t()))).real/np.sqrt(2)
    else:
        return torch.norm(Px - Py)/np.sqrt(2)
    
def ortho_complement(A):
    # A is a n x p real- or complex-valued matrix such that A^HA = I, p < n
    # return an n x (n-p) matrix A_perp such that [A A_perp] is orthogonal
    n, p = A.shape
    A_ext = np.eye(n)[:,0:(n-p)]
    tmp = np.hstack((A, A_ext))
    q, _ = np.linalg.qr(tmp)
    return(q[:,p:n])
    
    
def affnity_matrix(dist_m, y, v_w, v_b):
    N = dist_m.shape[0]
    affinity = np.eye(N)

    for i in range(N):
        for j in range(i):
            tmp1 = np.argsort(dist_m[i, y == y[i]])[v_w]
            tmp2 = np.argsort(dist_m[j, y == y[j]])[v_w]
            g_w = np.int((y[i] == y[j]) and (dist_m[i, j] < np.maximum(tmp1, tmp2)))
            tmp1 = np.argsort(dist_m[i, y != y[i]])[v_b-1]
            tmp2 = np.argsort(dist_m[j, y != y[j]])[v_b-1]
            g_b = np.int((y[i] != y[j]) and (dist_m[i, j] < np.maximum(tmp1, tmp2)))
            affinity[i, j] = g_w - g_b
            affinity[j, i] = affinity[i, j]
    
    return affinity
    
def NG_dr1(X, verbosity = 0):
    """
    X: array of N points on Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(n-1, p)) 
    where X_hat_i = A^T X_i, A \in St(n, n-1)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
    cpx = np.iscomplex(X).any() # true if X is complex-valued

    if cpx:
        man = Product([ComplexGrassmann(n, 1), Euclidean(p, 2)])
        
    else:
        man = Product([Grassmann(n, 1), Euclidean(p)])
    
    X_ = torch.from_numpy(X)
    
    @pymanopt.function.PyTorch
    def cost(v, b):
        vvT = torch.matmul(v, v.conj().t()) # n x n
        if cpx:
            b_ = b[:,0] + b[:,1]*1j
            b_ = torch.unsqueeze(b_, axis=1)
        else:
            b_ = torch.unsqueeze(b, axis=1)
        vbt = torch.matmul(v, b_.t()) # n x p
        IvvT = torch.eye(n, dtype=X_.dtype) - vvT
        d2 = 0
        for i in range(N):
            d2 = d2 + dist_proj(X_[i], torch.matmul(IvvT, X_[i]) + vbt)**2/N
            #d2 = d2 + dist_proj(X_[i], torch.matmul(AAT, X_[i]))**2/N
        return d2
    
    solver = SteepestDescent()
    problem = Problem(manifold=man, cost=cost, verbosity=verbosity)
    theta = solver.solve(problem)
    v = theta[0]
    b_ = theta[1]
    
    if cpx:
        b = b_[:,0] + b_[:,1]*1j
        b = np.expand_dims(b, axis=1)
    else:
        b = np.expand_dims(b_, axis=1)
    
    R = ortho_complement(v)

    #tmp = np.array([A.T for i in range(N)])
    tmp = np.array([R.conj().T for i in range(N)])
    X_low = multiprod(tmp, X)
    X_low = np.array([qr(X_low[i])[0] for i in range(N)])

    return X_low, R, v, b
    

def NG_dr(X, m, verbosity=0, *args, **kwargs):
    """
    X: array of N points on Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, R \in St(n, m)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
    cpx = np.iscomplex(X).any() # true if X is complex-valued

    if cpx:
        man = Product([ComplexGrassmann(n, m), Euclidean(n, p, 2)])
        
    else:
        man = Product([Grassmann(n, m), Euclidean(n, p)])
    
    X_ = torch.from_numpy(X)
    
    @pymanopt.function.PyTorch
    def cost(A, B):
        AAT = torch.matmul(A, A.conj().t()) # n x n
        if cpx:
            B_ = B[:,:,0] + B[:,:,1]*1j
        else:
            B_ = B
        IAATB = torch.matmul(torch.eye(n, dtype=X_.dtype) - AAT, B_) # n x p
        d2 = 0
        for i in range(N):
            d2 = d2 + dist_proj(X_[i], torch.matmul(AAT, X_[i]) + IAATB)**2/N
            #d2 = d2 + dist_proj(X_[i], torch.matmul(AAT, X_[i]))**2/N
        return d2

    #solver = ConjugateGradient()
    solver = SteepestDescent()
    problem = Problem(manifold=man, cost=cost, verbosity=verbosity)
    theta = solver.solve(problem)
    A = theta[0]
    B = theta[1]
    
    if cpx:
        B_ = B[:,:,0] + B[:,:,1]*1j
    else:
        B_ = B

    #tmp = np.array([A.T for i in range(N)])
    tmp = np.array([A.conj().T for i in range(N)])
    X_low = multiprod(tmp, X)
    X_low = np.array([qr(X_low[i])[0] for i in range(N)])

    return X_low, A, B_


def NG_sdr(X, y, m, v_w = 5, v_b = 5, verbosity=0, *args, **kwargs):
    """
    X: array of N points on complex Gr(n, p); N x n x p array
    aim to represent X by X_hat (N points on Gr(m, p), m < n) 
    where X_hat_i = R^T X_i, W \in St(n, m)
    minimizing the projection error (using projection F-norm)
    """
    N, n, p = X.shape
    cpx = np.iscomplex(X).any() # true if X is complex-valued
    if cpx:
        gr = ComplexGrassmann(n, p)
        man = ComplexGrassmann(n, m)
    else:
        gr = Grassmann(n, p)
        man = Grassmann(n, m)
    
    # distance matrix
    dist_m = np.zeros((N, N))

    for i in range(N):
        for j in range(i):
            dist_m[i, j] = gr.dist(X[i], X[j])
            dist_m[j, i] = dist_m[i, j]
    
    # affinity matrix
    affinity = affinity_matrix(dist_m, y, v_w, v_b)

            
    X_ = torch.from_numpy(X)
    affinity_ = torch.from_numpy(affinity)
    
    @pymanopt.function.PyTorch
    def cost(A):
        dm = torch.zeros((N, N))
        for i in range(N):
            for j in range(i):
                dm[i, j] = dist_proj(torch.matmul(A.conj().t(), X_[i]), torch.matmul(A.conj().t(), X_[j]))**2
                #dm[i, j] = gr_low.dist(X_proj[i], X_proj[j])**2
                dm[j, i] = dm[i, j]
    
        d2 = torch.mean(affinity_*dm)   
        return d2

    # solver = ConjugateGradient()
    solver = SteepestDescent()
    problem = Problem(manifold=man, cost=cost, verbosity=verbosity)
    A = solver.solve(problem)

    tmp = np.array([A.conj().T for i in range(N)]) # N x m x n
    X_low = multiprod(tmp, X) # N x m x p
    X_low = np.array([qr(X_low[i])[0] for i in range(N)])
    
    return X_low, A



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

    
