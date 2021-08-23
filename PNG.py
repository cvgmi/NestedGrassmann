"""
Principal Nested Grassmanns
"""
from pymanopt.manifolds import Grassmann
import numpy as np
from compute_centroid import *
from grass_DR import *
from sklearn.decomposition import PCA

def ortho_complement(A):
    # A is a n x p matrix such that A^TA = I, p < n
    # return an n x (n-p) matrix A_perp such that [A A_perp] is orthogonal
    n, p = A.shape
    A_ext = np.eye(n)[:,0:p]
    tmp = np.hstack((A, A_ext))
    q, _ = np.linalg.qr(tmp)
    return(q[:,p:n])

def PNG(X, log=True):
    # Assuming X consists of N points on Gr(p, n), p < n
    # return an N x p(n-p) score array
    N, n, p = X.shape
    scores = np.zeros((N,int(p*(n-p))))
    scores[:] = np.NaN
    X_old = X.copy()
    
    # Gr(p, n) -> Gr(p, n-1) -> ... -> Gr(p, p+1)
    for i in range(n-1, p, -1):
        if log:
            print(f'Gr({p}, {i+1}) -> Gr({p}, {i})')

        _, X_new, A, B = DR_proj(X_old, i)
        A_perp = ortho_complement(A)[:,0]
        AAT = np.matmul(A, A.T) 
        IAATB = np.matmul(np.eye(X_old.shape[1]) - AAT, B)
        X_new_embedded = np.array([np.linalg.qr(np.matmul(A, X_new[i]) + IAATB)[0] for i in range(N)])
        
        # compute scores
        d = np.zeros(N)
        gr = Grassmann(X_old.shape[1], X_old.shape[2])
        for j in range(N):
            scores[j,((n-i-1)*p):(n-i)*p] = gr.dist(X_old[j], X_new_embedded[j]) * np.matmul(X_old[j].transpose(), A_perp)
        
        X_old = X_new
        
    if p > 1:
        
        # Gr(p, p+1) -> Gr(1, p+1)
        X_new = np.zeros((N, p+1, 1))
        if log:
            print(f'Gr({p}, {p+1}) -> Gr(1, {p+1})')
        for i in range(N):
            X_new[i] = ortho_complement(X_old[i])

        X_old = X_new

        # Gr(1, p+1) -> ... -> Gr(1,2)
        for i in range(p, 1, -1):
            if log:
                print(f'Gr(1, {i+1}) -> Gr(1, {i})')
            _, X_new, A, B = DR_proj(X_old, i)
            A_perp = ortho_complement(A)[:,0]
            AAT = np.matmul(A, A.T) 
            IAATB = np.matmul(np.eye(X_old.shape[1]) - AAT, B)
            X_new_embedded = np.array([np.linalg.qr(np.matmul(A, X_new[i]) + IAATB)[0] for i in range(N)])

            # compute scores
            d = np.zeros(N)
            gr = Grassmann(X_old.shape[1], X_old.shape[2])
            for j in range(N):
                scores[j,(n-p)*p-i] = gr.dist(X_old[j], X_new_embedded[j]) * np.matmul(X_old[j].transpose(), A_perp)

            X_old = X_new
    
    # Gr(1,2) -> NGM
    if log:
        print('Gr(1, 2) -> NGM')
    gr = Grassmann(2,1)
    NGM = compute_centroid(gr, X_new)
    for j in range(N):
        v_direction = gr.log(NGM, X_new[j])
        scores[j, (n-p)*p-1] = np.sign(v_direction[0]) * gr.dist(X_new[j], NGM) # signed distance

    return scores

if __name__ == '__main__':
    m = 6
    n = 10
    p = 2
    N = 15
    sig = 0
    gr_low = Grassmann(m, p, N)
    gr = Grassmann(n, p, N)
    gr_map = Grassmann(n, m)

    X_low = gr_low.rand() # N x m x p
    A = gr_map.rand() # n x m
    #B = np.random.normal(0, 0.1, (n, p)) # n x p
    B = np.zeros((n,p))
    AAT = np.matmul(A, A.T) 
    IAATB = np.matmul(np.eye(n) - AAT, B)
    X = np.array([np.linalg.qr(np.matmul(A, X_low[i]) + IAATB)[0] for i in range(N)]) # N x n x p
    X = gr.exp(X, sig * gr.randvec(X)) # perturb the emdedded X
    
    
    print(f'Examples: {N} points on Gr({p}, {m}) embedded in Gr({p}, {n})\n') 
    
    scores = PNG(X, log = True)
    
    print('\n')
    print('PCA of the scores\n')
    n_c = 5
    pca = PCA(n_components = n_c)
    pca.fit(scores)
    print(f'The ratios of expressed variance first {n_c} PCs.')
    print(pca.explained_variance_ratio_.round(2))
    print(f'The cumulative ratios of expressed variance first {n_c} PCs.')
    print(np.cumsum(pca.explained_variance_ratio_).round(2))