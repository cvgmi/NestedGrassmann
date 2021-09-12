import numpy as np

class Complex_PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        X -= self.mean_
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        
        components_ = Vh
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        
        self.components_ = components_[:self.n_components]
        self.explained_variance_ = explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        
    def transform(self, X):
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.conj().T)
        return X_transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
    