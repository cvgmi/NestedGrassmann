# Nested Grassmanns for Dimensionality Reduction

## Reference:

If you use this code, please cite the following paper that contains the theory,
the algorithm, and some experiments showing the performance of this model.

Chun-Hao Yang and Baba C. Vemuri (2021), *Nested Grassmanns for Dimensionality
Reduction with Applications to Shape Analysis* in Proceedings of the International Conference on Information Processing in Medical Imaging, 2021, June 28-30, Bornholm, Denmark, pages 136-149.

Yang, C. H., & Vemuri, B. C. (2022). *Nested Grassmannians for Dimensionality Reduction with Applications.* Machine Learning for Biomedical Imaging, 1(IPMI 2021 special issue), 1-10.

This software was developed as part of research in part funded by NSF IIS-1724174 to Baba C. Vemuri, Department of CISE, Univ. of Florida.

## Dependencies

[Pymanopt](https://github.com/pymanopt/pymanopt) 0.2.4 

[PyTorch](https://pytorch.org/)

## Usage

Following the convention of pymanopt, $N$ points on $\text{Gr}(p,n)$ are
represented as an $(N,n,p)$ numpy array. The function `DR_proj` takes an $(N, n,
p)$ numpy array and an integer $m < n$ as inputs and projects the $N$ points on
$\text{Gr}(p, n)$ to $\text{Gr}(p, m)$. The outputs are the ratio of explained
variance, the projected points $\hat{X}$ which is an $(N, m ,p)$ numpy array,
and the projection parameters $A$ and $B$. 

An example can be found in `example.ipynb`.

