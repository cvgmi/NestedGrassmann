"""
multi-tools based on torch
ref: https://github.com/pymanopt/pymanopt/blob/master/pymanopt/tools/multi.py
"""
import torch

def multiprod_torch(A, B):
    if A.ndim == 2:
        return torch.matmul(A, B)

    return torch.einsum('ijk,ikl->ijl', A, B)

def multitransp_torch(A):
    if A.ndim == 2:
        return A.T
    return torch.transpose(A, 2, 1)


def multihconj_torch(A):
    return torch.conj(multitransp_torch(A))
