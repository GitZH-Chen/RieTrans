import math
import torch as th
import torch.nn as nn
from Gyrovector.SPD.sym_functionals import sym_sqrtm,sym_expm

class GyroTrans(nn.Module):
    def __init__(self,shape_in):
        """AIM GyroTrans"""
        super(__class__, self).__init__()
        self.shape_in = shape_in;
        self.W = nn.Parameter(th.zeros(*shape_in))
        self.init_parameter()

    def forward(self,X):
        W = sym_expm.apply(symmetrize_by_tril(self.W))
        W_sqrtm = sym_sqrtm.apply(W)
        X_new = W_sqrtm @ X @ W_sqrtm
        return X_new

    def init_parameter(self, factor=1):
        c_in, n_in, _, = self.shape_in
        dim_in = n_in * (n_in + 1) / 2
        bound = math.sqrt(factor / (c_in * dim_in))
        nn.init.normal_(self.W, std=bound)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape_in={self.shape_in})"

def symmetrize_by_tril(A):
    """"
    symmetrize A by the lower part of A, with [...,n,n]
    """
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    tmp_A_sym = str_tril_A + str_tril_A.transpose(-1, -2) + th.diag_embed(diag_A_vec)
    return tmp_A_sym