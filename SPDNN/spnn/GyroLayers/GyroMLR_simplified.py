import torch as th
import torch.nn as nn

from Gyrovector.SPD.spd_gyro import SPDLogEuclideanMetric,SPDAffineInvariantMetric,SPDLogCholeskyMetric
from Gyrovector.SPD.sym_functionals import sym_expm

class GyroMLR_simplified(nn.Module):
    """Implement the simplified GyroSPD++ MLR.

    For the GyroSPD++-AIM (AI-LE [1]), GyroSPD++-LEM (LE-LE [1]), and GyroSPD++-LCM (LC-LC [1]) in our paper, the matrix functions in the
    original GyroSPD++ MLR cancel algebraically, yielding the simplified expressions implemented here.

    Input X: (N, shape_in) SPD matrices.
    Output: (N, class_num) logits.
    SPD parameters have shape (class_num, *shape_in).

    Reference:
        [1] X. S. Nguyen, S. Yang, and A. Histace, "Matrix Manifold Neural Networks++," ICLR, 2024.
    """

    def __init__(self,shape_in,class_num,metric='LEM'):
        super(__class__, self).__init__()
        if metric == 'AIM':
            raise NotImplementedError('GyroMLR_simplified does not support an AIM classifier.')
        self.shape_in = shape_in;self.class_num = class_num;
        self.metric = metric;
        self.init_parameter()
        self.getmetric()

    def forward(self,X):
        if self.metric=='LEM':
            P = symmetrize_by_tril(self.P)
            W = symmetrize_by_tril(self.W)
            vec = self.spd.inner_product(X.unsqueeze(1) - P, W).sum(-1)
        elif self.metric=='LCM':
            P_phi = self.P.tril()
            W_phi = self.W.tril()
            vec = self.spd.inner_product(X.unsqueeze(1) - P_phi, W_phi).sum(-1)

        return vec

    def init_parameter(self, factor=1):
        self.param_shape = (self.class_num, *self.shape_in)
        self.P = nn.Parameter(th.zeros(*self.param_shape))  # (c, ..., n, n)
        W = th.randn_like(self.P) * 1e-3
        self.W = nn.Parameter(W)

    def getmetric(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
        }

        if self.metric in classes:
            self.spd = classes[self.metric](n=self.shape_in[-1])
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(shape_in={self.shape_in},class_num={self.class_num},metric={self.metric})"

def symmetrize_by_tril(A):
    """"
    symmetrize A by the lower part of A, with [...,n,n]
    """
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    tmp_A_sym = str_tril_A + str_tril_A.transpose(-1, -2) + th.diag_embed(diag_A_vec)
    return tmp_A_sym
