import math
import torch as th
import torch.nn as nn

from Gyrovector.SPD.spd_gyro import SPDLogEuclideanMetric,SPDAffineInvariantMetric,SPDLogCholeskyMetric
from Gyrovector.SPD.sym_functionals import sym_expm

class GyroMLR(nn.Module):
    def __init__(self,shape_in,class_num,metric='LEM'):
        """
            Input X: (N,shape_in) SPD matrices
            Output P: (N,class_num) vectors
            SPD parameter of size (class_num, shape_in), where c denotes the number of classes
        """
        super(__class__, self).__init__()
        self.shape_in = shape_in;self.class_num = class_num;
        self.metric = metric;

        self.param_shape = (class_num, *shape_in)
        self.P = nn.Parameter(th.zeros(*self.param_shape))  # (c, ..., n, n)
        self.W = nn.Parameter(th.empty_like(self.P))
        self.init_parameter()
        self.get()

    def forward(self,X):
        P = sym_expm.apply(symmetrize_by_tril(self.P))
        W = sym_expm.apply(symmetrize_by_tril(self.W))
        X_new = self.spd.GyroMLR(X.unsqueeze(1), P, W)
        return X_new

    def get(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
        }

        if self.metric in classes:
            self.spd = classes[self.metric](n=self.shape_in[-1])
        else:
            raise NotImplementedError

    def init_parameter(self, factor=1):
        c_in, n_in, _, = self.shape_in
        dim_in = n_in * (n_in + 1) / 2
        bound = math.sqrt(factor / (c_in * dim_in))
        nn.init.normal_(self.W, std=bound)

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