import torch as th
import torch.nn as nn

from Gyrovector.SPD.spd_gyro import SPDAffineInvariantMetric,SPDLogEuclideanMetric,SPDLogCholeskyMetric
from Gyrovector.SPD.sym_functionals import sym_expm,sym_logm,sym_invsqrtm,sym_expm

class GyroLinear(nn.Module):
    def __init__(self,shape_in,shape_out,metric='AIM'):
        """
            Input X: (bs,c1,n1,n1) SPD matrices
            Output P: (bs,c2,n2,n2) vectors
            Paramters:
                W: (c2,dim2,c1,n1,n1) with dim1=n1(n1+1)/2
                bias: (c2,dim2,c1,n1,n1)
        """
        super(__class__, self).__init__()
        self.shape_in = shape_in;self.shape_out = shape_out;self.metric=metric
        self.init_parameter()
        self.getmetric()

    def forward(self,X):
        S = X.unsqueeze(-4).unsqueeze(-4)

        if self.metric=='AIM':
            # always collapse
            P = sym_expm.apply(symmetrize_by_tril(self.P))

            P_invsqrtm = sym_invsqrtm.apply(P)
            item1 = sym_logm.apply(P_invsqrtm @ S @ P_invsqrtm)
            item2 = symmetrize_by_tril(self.W)
            vec = self.spd.inner_product(item1,item2).sum(-1)
            X_new = self.spd.v2V(vec)

        elif self.metric=='LEM':
            S_phi = sym_logm.apply(S)
            P_phi = symmetrize_by_tril(self.P)
            W_phi = symmetrize_by_tril(self.W)

            vec = self.spd.inner_product(S_phi - P_phi, W_phi).sum(-1)
            X_new = self.spd.v2V(vec)
        elif self.metric=='LCM':
            S_phi = self.spd.phi(S)
            P_phi = self.P.tril()
            W_phi = self.W.tril()
            vec = self.spd.inner_product(S_phi - P_phi, W_phi).sum(-1)
            X_new = self.spd.v2V(vec)

        return X_new

    def init_parameter(self,factor=1):
        c_in,n_in,_, = self.shape_in
        c_out, n_out, _, = self.shape_out
        dim_out=n_out*(n_out+1)/2
        W = nn.init.normal_(th.zeros(c_out, int(dim_out), c_in, n_in, n_in)) * 1e-3
        self.W = nn.Parameter(W)
        P_weight = th.zeros_like(W)
        self.P = nn.Parameter(P_weight)

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
        return f"{self.__class__.__name__}(shape_in={self.shape_in},shape_out={self.shape_out}," \
               f"metric={self.metric})"

def symmetrize_by_tril(A):
    """"
    symmetrize A by the lower part of A, with [...,n,n]
    """
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    tmp_A_sym = str_tril_A + str_tril_A.transpose(-1, -2) + th.diag_embed(diag_A_vec)
    return tmp_A_sym
