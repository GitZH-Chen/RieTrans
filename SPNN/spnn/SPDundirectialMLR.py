"""Product-SPD fully connected classification layer used by SPNN."""

import math
import torch as th
import torch.nn as nn

from geometry.spd.spd_matrices import SPDLogEuclideanMetric,SPDLogCholeskyMetric,SPDAffineInvariantMetric,SPDBuresWassersteinMetric,SPDEuclideanMetric,\
    tril_param_metrics,bi_param_metrics,single_param_metrics,tril_metrics

class SPDundirectialMLR(nn.Module):
    """Classify multi-channel SPD features with the proposed SPD FC layer."""

    def __init__(self,shape_in,c,is_phi=True,
                 metric='LEM',power=1.0):
        """
            Input X: (N,h,n,n) SPD matrices
            Output P: (N,c) vectors
            Sym parameters (c,h,dim) dim=n(n+1)/2
            Sym parameters (c,h,1,1)
        """
        super(__class__, self).__init__()
        self.shape_in = shape_in; self.c=c
        self.is_phi = is_phi
        self.metric = metric;self.power = power

        self.getmetric()
        self.init_parameter()

    def forward(self,X):
        Z = self.Z_vec if self.metric in tril_metrics else self.vec2sym(self.Z_vec)
        return self.spd.undirectional_RMLR(X.unsqueeze(1), Z, self.bias, is_phi=self.is_phi)

    def getmetric(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
            "BWM": SPDBuresWassersteinMetric,
            "PEM": SPDEuclideanMetric,
        }
        n = self.shape_in[-1]
        if self.metric in tril_param_metrics:
            self.spd = classes[self.metric](n=n, power=self.power)
        elif self.metric in bi_param_metrics:
            self.spd = classes[self.metric](n=n)
        elif self.metric in single_param_metrics:
            self.spd = classes[self.metric](n=n, power=self.power)
        else:
            raise NotImplementedError

    def init_parameter(self, factor=1):
        c_in, n_in, _, = self.shape_in
        dim_in = n_in * (n_in + 1) / 2
        bound = math.sqrt(factor / (c_in * dim_in))

        Z_vec = nn.init.normal_(th.zeros(self.c, c_in, int(dim_in)), std=bound)
        self.Z_vec = nn.Parameter(Z_vec)

        if self.metric in tril_metrics:
            bias_weight = th.zeros(self.c, c_in, 1)
        else:
            bias_weight = th.zeros(self.c, c_in, 1, 1)
        self.bias = nn.Parameter(bias_weight)

    def vec2sym(self, Z_vec):
        """
        Reconstruct the symmetric matrices from their lower triangular part.
        Z_vec: shape [..., vec_len], where vec_len = n(n+1)/2
        Returns: shape [..., n, n]
        """
        *batch_dims, vec_len = Z_vec.shape
        n = int((-1 + math.sqrt(1 + 8 * vec_len)) / 2)  # Recover the original n

        Z_sym = th.zeros(*batch_dims, n, n, dtype=Z_vec.dtype, device=Z_vec.device)
        tril_indices = th.tril_indices(n, n)

        Z_sym[..., tril_indices[0], tril_indices[1]] = Z_vec
        Z_sym = Z_sym + Z_sym.tril(-1).transpose(-2, -1)

        return Z_sym

    def __repr__(self):
        return f"{self.__class__.__name__}(shape_in={self.shape_in},c={self.c}," \
               f"is_phi={self.is_phi}," \
               f"metric={self.metric},power={self.power})"
