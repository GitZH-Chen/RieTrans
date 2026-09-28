import math
import torch as th
import torch.nn as nn

from geometry.spd.spd_matrices import SPDLogEuclideanMetric,SPDLogCholeskyMetric,SPDAffineInvariantMetric,SPDBuresWassersteinMetric,SPDEuclideanMetric,\
    tril_metrics

class SPDLinear(nn.Module):
    """SPD linear layer"""

    def __init__(self,shape_in,shape_out,is_phi_inv=True,
                 metric='LEM',power=1.0):
        """
            Input X: (bs,c1,n1,n1) SPD matrices
            Output P: (bs,c2,n2,n2) vectors
            Paramters:
                Z_vec: (c2,dim2,c1,dim1) with dimi=ni(ni+1)/2
                bias: (c2,dim2,c1,1,1)
        """
        super(__class__, self).__init__()
        self.shape_in=shape_in;self.shape_out=shape_out;self.n=self.shape_in[-1]
        self.is_phi_inv = is_phi_inv
        self.metric = metric
        if self.metric == 'PEM':
            self.power = power
        elif self.metric == 'BWM':
            # The 2theta-BWM family recovers standard BWM at theta=0.5; see "RMLR: Extending Multinomial Logistic Regression into General Geometries" (NeurIPS 2024).
            self.power = 0.5
        self.getmetric()
        self.init_parameter()

    def forward(self,X):
        Z = self.Z_vec if self.metric in tril_metrics else self.vec2sym(self.Z_vec)
        input = X.unsqueeze(-4).unsqueeze(-4)
        bias = self.bias
        return self.spd.SPDLinear(input, Z, bias, is_phi_inv=self.is_phi_inv)

    def getmetric(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
            "BWM": SPDBuresWassersteinMetric,
            "PEM": SPDEuclideanMetric,
        }

        if self.metric == 'PEM':
            self.spd = classes[self.metric](n=self.n,power=self.power)
        elif self.metric == 'BWM':
            self.spd = classes[self.metric](n=self.n,power=self.power)
        elif self.metric in classes:
            self.spd = classes[self.metric](n=self.n)
        else:
            raise NotImplementedError
    def init_parameter(self,factor=0.5):
        c_in,n_in,_, = self.shape_in
        c_out, n_out, _, = self.shape_out
        dim_out=n_out*(n_out+1)/2
        dim_in = n_in * (n_in + 1) / 2

        if self.metric in ['AIM','BWM']:
            # Small initialization improves numerical stability for AIM and BWM.
            Z_vec = th.randn(c_out,int(dim_out), c_in, int(dim_in)) * 1e-3
            self.Z_vec = nn.Parameter(Z_vec)

            if self.metric in tril_metrics:
                bias_weight = th.randn(c_out, int(dim_out), c_in, 1)* 1e-3
            else:
                bias_weight = th.randn(c_out, int(dim_out), c_in, 1, 1)* 1e-3
            self.bias = nn.Parameter(bias_weight)
        else:
            bound = math.sqrt(factor / (c_in * dim_in * dim_out * c_out))
            Z_vec = nn.init.normal_(th.zeros(c_out,int(dim_out), c_in, int(dim_in)), std=bound)
            self.Z_vec = nn.Parameter(Z_vec)

            if self.metric in tril_metrics:
                bias_weight = nn.init.normal_(th.zeros(c_out, int(dim_out), c_in, 1), std=bound)
            else:
                bias_weight = nn.init.normal_(th.zeros(c_out, int(dim_out), c_in, 1, 1), std=bound)
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
        description = f"{self.__class__.__name__}(shape_in={self.shape_in},shape_out={self.shape_out},is_phi_inv={self.is_phi_inv},metric={self.metric}"
        if self.metric == 'PEM':
            description += f",power={self.power}"
        return description + ")"
