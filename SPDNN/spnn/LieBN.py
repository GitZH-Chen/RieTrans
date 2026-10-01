"""SPD Lie-group batch-normalization implementation used by the baselines."""

import torch as th
import torch.nn as nn
from geometry.spd import sym_functionals
import geoopt
from geoopt.manifolds import SymmetricPositiveDefinite

dtype=th.double
device=th.device('cpu')

class LieBatchNormSPD(nn.Module):
    """Batch-normalize SPD matrices under the configured Lie-group metric.

    Inputs and outputs have shape ``(..., n, n)``. The learnable parameters are
    an SPD bias matrix and a scalar dispersion.
    """
    def __init__(self,n: int ,metric: str ='AIM',power: float =1.,ddevice='cpu',momentum=0.1):
        super(__class__,self).__init__()
        self.momentum=momentum;self.n=n; self.ddevice=ddevice;
        self.metric = metric;self.power=th.tensor(power,dtype=dtype)
        self.running_mean=th.eye(n,dtype=dtype)
        self.weight = geoopt.ManifoldParameter(th.eye(n, n, dtype=dtype, device=ddevice),
                                 manifold=SymmetricPositiveDefinite())
        self.eps=1e-05;
        self.running_var=th.ones(1,dtype=dtype);
        self.shift = nn.Parameter(th.ones(1, dtype=dtype))

        if metric != "AIM" and metric != "LCM" and metric != "LEM" :
            raise Exception('unknown metric {}'.format(metric))

    def forward(self,X):
        #deformation
        X_deformed = self.deformation(X)
        weight = self.deformation(self.weight)

        if(self.training):
            # centering
            batch_mean = self.cal_geom_mean(X_deformed)
            X_centered = self.Left_translation(X_deformed,batch_mean,is_inverse=True)
            # scaling and shifting
            batch_var = self.cal_geom_var(X_centered)
            X_scaled = self.scaling(X_centered, batch_var, self.shift)
            self.updating_running_statistics(batch_mean, batch_var)

        else:
            # centering, scaling and shifting
            X_centered = self.Left_translation(X_deformed, self.running_mean, is_inverse=True)
            X_scaled = self.scaling(X_centered, self.running_var, self.shift)
        #biasing
        X_normalized = self.Left_translation(X_scaled, weight, is_inverse=False)
        # inv_deformation
        X_new = self.inv_deformation(X_normalized)

        return X_new
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},momentum={self.momentum}," \
               f"power={self.power})"

    def spd_power(self,X):
        if self.power == 1.:
            X_power = X
        else:
            X_power = sym_functionals.sym_powm.apply(X, self.power)
        return X_power

    def inv_power(self,X):
        if self.power == 1.:
            X_power = X
        else:
            X_power = sym_functionals.sym_powm.apply(X, 1/self.power)
        return X_power

    def deformation(self,X):

        if self.metric=='AIM':
            X_deformed = self.spd_power(X)
        elif self.metric == 'LEM':
            X_deformed = sym_functionals.sym_logm.apply(X)
        elif self.metric == 'LCM':
            X_power = self.spd_power(X)
            L = th.linalg.cholesky(X_power)
            diag_part = th.diag_embed(th.log(th.diagonal(L, dim1=-2, dim2=-1)))
            X_deformed = L.tril(-1) + diag_part

        return X_deformed

    def inv_deformation(self,X):
        if self.metric=='AIM':
            X_inv_deformed = self.inv_power(X)
        elif self.metric == 'LEM':
            X_inv_deformed = sym_functionals.sym_expm.apply(X)
        elif self.metric == 'LCM':
            Cho = X.tril(-1) + th.diag_embed(th.exp(th.diagonal(X, dim1=-2, dim2=-1)))
            spd = Cho.matmul(Cho.transpose(-1,-2))
            X_inv_deformed = self.inv_power(spd)
        return X_inv_deformed
    def cal_geom_mean(self, X):
        """Frechet mean"""
        if self.metric == 'AIM':
            mean = self.BaryGeom(X.detach())
        elif self.metric == 'LEM' or self.metric == 'LCM':
            mean = X.detach().mean(dim=0, keepdim=True)

        return mean
    def cal_geom_var(self, X):
        """Frechet variance"""
        spd = X.detach()
        if self.metric == 'AIM':
            dists = th.linalg.matrix_norm(sym_functionals.sym_logm.apply(spd)).square()
            var = dists.mean()

        elif self.metric == 'LEM' or self.metric == 'LCM':
            dists = th.linalg.matrix_norm(spd)
            var = dists.mean()

        if self.metric == 'AIM' or self.metric == 'LCM':
            var_final = var * (1 / (self.power ** 2))
        else:
            var_final=var
        return var_final.unsqueeze(dim=0)
    def Left_translation(self, X, P, is_inverse):
        """Left translation by P"""
        if self.metric == 'AIM':
            L = th.linalg.cholesky(P);
            if is_inverse:
                tmp = th.linalg.solve(L, X)
                X_new = th.linalg.solve(L.transpose(-1, -2), tmp, left=False)
            else:
                X_new = L.matmul(X).matmul(L.transpose(-1, -2))

        elif self.metric == 'LEM' or self.metric == 'LCM':
            X_new = X - P if is_inverse else X + P

        return X_new
    def scaling(self, X, var,scale):
        """Scaling by variance"""
        factor = scale / (var+self.eps).sqrt()
        if self.metric == 'AIM':
            X_new = sym_functionals.sym_powm.apply(X, factor)
        elif self.metric == 'LEM' or self.metric == 'LCM':
            X_new = X * factor

        return X_new
    def updating_running_statistics(self,batch_mean,batch_var=None):
        """updating running mean"""
        with th.no_grad():
            # updating mean
            if self.metric == 'AIM':
                self.running_mean.data = sym_functionals.spd_2point_interpolation(self.running_mean, batch_mean,self.momentum)
            elif self.metric == 'LEM' or self.metric == 'LCM':
                self.running_mean.data = (1-self.momentum) * self.running_mean+ batch_mean * self.momentum
            # updating var
            self.running_var.data = (1 - self.momentum) * self.running_var + batch_var * self.momentum

    def BaryGeom(self,X,karcher_steps=1,batchdim=0):
        '''
        Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
        Input x is a batch of SPD matrices (...,n,n) to average
        Output is (n,n) Riemannian mean
        '''
        batch_mean = X.mean(dim=batchdim,keepdim=True)
        for _ in range(karcher_steps):
            bm_sq, bm_invsq = sym_functionals.sym_invsqrtm2.apply(batch_mean)
            XT = sym_functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq)
            GT = XT.mean(dim=batchdim,keepdim=True)
            batch_mean = bm_sq @ sym_functionals.sym_expm.apply(GT) @ bm_sq
        return batch_mean.squeeze()
