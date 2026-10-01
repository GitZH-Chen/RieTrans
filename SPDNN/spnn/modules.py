import math
from typing import Tuple
import torch
from torch import Tensor
from torch.types import Number
import torch.nn as nn
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel, Sphere, Euclidean
from . import functionals

class BiMap(nn.Module):
    """Apply the bilinear SPD transformation used by the retained baselines."""
    def __init__(self, shape : Tuple[int, ...] or torch.Size, W0 : Tensor = None, manifold='stiefel', init_mode='uniform', **kwargs):
        super().__init__()

        self.shape=shape;self.manifold=manifold;self.init_mode=init_mode
        if manifold == 'euclidean':
            mf = Euclidean()
        else:
            if manifold == 'stiefel':
                assert(shape[-2] >= shape[-1])
                mf = Stiefel()
            elif manifold == 'sphere':
                mf = Sphere()
                shape = list(shape)
                shape[-1], shape[-2] = shape[-2], shape[-1]
            else:
                raise NotImplementedError()

        # add constraint (also initializes the parameter to fulfill the constraint)
        self.W = ManifoldParameter(torch.empty(shape, **kwargs), manifold=mf)

        # optionally initialize the weights (initialization has to fulfill the constraint!)
        if W0 is not None:
            self.W.data = W0 # e.g., self.W = torch.nn.init.orthogonal_(self.W)
        else:
            self.reset_parameters()
    
    def forward(self, X : Tensor) -> Tensor:
        if isinstance(self.W.manifold, Sphere):
            return self.W @ X @ self.W.transpose(-2,-1)
        else:
            return self.W.transpose(-2,-1) @ X @ self.W

    @torch.no_grad()
    def reset_parameters(self):
        if isinstance(self.W.manifold, Euclidean):
            v = torch.empty_like(self.W).uniform_(0., 1.)
            vv = torch.svd(v.matmul(v.t()))[0][:, :self.W.shape[-1]]
            self.W.data = vv
        elif isinstance(self.W.manifold, Stiefel):
            if self.init_mode=='uniform':
                # uniform initialization on stiefel manifold after theorem 2.2.1 in Chikuse (2003): statistics on special manifolds
                W = torch.rand(self.W.shape, dtype=self.W.dtype, device=self.W.device)
                self.W.data = W @ functionals.sym_invsqrtm.apply(W.transpose(-1,-2) @ W)
            elif self.init_mode=='svd':
                v = torch.empty_like(self.W).uniform_(0., 1.)
                vv = torch.svd(v.matmul(v.t()))[0][:, :self.W.shape[-1]]
                self.W.data = vv
        elif isinstance(self.W.manifold, Sphere):
            W = torch.empty(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            # kaiming initialization std2uniformbound * gain * fan_in
            bound = math.sqrt(3) * 1. / W.shape[-1]
            W.uniform_(-bound, bound)
            # constraint has to be satisfied
            self.W.data = W / W.norm(dim=-1, keepdim=True)


    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape},manifold={self.manifold},init_mode={self.init_mode})"


class ReEig(nn.Module):
    def __init__(self, threshold : Number = 1e-4):
        super().__init__()
        self.threshold = Tensor([threshold])

    def forward(self, X : Tensor) -> Tensor:
        return functionals.sym_reeig.apply(X, self.threshold)

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class LogEig(nn.Module):
    """Map SPD matrices to log coordinates and vectorize them."""
    def __init__(self, ndim, tril=False):
        super().__init__()

        self.tril = tril
        if self.tril:
            ixs_lower = torch.tril_indices(ndim,ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None,:].tile((2,1)), ixs_lower), dim=1)
        self.ndim = ndim

    def forward(self, X : Tensor) -> Tensor:
        return self.embed(functionals.sym_logm.apply(X))

    def embed(self, X : Tensor) -> Tensor:
        if self.tril:
            x_vec = X[...,self.ixs[0],self.ixs[1]]
            x_vec[...,self.ndim:] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=1)
        return x_vec

    def __repr__(self):
        return f"{self.__class__.__name__}(ndim={self.ndim},tril={self.tril})"

class SPDPower(nn.Module):
    def __init__(self, power):
        super(__class__, self).__init__()
        self.register_buffer('power', torch.tensor(power))

    def forward(self, x):
        x_spd = functionals.sym_powm.apply(x,self.power)
        return x_spd

    def __repr__(self):
        return f"{self.__class__.__name__}(power={self.power})"
