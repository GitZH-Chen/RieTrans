import torch as th
import torch.nn as nn
from . import brooksbn_functional as functional
import geoopt
from geoopt.manifolds import SymmetricPositiveDefinite


dtype=th.double
device=th.device('cpu')

class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """
    def __init__(self,n,momentum=0.1):
        super(__class__,self).__init__()
        self.momentum=momentum;self.n=n
        self.running_mean=th.eye(n,dtype=dtype) ################################
        self.weight = geoopt.ManifoldParameter(th.eye(n, n, dtype=dtype),
                                               manifold=SymmetricPositiveDefinite())
    def forward(self,X):
        N,h,n,n=X.shape
        X_batched=X.permute(2,3,0,1).contiguous().view(n,n,N*h,1).permute(2,3,0,1).contiguous()
        if(self.training):
            mean=functional.BaryGeom(X_batched)
            with th.no_grad():
                self.running_mean.data=functional.geodesic(self.running_mean,mean,self.momentum)
            X_centered=functional.CongrG(X_batched,mean,'neg')
        else:
            X_centered=functional.CongrG(X_batched,self.running_mean,'neg')
        X_normalized=functional.CongrG(X_centered,self.weight,'pos')
        X_new = X_normalized.permute(2,3,0,1).contiguous().view(n,n,N,h).permute(2,3,0,1).contiguous()
        return X_new
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},momentum={self.momentum})"
