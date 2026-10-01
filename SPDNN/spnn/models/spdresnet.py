import torch as th
import torch.nn as nn
import rresnet
from rresnet import SPD

import geoopt
import spnn.modules as modules


class RResNet(nn.Module):
    """Build the Riemannian residual-network baseline on SPD matrices."""

    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]

        self.feature = []
        if len(dims) > 1:
            for i in range(len(dims) - 2):
                shape=[dims[i], dims[i + 1]]
                self.feature.append(modules.BiMap(shape))
                self.feature.append(modules.ReEig())

            self.feature.append(modules.BiMap([dims[-2], dims[-1]]))
            self.feature = nn.Sequential(*self.feature)
        else:
            self.feature = nn.Sequential(*self.feature)

        self.resblock=SPDResBlock(dims[-1],metric=args.ResBlockMetric)
        self.construct_classifier(args,dims[-1])

    def forward(self, x):
        x = self.feature(x)
        # Rest Block
        x_res=self.resblock(x)
        y=self.classifier(x_res)
        return y

    def construct_classifier(self,args,subspacedims):
        tsdim = int( subspacedims ** 2 )
        if args.ResBlockMetric=='LEM':
            self.classifier = th.nn.Sequential(
                nn.Flatten(),
                th.nn.Linear(tsdim, args.class_num),
            )
        else:
            self.classifier = th.nn.Sequential(
                modules.LogEig(subspacedims),
                th.nn.Linear(tsdim, args.class_num),
            )

class SPDResBlock(nn.Module):
    """Apply the metric-specific residual block used by the SPD RResNet."""

    def __init__(self,dim,metric='AIM'):
        # aff_inv,log_euc
        super(__class__, self).__init__()
        self.dim=dim;self.metric=metric

        self.P = th.empty(dim, dim)
        nn.init.normal_(self.P, std=1e-2)
        self.P = th.svd(self.P)[0]
        self.P = geoopt.ManifoldParameter(self.P, manifold=geoopt.manifolds.Stiefel())
        self.manifold = SPD(metric=metric)

        self.spectrum_map = nn.Sequential(
            nn.Conv1d(1, 3, 5, padding="same").double(),
            nn.LeakyReLU(),
            nn.BatchNorm1d(3).double(),
            nn.Conv1d(3, 3, 5, padding="same").double(),
            nn.LeakyReLU(),
            nn.BatchNorm1d(3).double(),
            nn.Conv1d(3, 1, 5, padding="same").double(),
        )

    def forward(self,x):
        if self.metric=='LEM':
            eigs, evecs = th.linalg.eigh(x)
            f_eigs = self.spectrum_map(eigs)
            v1 = rresnet.manifolds.spd._mvmt(self.P, f_eigs, self.P)
            v1 = self.manifold.proju(x, v1)
            eigs = th.clamp(eigs, 1e-8, 1e8)
            log_x = rresnet.manifolds.spd._mvmt(evecs, th.log(eigs), evecs)
            x_new = log_x + v1
        elif self.metric=='AIM':
            x=x.squeeze()
            eigs = th.linalg.eigvalsh(x)
            f_eigs = self.spectrum_map(eigs.unsqueeze(1)).squeeze()
            v1 = rresnet.manifolds.spd._mvmt(self.P, f_eigs,self.P)
            v1 = self.manifold.proju(x, v1) / self.manifold.norm(x, v1)[:, None, None]
            x = self.manifold.exp(x, v1)
            x_new = self.manifold.projx(x).unsqueeze(1)

        return x_new

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim},metric={self.metric})"
