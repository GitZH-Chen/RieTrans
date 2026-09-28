import torch
import torch.nn as nn

import spnn.modules as modules
from spnn.SPDMLR import SPDRMLR


class SPDNet(nn.Module):
    """Build the SPDNet baseline and its LogEigMLR or SPDMLR classifier."""

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

        self.construct_classifier(args,dims[-1])

    def forward(self, x):
        if len(self.feature) > 0:
            x_spd = self.feature(x)
        else:
            x_spd = x  # Bypass the feature extraction phase
        y = self.classifier(x_spd)
        return y

    def construct_classifier(self,args,subspacedims):
        if args.classifier=='SPDMLR':
            self.classifier = torch.nn.Sequential(
                SPDRMLR(n=subspacedims,c=args.class_num,metric=args.clf_metric,power=args.clf_power)
                )
        elif args.classifier=='LogEigMLR':
            """Following SPDNet and SPDNetBN, we use the full matrices"""
            tsdim = int( subspacedims ** 2 )
            self.classifier = torch.nn.Sequential(
                modules.LogEig(subspacedims),
                torch.nn.Linear(tsdim, args.class_num),
            )
        else:
            raise NotImplementedError
