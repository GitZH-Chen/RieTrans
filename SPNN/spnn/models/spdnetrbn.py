import torch
import torch.nn as nn

import spnn.modules as modules

from spnn.LieBN import LieBatchNormSPD
from spnn.brooksbn import BatchNormSPD


class SPDNetRBN(nn.Module):
    """Build SPDNet with Brooks or Lie-group SPD batch normalization."""

    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]
        self.feature = []
        if len(dims) > 1:
            for i in range(len(dims) - 2):
                shape=[dims[i], dims[i + 1]]
                self.feature.append(modules.BiMap(shape))
                self.feature.append(SPDBN(dims[i + 1], args))
                self.feature.append(modules.ReEig())

            self.feature.append(modules.BiMap([dims[-2], dims[-1]]))
            self.feature.append(SPDBN(dims[-1], args))
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
        tsdim = int( subspacedims ** 2 )
        self.classifier = torch.nn.Sequential(
            modules.LogEig(subspacedims),
            torch.nn.Linear(tsdim, args.class_num),
        )

class SPDBN(nn.Module):
    """Select the SPD batch-normalization layer configured for SPDNetBN."""

    def __init__(self, n,args):
        super(__class__, self).__init__()
        if args.BN_type == 'brooks':
            self.BN = BatchNormSPD(n,args.momentum)
        elif args.BN_type == 'LieBN':
            self.BN = LieBatchNormSPD(n,
                                      metric=args.bn_metric,
                                      power=args.bn_power, momentum=args.momentum)
        else:
            raise Exception('unknown BN {}'.format(args.BN_type))

    def forward(self, x):
        x_spd = self.BN(x)
        return x_spd
