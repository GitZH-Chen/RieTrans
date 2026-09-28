import torch.nn as nn

import spnn.cplx.nn as nn_cplx
import spnn.modules as modules
from spnn.SPDLinear import SPDLinear
from spnn.SPDundirectialMLR import SPDundirectialMLR


class SPDNN(nn.Module):
    """Build SPDNN with a global-receptive-field SPD transformation.

    GyroSPD++ configures its SPD convolution with a global receptive field [1]. We follow this configuration in SPNN, so the transformation acts on all input SPD descriptors at once. 
    Our experiments likewise show that performance generally saturates once the receptive field is global.

    Reference:
        [1] X. S. Nguyen, S. Yang, and A. Histace, "Matrix Manifold Neural Networks++," ICLR, 2024.
    """

    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]
        self.feature = []
        if args.dataset == "RADAR":
            self.feature.append(nn_cplx.SplitSignal_cplx(2, 20, 10))
            self.feature.append(nn_cplx.CovPool_cplx())
            self.feature.append(modules.ReEig())
        if args.conv_metric == 'PEM' and args.clf_metric == 'PEM':
            args.clf_power=args.conv_power
        if args.act_power:
            self.feature.append(modules.SPDPower(args.act_power))
        self.feature.append(
            SPDLinear(
                shape_in=[args.channels[0], dims[0], dims[0]],
                shape_out=[args.channels[1], dims[1], dims[1]],
                is_phi_inv=args.conv_is_phi_inv,
                metric=args.conv_metric,
                power=args.conv_power,
            )
        )
        self.feature = nn.Sequential(*self.feature)
        self.classifier = SPDundirectialMLR(
            shape_in=[args.channels[-1], dims[-1], dims[-1]],
            c=args.class_num,
            is_phi=args.clf_is_phi,
            metric=args.clf_metric,
            power=args.clf_power,
        )

    def forward(self, x):
        x_spd = self.feature(x)
        return self.classifier(x_spd)


# Preserve the existing release configuration and launch commands.
SPNN = SPDNN
