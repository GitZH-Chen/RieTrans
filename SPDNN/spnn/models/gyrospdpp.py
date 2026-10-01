import torch.nn as nn

from spnn.GyroLayers.GyroMLR_simplified import GyroMLR_simplified
from spnn.GyroLayers.GyroLinear import GyroLinear


class GyroSPDpp(nn.Module):
    """Build the one-transformation-layer GyroSPD++ baseline."""

    def __init__(self,args):
        super(__class__, self).__init__()
        self.feature=[]
        shape_in = [args.channels[0], args.architecture[0], args.architecture[0]]
        shape_out = [args.channels[1], args.architecture[1], args.architecture[1]]
        self.feature.append(GyroLinear(shape_in=shape_in,shape_out=shape_out,metric=args.conv_metric))

        self.feature = nn.Sequential(*self.feature)

        self.mlr = GyroMLR_simplified(shape_in=shape_out, class_num=args.class_num, metric=args.clf_metric)

    def forward(self, x):
        x_spd = self.feature(x)
        y = self.mlr(x_spd)
        return y
