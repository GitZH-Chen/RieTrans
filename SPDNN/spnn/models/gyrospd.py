import torch.nn as nn

from spnn.GyroLayers.GyroMLR import GyroMLR
from spnn.GyroLayers.GyroLayers import GyroTrans


class GyroSPD(nn.Module):
    """Build the GyroLE, GyroAI, or GyroLC baseline."""

    def __init__(self,args):
        super(__class__, self).__init__()
        self.feature=[]
        self.feature.append(GyroTrans(shape_in=args.architecture))
        self.feature = nn.Sequential(*self.feature)

        self.mlr = GyroMLR(shape_in=args.architecture,class_num=args.class_num,metric=args.metric)

    def forward(self, x):
        x_spd = self.feature(x)
        y = self.mlr(x_spd)
        return y
