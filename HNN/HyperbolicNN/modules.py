import torch as th
import torch.nn as nn

from HyperbolicNN.hyperbolic_layers import (
    Hyperboloid_layers,
    Klein_layers,
    PoincareBall_layers,
)

class HyperbolicLinear(nn.Module):
    """HFC using the batch-efficient closed forms in Theorem 4.3.

    Parameters:
        z: [m, n], gamma: [m]
        K: negative curvature with K < 0
    """
    def __init__(self, in_dim, out_dim, metric='PoincareBall', gyro_bias=True, K=-1., dropout=0, drop_mode='feature', normalize_v=False):
        super(__class__, self).__init__()

        if th.any(th.as_tensor(K).detach() >= 0):
            raise ValueError(f"HyperbolicLinear requires negative curvature K < 0, got {K}.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_gyro_bias = gyro_bias
        self.K = K
        self.metric=metric
        self.dropout = dropout
        self.drop_mode=drop_mode
        self.normalize_v = bool(normalize_v)
        self.getmetric()
        self.init_parameter()

    def forward(self, x):
        # The manifold implementation uses the curvature magnitude |K|, whereas HFC uses K < 0.
        curvature_magnitude = -self.K
        if self.drop_mode=='weight':
            drop_weight = nn.functional.dropout(self.weight, self.dropout, training=self.training)
            transformed = self.manifold.linear(
                x, drop_weight, self.bias_p, self.K, normalize_v=self.normalize_v
            )
            res = self.manifold.proj(transformed, curvature_magnitude)
        elif self.drop_mode=='feature':
            if self.metric == 'Hyperboloid':
                spatial = nn.functional.dropout(
                    x[..., 1:],
                    self.dropout,
                    training=self.training,
                )
                time = th.sqrt(
                    curvature_magnitude.reciprocal()
                    + spatial.pow(2).sum(dim=-1, keepdim=True)
                )
                x_drop = th.cat([time, spatial], dim=-1)
            else:
                x_drop = nn.functional.dropout(
                    x,
                    self.dropout,
                    training=self.training,
                )
            transformed = self.manifold.linear(
                x_drop, self.weight, self.bias_p, self.K, normalize_v=self.normalize_v
            )
            res = self.manifold.proj(transformed, curvature_magnitude)
        else:
            raise ValueError(
                f"Unsupported drop_mode={self.drop_mode!r}."
            )
        if self.is_gyro_bias:
            tan_bias = self.manifold.totangent(self.gyro_bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(tan_bias, curvature_magnitude)
            hyp_bias = self.manifold.proj(hyp_bias, curvature_magnitude)
            res = self.manifold.add(res, hyp_bias, curvature_magnitude)
            res = self.manifold.proj(res, curvature_magnitude)
        return res

    def getmetric(self):
        classes = {
            "PoincareBall": PoincareBall_layers,
            "Hyperboloid": Hyperboloid_layers,
            "Klein": Klein_layers,
        }
        self.manifold = classes[self.metric]()

    def init_parameter(self):
        gain = 1.
        weight = th.empty(self.out_dim,self.in_dim).normal_(
            mean=0, std=(2 * self.in_dim * self.out_dim ) ** -0.5 * gain)
        self.weight = nn.Parameter(weight)
        self.bias_p = nn.Parameter(th.zeros(self.out_dim))
        if self.is_gyro_bias:
            self.gyro_bias = nn.Parameter(th.zeros(self.out_dim))


    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, metric={self.metric}, ' \
               f'gyro_bias={self.is_gyro_bias}, K={self.K},dropout={self.dropout}, drop_mode={self.drop_mode}, normalize_v={self.normalize_v}'


class Gyrobias(nn.Module):
    """Trainable gyroaddition bias for Poincare and Lorentz features."""

    def __init__(self, dim, metric="poincare", K=-1.0):
        super().__init__()
        self.dim = dim
        self.metric = metric.lower()
        self.K = K

        manifolds = {
            "poincare": PoincareBall_layers,
            "lorentz": Hyperboloid_layers,
        }
        if self.metric not in manifolds:
            raise NotImplementedError(
                f"Unsupported metric {self.metric!r}. Expected 'poincare' or 'lorentz'."
            )
        self.manifold = manifolds[self.metric]()
        self.tangent = nn.Parameter(th.zeros(self.dim))

    def forward(self, x):
        tangent = self.tangent
        if self.metric == "lorentz":
            tangent = th.cat([tangent.new_zeros(1), tangent], dim=0)
        tangent = tangent.unsqueeze(0)
        curvature_magnitude = -self.K
        bias = self.manifold.expmap0(tangent, curvature_magnitude)
        return self.manifold.add(bias, x, curvature_magnitude)
