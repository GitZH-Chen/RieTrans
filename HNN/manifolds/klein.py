"""Klein Model."""

import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh


class Klein(Manifold):
    """
    Klein Manifold class.
    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c
    Note that 1/sqrt(c) is the Klein ball radius and c=-K
    """

    def __init__(self, ):
        super(__class__, self).__init__()
        self.name = 'Klein'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def klein_to_poincare(self, x, c):
        norm_x = x.norm(dim=-1, keepdim=True, p=2)  # Compute the norm of x
        return x / (torch.sqrt(1 - c * norm_x ** 2) + 1)

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        add_norm = self.add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        dom = 1 + torch.sqrt(1 - c * add_norm **2)
        dist_c = artanh(sqrt_c * add_norm / dom)
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def add(self, x, y, c, dim=-1):
        """einstein_add"""
        x2 = x.pow(2).sum(dim=dim, keepdim=True)  # ||x||^2
        xy = (x * y).sum(dim=dim, keepdim=True) # <x, y>
        gamma_x = 1 / torch.sqrt(1 - c * x2)
        num = x + (1 / gamma_x) * y + c * (gamma_x / (1 + gamma_x)) * xy * x
        denom = 1 + c * xy
        return num / denom.clamp_min(self.min_norm)

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan0(self, u, c):
        return u

    def matvec(self, m, x, c):
        """Prop. I.5: einstein_matvec share the same expression as the mobius_matvec"""
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True).bool()
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

