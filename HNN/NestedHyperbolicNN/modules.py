"""Hyperbolic Feature Transformation from Nested Hyperbolic Spaces.

The rotation, boost, and projection implementations are copied from
``Nested-Hyperbolic-Space/NHGCN/layers/hyp_layers.py`` and adapted only to
the manifold interface used by RConvNet-H.
"""

import math

import geotorch
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.module import Module


class _ManifoldAdapter:
    """Expose the official ``regularizex/projx`` API on RConvNet-H."""

    def __init__(self, manifold, c):
        self.manifold = manifold
        self.c = c

    def regularizex(self, x):
        if torch.is_tensor(self.c):
            k = self.c.to(device=x.device, dtype=x.dtype).reciprocal()
        else:
            k = x.new_tensor(1.0 / self.c)
        sq_norm = torch.abs(
            torch.sum(x * x, dim=-1) - 2 * x[..., 0] * x[..., 0]
        ).clamp(min=1e-2)
        real_norm = torch.sqrt(torch.abs(sq_norm))
        return torch.einsum("...i,...->...i", x, k * 1.0 / real_norm)

    def projx(self, x):
        return self.manifold.proj(x, self.c)


class LorentzBoost(Module):
    """hyperbolic rotaion achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor(1))

    def forward(self, x): # x =[x_0,x_1,...,x_n]
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2) #x_narrow = [x_1,...,x_n-1]
        # x_0 = torch.cosh(self.weight) * x.narrow(-1, 0, 1) + torch.sinh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        # x_n = torch.sinh(self.weight) * x.narrow(-1, 0, 1) + torch.cosh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        x_0 = torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, 0, 1) + self.weight * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        x_n = self.weight * x_narrow.narrow(-1, 0, 1) + torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))


class LorentzRotation(Module):
    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        if_dropout=False,
        dropout=0,
        if_regularize = False,
        if_projected = False
        ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features-1, self.out_features-1,bias =False)
        geotorch.orthogonal(self.linear,"weight")
        self.reset_parameters()
        self.if_dropout = if_dropout
        self.dropout = nn.Dropout(dropout)
        self.if_regularize = if_regularize
        self.if_projected = if_projected

    def forward(self, x):

        x_0 = x.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        if self.if_dropout is True:
            x_narrow = self.dropout(x_narrow)

        x_ = self.linear.forward(x_narrow)
        x = torch.cat([x_0, x_], dim=-1)
        if self.if_regularize is True:
            x = self.manifold.regularizex(x)
        if self.if_projected is True:
            x = self.manifold.projx(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.linear.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.linear.weight[:, idx] = 0


class HyperbolicFeatureTransformation(nn.Module):
    r"""Hyperbolic Feature Transformation in Eq. (14).

    Eq. (14) defines the transformation from :math:`\mathbb{L}^n` to
    :math:`\mathbb{L}^m` as

    .. math::
        y = \frac{Wx}{\lVert Wx\rVert_{\mathcal{L}}},
        \qquad WJ_nW^\top = J_m.

    Section 3.3 parameterizes the constrained transformation matrix as

    .. math::
        W =
        \begin{bmatrix}1&0\\0&\widetilde P\end{bmatrix}
        B(\alpha)
        \begin{bmatrix}1&0\\0&Q^\top\end{bmatrix},

    where :math:`Q\in SO(n)`, :math:`\alpha\in\mathbb{R}`, and
    :math:`\widetilde P\in\mathrm{St}(m,n)`. Because these matrices act on
    a column vector from right to left, the implementation applies the
    Lorentz rotation, Lorentz boost, and nested projection sequentially.
    The final projection module also performs the Lorentz regularization
    and manifold projection corresponding to the normalization in Eq. (14).
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        c,
        use_bias=False,
        dropout=0,
    ):
        super().__init__()
        self.use_bias = use_bias
        manifold = _ManifoldAdapter(manifold, c)
        self.rotation = LorentzRotation(
            manifold,
            in_features,
            in_features,
            if_dropout=False,
            if_regularize=False,
            if_projected=False,
        )
        self.boost = LorentzBoost(manifold)
        # The copied LorentzBoost constructor leaves its one-element storage
        # uninitialized. Start Eq. (14) from the identity boost (alpha = 0)
        # without changing the original LorentzBoost implementation.
        nn.init.zeros_(self.boost.weight)
        self.projection = LorentzRotation(
            manifold,
            in_features,
            out_features,
            if_dropout=True,
            dropout=dropout,
            if_regularize=True,
            if_projected=True,
        )

    def forward(self, x):
        x = self.rotation(x)  # [1, 0; 0, Q^T] x
        x = self.boost(x)  # B(alpha)
        return self.projection(x)  # [1, 0; 0, P_tilde] + normalization
