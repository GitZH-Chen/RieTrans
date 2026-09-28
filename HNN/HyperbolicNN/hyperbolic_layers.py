"""Batch-efficient implementation of HFC-P, HFC-K, and HFC-H from "Building Transformation Layers for Riemannian Neural Networks."

Let ``B`` be the batch size, ``n`` the input dimension, and ``m`` the output dimension. All three layers take ``z`` of shape ``[m, n]``, ``gamma`` of shape ``[m]``, and negative curvature ``K < 0``.
HFC-P and HFC-K take ``x`` of shape ``[B, n]``; ``MLR`` returns responses of shape ``[B, m]``, and ``linear`` returns manifold-valued outputs of shape ``[B, m]``.
HFC-H takes ``x`` of shape ``[B, n + 1]``; ``MLR`` returns responses of shape ``[B, m]``, and ``linear`` returns hyperboloid outputs of shape ``[B, m + 1]``.
"""

import torch as th
import torch.nn as nn

from manifolds.hyperboloid import Hyperboloid
from manifolds.klein import Klein
from manifolds.poincare import PoincareBall
from utils.math_utils import arcosh, artanh


class Hyperbolic_layers(nn.Module):
    """Shared operations for the three closed-form HFC manifestations.

    HFC formulas use the negative sectional curvature ``K < 0``. The general manifold implementations use the positive curvature magnitude ``c = |K| = -K > 0``.
    """

    def __init__(self):
        super().__init__()
        self.min_norm = 1e-15
        self.eps = {th.float32: 4e-3, th.float64: 1e-5}

    @staticmethod
    def pairwise_inner(x, y):
        """Return all Euclidean row-wise pairings between ``x`` and ``y``."""
        return x @ y.transpose(-1, -2)

    def normalize(self, z):
        """Normalize vectors along the final dimension."""
        return z / th.clamp(
            th.norm(z, p=2, dim=-1, keepdim=True),
            min=self.eps[z.dtype],
        )

    def trivialized_radial_parameters(self, z, gamma, K):
        """Return the quantities used by the trivialized parameterization.

        ``direction_scale = ||z_i|| / max(||z_i||, eps)`` has the following piecewise form:

        - ``direction_scale = 1`` when ``||z_i|| >= eps``;
        - ``direction_scale = ||z_i|| / eps`` when ``0 < ||z_i|| < eps``;
        - ``direction_scale = 0`` when ``z_i = 0``.

        Its purpose is to shrink the effective radial parameter ``gamma_i direction_scale`` when ``z_i`` is small.

        The returned tuple contains:

        - ``weight_norm``: the row-wise norm ``||z_i||``;
        - ``unit_direction``: the row-wise unit vector ``[z_i] = z_i / ||z_i||``, with a zero row represented by zero;
        - ``direction_scale``: an implementation-only numerical safeguard
        - ``rho``: ``rho_i = tanh(sqrt(|K|) gamma_i direction_scale) / sqrt(|K|)``, so ``p_i = rho_i [z_i]``.
        """
        weight_norm = th.norm(z, p=2, dim=-1)
        safe_weight_norm = weight_norm.clamp_min(self.eps[z.dtype])
        direction_scale = weight_norm / safe_weight_norm
        unit_direction = z / th.where(
            weight_norm.gt(0),
            weight_norm,
            th.ones_like(weight_norm),
        ).unsqueeze(-1)
        sqrt_abs_K = (-K) ** 0.5
        rho = (
            th.tanh(sqrt_abs_K * gamma * direction_scale)
            / sqrt_abs_K
        )
        return weight_norm, unit_direction, direction_scale, rho

    def MLR(self, x, z, gamma, K):
        """Compute the v_i."""
        raise NotImplementedError

    def linear(self, x, z, gamma, K, normalize_v=False):
        """Optional response normalization."""
        response = self.MLR(x, z, gamma, K)
        if normalize_v:
            response = self.normalize(response)
        return self.expmap0(self.totangent(response), -K)

    def totangent(self, v):
        """Map the response coordinates to the tangent representation used by the manifold model."""
        raise NotImplementedError

class _Ball_layers(Hyperbolic_layers):
    """Operations shared only by the Poincare and Klein ball models."""

    @staticmethod
    def totangent(v):
        """Ball-model response coordinates already represent a tangent vector at the origin."""
        return v

    def logmap0_inner(self, q_norm_sq, q_dot_z, K):
        """Compute ``<Log_0(q), z>`` for the two ball models.

        The Poincare and Klein models use the same logarithmic map at the origin, so this scalar reduction can be shared; their distinction remains in the preceding gyroaddition.
        """
        sqrt_abs_K = (-K) ** 0.5
        q_norm = th.sqrt(th.clamp(q_norm_sq, min=0.0))
        q_norm = th.clamp(q_norm, min=self.min_norm)
        # scaled_q_norm = sqrt_abs_K * q_norm
        # scaled_q_norm = th.clamp(scaled_q_norm, max=1.0 - self.eps[q_norm.dtype])
        # scale = artanh(scaled_q_norm) / scaled_q_norm
        scale = artanh(sqrt_abs_K * q_norm) / (sqrt_abs_K * q_norm)
        return scale * q_dot_z


class PoincareBall_layers(PoincareBall, _Ball_layers):
    """Closed-form HFC-P response corresponding to Theorem 4.3 (``thm:hfc_batch_closed_form``).

    For ``x`` of shape ``[B, n]``, ``z`` of shape ``[m, n]``, and ``gamma`` of shape ``[m]``, ``MLR`` evaluates every ``v_i(x)`` without constructing ``p_i`` or a ``[B, m, n]`` tensor.
    The code variables correspond to the theorem as follows:

    - ``abs_K = |K|``, ``weight_norm = ||z_i||``, and ``unit_direction = [z_i]``.
    - ``rho_i``: ``p_i = Exp_0(gamma_i [z_i]) = rho_i [z_i]``, where ``rho_i = tanh(sqrt(|K|) gamma_i) / sqrt(|K|)`` is implemented as ``rho``.
    - ``input_sqnorm = ||x||^2`` and ``directional_inner = <x, [z_i]>``; the latter is computed for all batch-output pairs by one matrix multiplication.
    - ``denominator = D_i^P`` and ``q_norm_sq = (Q_i^P)^2``, where ``q_i = (-p_i) op_M x``.
    - ``q_dot_z = <q_i, z_i>`` is the theorem's fraction containing ``||z_i||`` and ``D_i^P`` before applying the logarithmic-map scale.
    - ``logmap0_inner`` multiplies ``q_dot_z`` by ``atanh(sqrt(|K|) Q_i^P) / (sqrt(|K|) Q_i^P)`` and therefore returns the complete ``v_i(x)`` in Theorem 4.3.
    """

    def MLR(self, x, z, gamma, K):
        abs_K = -K
        weight_norm, unit_direction, _, rho = (
            self.trivialized_radial_parameters(z, gamma, K)
        )
        input_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        directional_inner = self.pairwise_inner(x, unit_direction)
        denominator = (
            1
            - 2 * abs_K * rho * directional_inner
            + abs_K ** 2 * rho.pow(2) * input_sqnorm
        ).clamp_min(self.min_norm)
        q_norm_sq = (
            input_sqnorm
            + rho.pow(2)
            - 2 * rho * directional_inner
        ) / denominator
        q_dot_z = (
            weight_norm
            * (
                (1 + abs_K * rho.pow(2)) * directional_inner
                - rho * (1 + abs_K * input_sqnorm)
            )
        ) / denominator
        # Theorem 4.3: q_norm_sq = (Q_i^P)^2, while q_dot_z = ||z_i||[(1 + |K|rho_i^2)<x,[z_i]> - rho_i(1 + |K|||x||^2)]/D_i^P.
        # logmap0_inner multiplies q_dot_z by artanh(sqrt(|K|)Q_i^P)/(sqrt(|K|)Q_i^P), returning v_i(x).
        return self.logmap0_inner(
            q_norm_sq,
            q_dot_z,
            K,
        )

class Klein_layers(Klein, _Ball_layers):
    """Closed-form HFC-K response corresponding to Theorem 4.3 (``thm:hfc_batch_closed_form``).

    For ``x`` of shape ``[B, n]``, ``z`` of shape ``[m, n]``, and ``gamma`` of shape ``[m]``, ``MLR`` evaluates every ``v_i(x)`` without constructing ``p_i`` or a ``[B, m, n]`` tensor.
    The code variables correspond to the theorem as follows:

    - ``abs_K = |K|``, ``weight_norm = ||z_i||``, and ``unit_direction = [z_i]``.
    - ``rho_i``: ``p_i = Exp_0(gamma_i [z_i]) = rho_i [z_i]``, where ``rho_i = tanh(sqrt(|K|) gamma_i) / sqrt(|K|)`` is implemented as ``rho``.
    - ``input_sqnorm = ||x||^2`` and ``directional_inner = <x, [z_i]>``; the latter is computed for all batch-output pairs by one matrix multiplication.
    - ``denominator = D_i^K`` and ``q_norm_sq = (Q_i^K)^2``, where ``q_i = (-p_i) op_E x``.
    - ``q_dot_z = <q_i, z_i>`` is the theorem's fraction containing ``||z_i||`` and ``D_i^K`` before applying the logarithmic-map scale.
    - ``logmap0_inner`` multiplies ``q_dot_z`` by ``atanh(sqrt(|K|) Q_i^K) / (sqrt(|K|) Q_i^K)`` and therefore returns the complete ``v_i(x)`` in Theorem 4.3.
    """

    def MLR(self, x, z, gamma, K):
        abs_K = -K
        weight_norm, unit_direction, _, rho = (
            self.trivialized_radial_parameters(z, gamma, K)
        )
        input_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        directional_inner = self.pairwise_inner(x, unit_direction)
        denominator = (
            1 - abs_K * rho * directional_inner
        ).clamp_min(self.min_norm)
        q_norm_sq = (
            input_sqnorm
            + rho.pow(2)
            - 2 * rho * directional_inner
            - abs_K
            * rho.pow(2)
            * (input_sqnorm - directional_inner.pow(2))
        ) / denominator.pow(2)
        q_dot_z = (
            weight_norm
            * (directional_inner - rho)
            / denominator
        )
        # Theorem 4.3: q_norm_sq = (Q_i^K)^2, while q_dot_z = ||z_i||(<x,[z_i]> - rho_i)/D_i^K.
        # logmap0_inner multiplies q_dot_z by artanh(sqrt(|K|)Q_i^K)/(sqrt(|K|)Q_i^K), returning v_i(x).
        return self.logmap0_inner(
            q_norm_sq,
            q_dot_z,
            K,
        )

class Hyperboloid_layers(Hyperboloid, Hyperbolic_layers):
    """Closed-form HFC-H response corresponding to Theorem 4.3 (``thm:hfc_batch_closed_form``).

    For ``x = (x_1, x_s)`` of shape ``[B, n + 1]``, ``z`` of shape ``[m, n]``, and ``gamma`` of shape ``[m]``, ``MLR`` evaluates every ``v_i(x)`` by batched inner products.
    The code variables correspond to the theorem as follows:

    - ``abs_K = |K|``, ``sqrt_abs_K = sqrt(|K|)``, ``weight_norm = ||z_i||``, and ``unit_direction = [z_i]``.
    - ``theta = sqrt(|K|) gamma_i``, while ``cosh_theta`` and ``sinh_theta`` are ``cosh(theta_i)`` and ``sinh(theta_i)``.
    - ``x_time = x_1`` and ``directional_inner = <x_s, [z_i]>``; the latter is computed for all batch-output pairs by one matrix multiplication.
    - ``alpha = sqrt(|K|)[cosh(theta_i)x_1 - sinh(theta_i)<x_s,[z_i]>]`` is the argument of ``acosh`` in Theorem 4.3.
    - ``scale = acosh(alpha_i) / sqrt(alpha_i^2 - 1)`` and ``transported_inner = ||z_i||[cosh(theta_i)<x_s,[z_i]> - sinh(theta_i)x_1]``.
    - Multiplying ``scale`` and ``transported_inner`` returns the complete ``v_i(x)`` in Theorem 4.3.
    """

    @staticmethod
    def totangent(v):
        zero = th.zeros(
            *v.shape[:-1],
            1,
            dtype=v.dtype,
            device=v.device,
        )
        return th.cat([zero, v], dim=-1)

    def MLR(self, x, z, gamma, K):
        abs_K = -K
        weight_norm, unit_direction, direction_scale, _ = (
            self.trivialized_radial_parameters(z, gamma, K)
        )
        sqrt_abs_K = abs_K ** 0.5
        theta = sqrt_abs_K * gamma * direction_scale
        cosh_theta = th.cosh(theta)
        sinh_theta = th.sinh(theta)
        x_time = x[..., :1]
        directional_inner = self.pairwise_inner(
            x[..., 1:],
            unit_direction,
        )
        alpha = sqrt_abs_K * (
            x_time * cosh_theta
            - directional_inner * sinh_theta
        )
        clamped_alpha = th.clamp(
            alpha,
            min=1.0 + self.eps[x.dtype],
        )
        sqdist = (
            arcosh(clamped_alpha).pow(2) / abs_K
        ).clamp(max=50.0)  # Cap large squared distances to prevent overflow and NaNs in subsequent computations.
        log_direction_norm = th.sqrt(
            th.clamp(
                (alpha.pow(2) - 1) / abs_K,
                min=self.eps[x.dtype],
            )
        ).clamp_min(self.min_norm)
        scale = th.sqrt(sqdist) / log_direction_norm
        transported_inner = weight_norm * (
            directional_inner * cosh_theta
            - x_time * sinh_theta
        )
        # Theorem 4.3: scale = acosh(alpha_i)/sqrt(alpha_i^2 - 1), while transported_inner = ||z_i||[cosh(theta_i)<x_s,[z_i]> - sinh(theta_i)x_1].
        # Their product is the complete HFC-H response v_i(x).
        return scale * transported_inner
