import torch
from typing import Callable
from typing import Any
from torch.autograd import Function
from torch.functional import Tensor

# define the epsilon precision depending on the tensor datatype
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False, decom_mode='svd') -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function          
        ensure_symmetric : bool = False (optional) 
            if ensure_symmetric=True, then M is symmetrized          
        ensure_psd : bool = False (optional) 
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0                  
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        if ensure_symmetric:
            M = ensure_sym(M)

        # compute the eigenvalues and vectors
        if decom_mode=='eigh':
            s, U = torch.linalg.eigh(M)
        elif decom_mode=='svd':
            U, s,_ = torch.linalg.svd(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor, 
                    fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:   
        """Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative               
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_reeig(Function):
    """
    Rectifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).
    """
    @staticmethod
    def value(s : Tensor, threshold : Tensor) -> Tensor:
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s : Tensor, threshold : Tensor) -> Tensor:
        return (s>threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold), None, None

class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative 
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """
    Computes the matrix exponential for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric,decom_mode='eigh')
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """
    Computes the matrix power for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, exponent : Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s : Tensor, exponent : Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent-1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        dXs = (U.transpose(-1,-2) @ ensure_sym(dX) @ U).diagonal(dim1=-1,dim2=-2)
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(Function):
    """
    Computes the matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(Function):
    """
    Computes the inverse matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(Function):
    """
    Computes the square root and inverse square root matrices for a batch of SPD matrices.
    """

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ torch.diag_embed(smod2) @ U.transpose(-1,-2)
        ctx.save_for_backward(s, smod, smod2, U)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        return dMsq + dMinvsq, None

