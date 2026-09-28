import math
import torch as th
import torch.nn as nn

from .sym_functionals import sym_logm,sym_invsqrtm,sym_expm

class SPDMatrices(nn.Module):
    """Computation for SPD data with [...,n,n]"""
    def __init__(self, n):
        super().__init__()
        self.n=n; self.dim = int(n * (n + 1) / 2)
        self.register_buffer('I', th.eye(n))

    def inner_product(self, A, B):
        """"
        compute the batch inner product of A and B, with [...,n,n] [...,n,n]
        """
        return th.einsum("...ij,...ij->...", A, B)

    def GyroMLR(self, S, P, W):
        """
        GyroMLR based on margin distance
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        W: [class,n,n] SPD matrices
        """
        raise NotImplementedError

    def v2V(self, v):
        """
        transfer a vector tensor [...,n(n+1)/2] into a symmetric matrix
        the first n elements are the diagonal elements, and the rest are the tril triangular parts
        """
        raise NotImplementedError

    def V2SPD(self, V):
        """transforming V to SPD, used in SPD3DConv,
            mostly is related to Exp_{E}, but could be further simplified with orthonormal bases
        """
        return self.expmap_I(V)

    def expmap_I(self, v):
        """exponential map at the identity"""
        print("Not yet implemented")

    def GyroLinear(self, S, P, W):
        """ without power deformation
        SPD3DConv:
            generating A by parallel transportation or the differential of Lie groups translation of Z
            generating P by Exp_{I}(\gamma * [Z])
        Inputs:
            S: [b,c1,n1,n1] SPD
            P: [c2,dim2,c1,n1,n1] SPD, with dimi=ni(ni+1)/2
            W: [c2,dim2,c1,n1,n1] SPD, with dimi=ni(ni+1)/2
        Outputs:
            S_final: [b,c2,n2,n2] SPD
        """
        v = self.GyroMLR(S, P, W)
        V = self.v2V(v)
        S_final = self.V2SPD(V)
        return S_final

class SPDLogEuclideanMetric(SPDMatrices):
    """ (\alpha,\beta)-LEM """
    def __init__(self,n):
        super(__class__, self).__init__(n)

    def GyroMLR(self,S,P,W):
        S_phi = sym_logm.apply(S)
        P_phi = sym_logm.apply(P)
        W_phi = sym_logm.apply(W)

        X_new = self.inner_product(S_phi-P_phi, W_phi)

        return X_new.sum(-1)
    def v2V(self, v):
        # Separate the diagonal and lower triangular parts
        dim=v.shape[-1]
        n = int((-1 + math.sqrt(1 + 8 * dim)) / 2)
        diag_part = v[..., :n]  # First n elements are the diagonal
        tril_part = v[..., n:]  # The rest are the lower triangular part (excluding diagonal)

        # Create an empty tensor to store the symmetric matrix
        V_new = th.zeros(*v.shape[:-1], n, n, dtype=v.dtype, device=v.device)

        # Assign the diagonal elements
        diag_indices = th.arange(n)
        V_new[..., diag_indices, diag_indices] = diag_part

        # Indices for the lower triangular part
        tril_indices = th.tril_indices(n, n, offset=-1)

        # Assign the off-diagonal elements from the tril_part
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part

        return V_new + V_new.tril(-1).transpose(-2, -1)

    def expmap_I(self,V):
        return sym_expm.apply(V)

    def SymLinear(self, S, P, W):
        v = self.GyroMLR(S, P, W)
        V = self.v2V(v)
        return V

class SPDLogCholeskyMetric(SPDMatrices):
    """ \theta-LCM """
    def __init__(self, n):
        super(__class__, self).__init__(n)

    def GyroMLR(self,S,P,W):
        S_phi = self.phi(S)
        P_phi = self.phi(P)
        W_phi = self.phi(W)

        X_new = self.inner_product(S_phi-P_phi, W_phi)
        return X_new.sum(-1)

    def diag_phi_inv(self,diag):
        return th.exp(diag)

    def phi(self,S):
        """ The diffeomorphism Dlog \circ chol: \spd{n} \rightarrow \tril{n}
            S: [...,n,n] SPD matrices
            return: [...,n,n], tril matrix
        """
        L = th.linalg.cholesky(S)  # Compute Cholesky decomposition, shape [..., n, n]
        tril  = th.diag_embed(th.log(th.diagonal(L, dim1=-2, dim2=-1))) + L.tril(-1)
        return tril

    def v2V(self,v):
        """chol^{-1} \circ Dexp: diffeomorphism from \tril{n2} \cong \bbR{dim2} to \spd{n2}
                                [b,c2,dim2] to [b,c2,n2,n2]
           assuming the first n elements are the diagonal
        """
        dim = v.shape[-1]
        n = int((-1 + math.sqrt(1 + 8 * dim)) / 2)
        diag_part = v[..., :n]  # First n elements are the diagonal
        tril_part = v[..., n:]  # The rest are the lower triangular part (excluding diagonal)

        # Create an empty tensor to store the symmetric matrix
        V_new = th.zeros(*v.shape[:-1], n, n, dtype=v.dtype, device=v.device)

        # Assign the diagonal elements
        diag_indices = th.arange(n)
        V_new[..., diag_indices, diag_indices] = diag_part

        # Indices for the lower triangular part
        tril_indices = th.tril_indices(n, n, offset=-1)

        # Assign the off-diagonal elements from the tril_part
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part

        return V_new

class SPDAffineInvariantMetric(SPDMatrices):
    """ Affine Invariant Metrics """
    def __init__(self, n):
        super(__class__, self).__init__(n)

    def GyroMLR(self,S,P,W):
        P_invsqrtm = sym_invsqrtm.apply(P)
        item1 = sym_logm.apply(P_invsqrtm @ S @ P_invsqrtm)
        item2 = sym_logm.apply(W)
        vec = self.inner_product(item1,item2)
        return vec.sum(-1)

    def v2V(self, v):
        # Separate the diagonal and lower triangular parts
        dim=v.shape[-1]
        n = int((-1 + math.sqrt(1 + 8 * dim)) / 2)
        diag_part = v[..., :n]  # First n elements are the diagonal
        tril_part = v[..., n:]  # The rest are the lower triangular part (excluding diagonal)

        # Create an empty tensor to store the symmetric matrix
        V_new = th.zeros(*v.shape[:-1], n, n, dtype=v.dtype, device=v.device)

        # Assign the diagonal elements
        diag_indices = th.arange(n)
        V_new[..., diag_indices, diag_indices] = diag_part

        # Indices for the lower triangular part
        tril_indices = th.tril_indices(n, n, offset=-1)

        # Assign the off-diagonal elements from the tril_part
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part

        return V_new + V_new.tril(-1).transpose(-2, -1)

    def expmap_I(self,V):
        return sym_expm.apply(V)




