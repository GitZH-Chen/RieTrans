import math
import torch as th
import torch.nn as nn

from .sym_functionals import sym_powm,sym_logm,sym_invm,sym_sqrtm,sym_expm,sym_powm_reeig,sym_reeig_eigh,sym_invsqrtm2
from .functional import inner_product,trace,tril_half_diag,Lyapunov_eig_solver

tril_param_metrics = {'AIM','PEM'}
bi_param_metrics = {'LEM'}
single_param_metrics = {'LCM','BWM'}
tril_metrics = {'LCM'}

class SPDMatrices(nn.Module):
    """Computation for SPD data with [...,n,n]"""
    def __init__(self, n,power=1.):
        super().__init__()
        self.n=n; self.dim = int(n * (n + 1) / 2)
        self.register_buffer('power', th.tensor(power))
        self.register_buffer('I', th.eye(n))

        if power == 0:
            raise Exception('power should not be zero with power={:.4f}'.format(power))
        self.sgn_power = -1 if self.power < 0 else 1

    def spd_pow(self, S,power):
        r""" computing S^{\theta}"""
        if power == 2.:
            Power_S = S.matmul(S)
        elif power == 1.:
            Power_S = S
        else:
            Power_S = sym_powm.apply(S, power)
        return Power_S

    def RMLR(self, S, P, A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        """
        raise NotImplementedError

    def undirectional_RMLR(self, S, Z, gamma, is_phi=True):
        r""" without power deformation
        is_phi: for pullback metrics (LEM,LCM,PEM) and AIM;
                this can simplify the calculation together with is_phi_inv in SPDLinear
        """
        raise NotImplementedError

    def v2V(self, v):
        """
        transfer a vector tensor [...,n(n+1)/2] into a symmetric matrix
        the first n elements are the diagonal elements, and the rest are the tril triangular parts
        """
        raise NotImplementedError

    def V2SPD(self, V):
        """transforming V to SPD, used in SPDLinear,
            mostly is related to Exp_{E}, but could be further simplified with orthonormal bases
        """
        return self.expmap_I(V)

    def expmap_I(self, v):
        """exponential map at the identity"""
        print("Not yet implemented")

    def SPDLinear(self, S, Z, gamma,is_phi_inv=True):
        """ without power deformation
        SPDLinear:
            generating A by parallel transportation or the differential of Lie groups translation of Z
            generating P by Exp_{I}(\gamma * [Z])
        Inputs:
            S: [b,c1,n1,n1] SPD
            Z: [c2,dim2,c1,n1,n1] symmetric matrices, with dimi=ni(ni+1)/2
            gamma: [c2,dim2,c1,1,1] bias
            is_phi_inv: for pullback metrics (LEM,LCM,PEM) and AIM;
                        this can simplify the calculation together with is_phi in undirectional_RMLR
        Outputs:
            S_final: [b,c2,n2,n2] SPD
        """
        v = self.undirectional_RMLR(S, Z, gamma)
        V = self.v2V(v)
        S_final = self.V2SPD(V) if is_phi_inv else V
        return S_final

class SPDOnInvariantMetric(SPDMatrices):
    r"""
    Computation for SPD data with [...,n,n], the base class of (\theta,\alpha,\beta)-EM/LEM/AIM/
    """
    def __init__(self, n, alpha=1.0, beta=0.,power=1.):
        super(__class__, self).__init__(n,power)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.alpha = alpha;self.beta = beta;
        self.p = (self.alpha + n * self.beta)**0.5
        self.q = self.alpha**0.5
        self.mu = (1 / self.p - 1 / self.q )/n

    def alpha_beta_Euc_inner_product(self, tangent_vector1, tangent_vector2):
        """"computing the O(n)-invariant Euclidean inner product"""
        if self.alpha==1. and self.beta==0.:
            X_new = inner_product(tangent_vector1, tangent_vector2)
        else:
            item1 = inner_product(tangent_vector1, tangent_vector2)
            trace_vec1 = trace(tangent_vector1)
            trace_vec2 = trace(tangent_vector2)
            item2 = th.mul(trace_vec1, trace_vec2)
            X_new = self.alpha * item1 + self.beta * item2
        return X_new

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
        if self.alpha == 1 and self.beta == 0:
            diag_values = diag_part
        else:
            diag_values = (1 / self.q) * diag_part + self.mu * diag_part.sum(dim=-1, keepdim=True)
        V_new[..., diag_indices, diag_indices] = diag_values

        # Indices for the lower triangular part
        tril_indices = th.tril_indices(n, n, offset=-1)

        # Assign the off-diagonal elements from the tril_part
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part / (2 ** 0.5 * self.q)

        return V_new + V_new.tril(-1).transpose(-2, -1)

class SPDLogEuclideanMetric(SPDOnInvariantMetric):
    r""" (\alpha,\beta)-LEM """
    def __init__(self,n,alpha=1.0, beta=0.):
        super(__class__, self).__init__(n,alpha, beta)

    def RMLR(self, S, P, A):
        P_phi = sym_logm.apply(P)
        S_phi = sym_logm.apply(S)
        X_new = self.alpha_beta_Euc_inner_product(S_phi - P_phi, A)
        return X_new

    def undirectional_RMLR(self, S, Z, gamma,is_phi=True):
        # Apply the matrix logarithm to S
        S_phi = sym_logm.apply(S) if is_phi else S
        # Compute the O(n)-invariant Euclidean inner product
        X_new = self.alpha_beta_Euc_inner_product(S_phi, Z) - gamma.squeeze(-1,-2).mul(self.alpha_beta_Euc_inner_product(Z,Z).sqrt())

        return X_new.sum(-1)

    def expmap_I(self,V):
        return sym_expm.apply(V)

class SPDAffineInvariantMetric(SPDOnInvariantMetric):
    """ Three parameters Affine Invariant Metrics """
    def __init__(self, n, alpha=1.0, beta=0.,power=1.0):
        super(__class__, self).__init__(n,alpha, beta,power)

    def RMLR(self,S,P,A):
        Power_S = self.spd_pow(S, self.power)
        invSquare_power_P = self.spd_pow(P, -self.power / 2)
        in_log = invSquare_power_P.matmul(Power_S).matmul(invSquare_power_P)
        log_data = sym_logm.apply(in_log)
        # computing inner product
        X_new = (1/self.power) *self.alpha_beta_Euc_inner_product(log_data, A)
        return X_new

    def undirectional_RMLR(self, S, Z, gamma, is_phi=None):
        # Normalize Z by its Frobenius norm
        Z_fro_norm = th.linalg.norm(Z, dim=(-1, -2), keepdim=True)  # Frobenius norm of Z, assuming (alpha,beta)=(1,0)
        Z_unit = Z / Z_fro_norm  # Normalized Z
        # Compute P_phi as gamma multiplied element-wise by Z_unit
        P_vec = -1 / 2 * gamma * Z_unit  # Broadcasting gamma to match Z's shape

        # Apply the matrix logarithm to S
        invsqrt_P = sym_expm.apply(P_vec)
        in_log = invsqrt_P.matmul(S).matmul(invsqrt_P)
        log_data = sym_logm.apply(in_log)

        # computing inner product
        X_new = self.alpha_beta_Euc_inner_product(log_data, Z)

        return X_new.sum(-1)

    def expmap_I(self,V):
        return sym_expm.apply(V)

class SPDEuclideanMetric(SPDOnInvariantMetric):
    """
    Three parameters Euclidean Metrics
    (1,1,0) standard EM, (-1,1,0) Inverse-Euclidean, (theta,1,0) power Euclidean, (theta,alpha,beta) with theta to 0 is (alpha,beta)-LEM
     """
    def __init__(self,n,alpha=1.0, beta=0.,power=1.0,eps=1e-4):
        super(SPDEuclideanMetric, self).__init__(n,alpha, beta,power)
        self.register_buffer('eps', th.tensor(eps))

    def RMLR(self,S,P,A):
        P_power=self.spd_pow(P,self.power)
        S_power = self.spd_pow(S, self.power)

        item1 = (S_power - P_power)
        X_new = 1/self.power * self.alpha_beta_Euc_inner_product(item1,A)
        return X_new
    def spd2v(self, S, Z, gamma,is_phi=True):
        # Normalize Z by its Frobenius norm
        Z_fro_norm = th.linalg.norm(Z, dim=(-1, -2), keepdim=True)  # Frobenius norm of Z
        Z_unit = Z / Z_fro_norm  # Normalized Z
        # Compute P_phi as gamma multiplied element-wise by Z_unit
        P_vec = sym_reeig_eigh.apply(self.I + self.power * gamma * Z_unit,self.eps)

        # Apply the matrix power to S
        S_phi = sym_powm.apply(S,self.power) if is_phi else S
        # Compute the O(n)-invariant Euclidean inner product
        X_new = self.alpha_beta_Euc_inner_product(S_phi - P_vec, Z)
        return X_new.sum(-1)
    def undirectional_RMLR(self, S, Z, gamma,is_phi=True):
        # differs with spd2v in a scalar 1/self.power, assuming (alpha,beta)=(1,0)
        X_new = (1/self.power) * self.spd2v(S, Z, gamma,is_phi)
        return X_new
    def V2SPD(self,V):
        return sym_powm_reeig.apply(V, 1/self.power,self.eps)

    def SPDLinear(self, S, Z, gamma,is_phi_inv=True):
        v = self.spd2v(S, Z, gamma)
        V = self.v2V(v)
        I = th.eye(V.shape[-1], dtype=V.dtype, device=V.device)
        S_final = self.V2SPD(V + I) if is_phi_inv else V + I
        return S_final

class SPDLogCholeskyMetric(SPDMatrices):
    r""" \theta-LCM """
    def __init__(self, n,power=1.):
        super(__class__, self).__init__(n,power)

    def RMLR(self, S, P, A):
        Power_S = self.spd_pow(S,self.power)
        Power_P = self.spd_pow(P,self.power)

        Chol_of_Power_S = th.linalg.cholesky(Power_S)
        Chol_of_Power_P = th.linalg.cholesky(Power_P)

        item1_diag_vec = th.log(th.diagonal(Chol_of_Power_S, dim1=-2, dim2=-1)) - th.log(th.diagonal(Chol_of_Power_P, dim1=-2, dim2=-1))
        item1 = Chol_of_Power_S.tril(-1) - Chol_of_Power_P.tril(-1) + th.diag_embed(item1_diag_vec)
        X_new = (1 / self.power) * inner_product(item1, tril_half_diag(A))

        return X_new
    def diag_phi(self,diag):
        return th.log(diag)
    def diag_phi_inv(self,diag):
        return th.exp(diag)

    def phi(self,S):
        r""" The diffeomorphism Dlog \circ chol: \spd{n} \rightarrow \tril{n} \cong \bbR^{dim}, dim = n(n+1)/2
            S: [...,n,n] SPD matrices
            return: [...,dim], concatenate diagonal and strictly lower part, with diagonal first
        """
        L = th.linalg.cholesky(S)  # Compute Cholesky decomposition, shape [..., n, n]
        n = S.shape[-1]  # Get the matrix dimension

        # Extract diagonal elements of L
        diag_elements = th.diagonal(L, dim1=-2, dim2=-1)  # Shape [..., n]
        diag_applied = self.diag_phi(diag_elements)  # Apply the function to the diagonal elements

        # Extract strictly lower triangular part of L (excluding diagonal)
        tril_indices = th.tril_indices(row=n, col=n, offset=-1)  # Strictly lower triangular (below diagonal)
        strictly_lower = L[..., tril_indices[0], tril_indices[1]]  # Shape [..., n(n-1)/2]

        # Concatenate diagonal and strictly lower part, with diagonal first
        S_lower_diag_applied = th.cat([diag_applied, strictly_lower], dim=-1)
        return S_lower_diag_applied

    def diff_chol_I_au(self,V):
        r"""auxiliary function of the differential of the Cholesky decomposition at I
            T_I\spd{n} \cong \sym{n} \cong \bbR{dim} \rightarrow T_I\cho{n} \cong \tril{n} \cong \bbR{dim}
            assuming the first n are the diagonal part, the rest are the tril part
            V: [...,dim], dim=n(n+1)/2
            return \lfloor V \rfloor + 0.5*\bbV
        """
        return th.cat([V[..., :self.n] * 0.5, V[..., self.n:]], dim=-1)

    def undirectional_RMLR(self, S, Z, gamma, is_phi=True):
        """
        Inputs:
            S: if is_phi [b,1,1,c1,n1,n1] or [b,1,c1,n1,n1] SPD
               else      [b,1,1,c1,dim1]  or [b,1,c1,dim1], phi(S_final)
            Z: [c2,dim2,c1,dim1] or [cls,c1,dim1], tril part of symmetric matrices, with dimi=ni(ni+1)/2,
                assuming the first n elements are the diagonal
            gamma: [c2,dim2,c1,1] or [cls,c1,1] bias
        Outputs:
            X_new: [b,c2,dim2]  or [b,cls]
        """
        Z_norm = th.linalg.norm(Z, dim=-1, keepdim=True)  # vector norm of Z
        Z_unit = Z / Z_norm  # Normalized Z
        P_vec = self.diff_chol_I_au(gamma * Z_unit)
        # phi(S)=Dlog \circ Chol, [...,dim]
        L = self.phi(S) if is_phi else S
        X_new=(L-P_vec).mul(self.diff_chol_I_au(Z)).sum((-1,-2)) #[b,c2,dim2] or [b,cls]

        return X_new
    def v2SPD(self,v):
        r"""chol^{-1} \circ Dexp: diffeomorphism from \tril{n2} \cong \bbR{dim2} to \spd{n2}
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
        diag_values = self.diag_phi_inv(diag_part)
        V_new[..., diag_indices, diag_indices] = diag_values

        # Indices for the lower triangular part
        tril_indices = th.tril_indices(n, n, offset=-1)

        # Assign the off-diagonal elements from the tril_part
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part

        return V_new @ V_new.transpose(-1, -2)

    def SPDLinear(self, S, Z, gamma,is_phi_inv=True):
        """
        Inputs:
            S: [b,1,1,c1,n1,n1] SPD
            Z: [c2,dim2,c1,dim1] symmetric matrices, with dimi=ni(ni+1)/2,
                                assuming the first n elements are the diagonal
            gamma: [c2,dim2,c1,1] bias
        Outputs:
            S_final: [b,c2,n2,n2] SPD if is_phi_inv else [b,c2,dim2], phi(S_final)
        """
        v = self.undirectional_RMLR(S, Z, gamma) #[b,c2,dim2] or [b,cls]
        S_final = self.v2SPD(v) if is_phi_inv else v
        return S_final

class SPDBuresWassersteinMetric(SPDOnInvariantMetric):
    r""" 2\theta-BWM """
    def __init__(self, n,power=0.5,eps=1e-4):
        super(__class__, self).__init__(n,power=power)
        self.register_buffer('eps', th.tensor(eps))


    def Log(self, point, base_point,power=0.5,omitting_factor=False):
        r"""
        (PX)^{1/2} = P^{1/2} (P^{1/2} X P^{1/2}) P^{-1/2}
        if omitting_factor = True, omit the factor 1/(|2\theta|)
        [b,c,n,n] point and base_point
        """
        if power == 0.5:
            sqrt_P = sym_sqrtm.apply(base_point)
            sqrtinv_P = sym_invm.apply(sqrt_P)
            inter_term = sqrt_P.matmul(point).matmul(sqrt_P)
            sqrt_inter_term = sym_sqrtm.apply(inter_term)
            sqrt_PX = sqrt_P.matmul(sqrt_inter_term).matmul(sqrtinv_P)
            log_P_X = sqrt_PX + sqrt_PX.transpose(-1, -2) - 2 * base_point
        else:
            power_P = sym_powm.apply(base_point,power)
            invpower_P = sym_invm.apply(power_P)
            squarepower_S = sym_powm.apply(point, 2*power)
            inter_term = power_P.matmul(squarepower_S).matmul(power_P)
            sqrt_inter_term = sym_sqrtm.apply(inter_term)
            sqrt_power_2theta_PS = power_P.matmul(sqrt_inter_term).matmul(invpower_P)
            log_P_X = sqrt_power_2theta_PS + sqrt_power_2theta_PS.transpose(-1, -2) - 2 * power_P.matmul(power_P)
        if omitting_factor:
            return log_P_X
        else:
            return 1/(2*abs(self.power)) * log_P_X

    def RMLR(self, S, P, A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        para_a, para_b are coefficients in F_{p,q}
        """
        if self.power == 0.5:
            Power_S = S;
            Power_P = P
        else:
            Power_S = self.spd_pow(S, 2 * self.power)
            Power_P = self.spd_pow(P, 2 * self.power)

        log_P_S = self.Log(Power_S,Power_P,power=0.5,omitting_factor=True)

        Chol_of_power_P = th.linalg.cholesky(Power_P)
        LAL_t = Chol_of_power_P.matmul(A).matmul(Chol_of_power_P.transpose(-1, -2))
        item2 = Lyapunov_eig_solver.apply(Power_P, LAL_t)
        X_new = (1 / (4*self.power)) * inner_product(log_P_S, item2)

        return X_new

    def aux_exp_I(self, I, V):
        """Regularized BWM exponential map at the identity.

        The identity is passed explicitly because the input and output SPD
        dimensions can differ inside ``SPDLinear``.
        """
        # Exp_I(V) = (I + 0.5 * V)^2, with ReEig enforcing its local domain.
        tmp = sym_reeig_eigh.apply(I + 0.5 * V, self.eps)
        return tmp @ tmp
    def spd2v(self, S, Z, gamma,M):
        # Normalize Z by its Frobenius norm
        Z_fro_norm = th.linalg.norm(Z, dim=(-1, -2), keepdim=True)  # Frobenius norm of Z
        Z_unit = Z / Z_fro_norm  # Normalized Z

        # Compute P_pi,Z_pi
        if M is not None:
            M_sqrt,M_invsqrt = sym_invsqrtm2.apply(M)

            P_pi = self.aux_exp_I(self.I,gamma * (M_invsqrt @ Z_unit @ M_invsqrt))
            Z_pi = M_invsqrt @ Z @ M_invsqrt
            S_pi=M_invsqrt @ S @ M_invsqrt
        else:
            P_pi = self.aux_exp_I(self.I,gamma * Z_unit)
            Z_pi = Z
            S_pi = S

        log_P_S = self.Log(S_pi, P_pi)

        Chol_P_pi = th.linalg.cholesky(P_pi)
        LZpiL_t = Chol_P_pi.matmul(Z_pi).matmul(Chol_P_pi.transpose(-1, -2))
        item2 = Lyapunov_eig_solver.apply(P_pi, LZpiL_t)
        X_new =  inner_product(log_P_S, item2)
        return X_new.sum(-1)
    def undirectional_RMLR(self, S, Z, gamma,M=None,is_phi=None):
        # differs with spd2v in a scalar 1/2
        v = self.spd2v(S, Z, gamma, M)
        return 0.5*v
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

        # Assign the off-diagonal elements from the tril_part
        tril_indices = th.tril_indices(n, n, offset=-1)
        V_new[..., tril_indices[0], tril_indices[1]] = tril_part / (2 ** 0.5)

        return V_new + V_new.tril(-1).transpose(-2, -1)
    def V2SPD(self,V,M):
        I = th.eye(V.shape[-1], dtype=V.dtype, device=V.device)
        # make sure it is in the domain,i.e., I + 0.5 *V \in \spd{n}
        tmp = self.aux_exp_I(I, V)
        if M is not None:
            M_sqrt = sym_sqrtm.apply(M)
            spd = M_sqrt @ tmp @ M_sqrt
        else:
            spd=tmp
        return spd

    def SPDLinear(self, S, Z, gamma,M_in=None,M_out=None,is_phi_inv=None):
        v = self.spd2v(S, Z, gamma,M_in)
        V = self.v2V(v)
        S_final = self.V2SPD(V,M_out)
        return S_final
