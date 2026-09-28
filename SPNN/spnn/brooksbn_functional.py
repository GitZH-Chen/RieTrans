import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

def modeig_forward(P,op,eig_mode='svd',param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPDDR matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size,channels,n,n=P.shape #batch size,channel depth,dimension
    U,S=th.zeros_like(P,device=P.device),th.zeros(batch_size,channels,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if(eig_mode=='eig'):
                s,U[i,j]=th.linalg.eigh(P[i,j]); S[i,j]=s
            elif(eig_mode=='svd'):
                U[i,j],S[i,j],_=th.svd(P[i,j])
    S_fn=op.fn(S,param)
    X=U.matmul(BatchDiag(S_fn)).matmul(U.transpose(-1,-2))
    return X,U,S,S_fn

def modeig_backward(dx,U,S,S_fn,op,param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPDDR matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    S_fn_deriv=BatchDiag(op.fn_deriv(S,param))
    SS=S[...,None].repeat(1,1,1,S.shape[-1])
    SS_fn=S_fn[...,None].repeat(1,1,1,S_fn.shape[-1])
    L=(SS_fn-SS_fn.transpose(2,3))/(SS-SS.transpose(2,3))
    L[L==-np.inf]=0; L[L==np.inf]=0; L[th.isnan(L)]=0
    L=L+S_fn_deriv
    dp=L*(U.transpose(2,3).matmul(dx).matmul(U))
    dp=U.matmul(dp).matmul(U.transpose(2,3))
    return dp

class LogEig(F):
    """
    Input P: (batch_size,h) SPDDR matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Log_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Log_op)

class ReEig(F):
    """
    Input P: (batch_size,h) SPDDR matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Re_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Re_op)

class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Exp_op,eig_mode='eig')
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Exp_op)

class SqmEig(F):
    """
    Input P: (batch_size,h) SPDDR matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqm_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqm_op)



class SqminvEig(F):
    """
    Input P: (batch_size,h) SPDDR matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqminv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqminv_op)

class PowerEig(F):
    """
    Input P: (batch_size,h) SPDDR matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P,power):
        Power_op._power=power
        X,U,S,S_fn=modeig_forward(P,Power_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Power_op),None


def geodesic(A,B,t):
    '''
    Geodesic from A to B at step t
    :param A: SPDDR matrix (n,n) to start from
    :param B: SPDDR matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPDDR matrix (n,n) along the geodesic
    '''
    M=CongrG(PowerEig.apply(CongrG(B,A,'neg'),t),A,'pos')[0,0]
    return M

def CongrG(P,G,mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if(mode=='pos'):
        GG=SqmEig.apply(G[None,None,:,:])
    elif(mode=='neg'):
        GG=SqminvEig.apply(G[None,None,:,:])
    PP=GG.matmul(P).matmul(GG)
    return PP

def LogG(x,X):
    """ Logarithmc mapping of x on the SPDDR manifold at X """
    return CongrG(LogEig.apply(CongrG(x,X,'neg')),X,'pos')

def ExpG(x,X):
    """ Exponential mapping of x on the SPD manifold at X """
    if len(X.shape)==3:
        c,n,n = X.shape
        X_expg = th.empty_like(X)
        for i in range(c):
            X_expg[i] = CongrG(ExpEig.apply(CongrG(x[i], X[i], 'neg')), X[i], 'pos')
    else:
        X_expg = CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')
    return X_expg

def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size,channels,n=P.shape #batch size,channel depth,dimension
    Q=th.zeros(batch_size,channels,n,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i,j]=P[i,j].diag()
    return Q

def karcher_step(x,G,alpha):
    '''
    One step in the Karcher flow
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mean(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G
def BaryGeom(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k=1
    alpha=1
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:]
        for _ in range(k):
            G=karcher_step(x,G,alpha)
        return G


class Log_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.log(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 1/S

class Re_op():
    """ Log function and its derivative """
    _threshold=1e-4
    @classmethod
    def fn(cls,S,param=None):
        return nn.Threshold(cls._threshold,cls._threshold)(S)
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (S>cls._threshold).double()

class Sqm_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 0.5/th.sqrt(S)

class Sqminv_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return 1/th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return -0.5/th.sqrt(S)**3

class Power_op():
    """ SPDNet-PowerEM-MLR function and its derivative """
    _power=1
    @classmethod
    def fn(cls,S,param=None):
        return S**cls._power
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (cls._power)*S**(cls._power-1)

class Exp_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.exp(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return th.exp(S)
