"""
---------------------------------------------------------------------

This module contains the following user-callable functions:

wittenlw2:
    the primary function of this module, this function 
    computed the linearized w2 distance between two inputted
    functions tabulated on a grid in R^2

wittensolve: 
    solve the linear system

wittenpotential:
    construct the potential of the pde

---------------------------------------------------------------------

The functions contained in this module are used for computing the 
linearized Wasserstein-2 (earthmover's) distance. These functions 
were written for the purposes of approximating W2 distance for 
distributions defined on a rectangular region of R^2. A primary 
motivation for these codes was the efficient evaluation of 
W2 distance between images that are closely related up to rotations. 

See [arXiv paper] for details.

"""

import numpy as np
from nfmacros import *
import time
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
from scipy.io import savemat 
import scipy.special as sp
from scipy.sparse.linalg import eigsh
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator


def main():
    testgauss()


def testgauss():
    '''
    this function exists for testing purposes. it compares
    the linearized w2 distance computed using the functions of 
    this module to the analytically available formula for 
    the distance between two gaussians. 
    '''
    
    # Parameters
    n = 2**9

    print("")
    print("Gaussian example")
    print("image dimensions = ",n+1,"x",n+1)
    
    # Gaussian example
    X,Y,sol,gcheck, dx1, dy1 = gaussian_2d_example(n)
    f = X

    t0 = time.time()
    Q = wittenpotential(f)
    t1 = time.time()
    print("wittenpotential time (seconds)=",t1 -t0)
    print("")

    t0 = time.time()
    v= wittenlw2(X,Y,Q,tol=1e-4)
    t1 = time.time()
    print("wittenlw2 time (seconds)=",t1 -t0)
    print("sol = ",sol)
    print("linear w2 = ",v)
    print("err = ",np.abs(sol - v))

    return

def gaussian_2d_example(n):
    '''
    This function exists for testing purposes. it computes
    the distance between two gaussians tabulated at equispaced
    nodes. 
    '''
    x = np.pi*np.float64(range(n+1))/n
    y = np.pi*np.float64(range(n+1))/n
    xs, ys = np.meshgrid(x, y, indexing='ij')

    # Parameters
    mx = np.float64(np.pi)/2
    sx = np.float64(np.pi)/16
    mx1 = mx + 0.002
    #sx1 = sx + 0.006
    sx1 = sx
    dx1 = mx1 - mx

    my = np.float64(np.pi)/2
    sy = np.float64(np.pi)/16
    my1 = my + 0.003
    #sy1 = sy + 0.001
    sy1 = sy 
    dy1 = my1 - my
    
    # Function Handles
    fx = lambda x: 1/(np.sqrt(2*np.pi)*sx)*np.exp(-(x-mx)**2/(2*sx**2))
    hx = lambda x: 1/(np.sqrt(2*np.pi)*sx1)*np.exp(-(x-mx1)**2/(2*sx1**2))
    fxp = lambda x: -(x-mx)/sx**2*fx(x)
    fxpp = lambda x: -1/sx**2*fx(x) + (x-mx)**2/sx**4*fx(x)

    fy = lambda y: 1/(np.sqrt(2*np.pi)*sy)*np.exp(-(y-my)**2/(2*sy**2))
    hy = lambda y: 1/(np.sqrt(2*np.pi)*sy1)*np.exp(-(y-my1)**2/(2*sy1**2))
    fyp = lambda y: -(y-my)/sy**2*fy(y)
    fypp = lambda y: -1/sy**2*fy(y) + (y-my)**2/sy**4*fy(y)

    # Evaluate
    fs = fx(xs)*fy(ys)
    hs = hx(xs)*hy(ys)
    fgs = np.sqrt((fy(ys)*fxp(xs))**2 + (fx(xs)*fyp(ys))**2)
    fds = fy(ys)*fxpp(xs) + fy(xs)*fypp(ys)
    gcheck = -(1/4)*fs**(-2)*fgs**2 + (1/2)*fs**(-1)*fds
  
    sol = np.sqrt((my1-my)**2+(mx1-mx)**2+(sx-sx1)**2+(sy-sy1)**2)
    sol = sol/(np.pi)

    return fs,hs,sol, gcheck, dx1, dy1
#
#
#
#
#
################################################################################
#                           END OF TEST CODE
#################################################################################
#
#
#
#
#
def wittenlw2(X,Y,Q,tol=1e-10,maxiter=100,printing=False):
    '''
    computed the linearized W2 distance between the two-dimensional 
    array X and the two-dimensional array Y

    Parameters
    ----------
    X: array_like
        function tabulations on a grid
    Y: array_like
        function tabulations on a grid
    Q: array_like
        witten potential 
    tol: float
        accuracy of conjugate gradient solve
    maxiter: int
        maximum number of iterations of conjugate gradient 
    printing: bool
        print information 

    Returns
    -------
    v: float
        the linearized W2 distance between X and Y
    '''

    # for unit square [0,1]^2

    # make input numpy array
    X = np.array(X,dtype=np.float64)
    Y = np.array(Y,dtype=np.float64)

    # make input nonnegative
    X = np.maximum(X,0)
    Y = np.maximum(Y,0)

    # definitions
    sz = X.shape
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

    # Unpack potential
    q = Q["potential"] 
    ff = Q["fsqrt"]
    dV = np.float64(1)/n**2

    X = weight_endpoints(X,half,sz)
    Y = weight_endpoints(Y,half,sz)

    # scale inputs so they are probability distribution
    Xint = np.sum(X*dV,axis=(0,1))
    Yint = np.sum(Y*dV,axis=(0,1))
    X = X/Xint
    Y = Y/Yint

    # make into vectors
    X = np.reshape(X,-1)
    Y = np.reshape(Y,-1)

    
    # Prepare u for linear system A psi = u
    u = (X - Y)/ff
    u = np.reshape(u,-1)

    u = weight_endpoints(u,sqrt2,sz)
    u = np.reshape(u,-1)

    # Solve A psi = u
    psi = wittensolve(u,q,sz,printing,tol,maxiter)

    # Compute utilde = L*psi (since L is p.s.d. this ensures v is non-negative
    # even when cg fails to convergence in maxiter)
    utilde = Lfun(psi,q,sz)
    
    # Compute sqrt(integral psi(x) u(x) dx)
    v = np.reshape(psi*utilde,-1)
    v = np.sum(v)*dV/(np.pi**2)
    v = np.sqrt(v)

    return v



def wittenpotential(f,eps1=1e-6,tau=1e-4,printing=False):
    '''

    Parameters
    ----------
    f: array_like
        f
    eps1: float
        constant to add to f after normalization
    tau: float
        diffusion time
    printing: bool
        print stuff

    Returns
    -------
    Q: dictionary
        includes keys "potential", "fsqrt", and "sz" that...

    '''

    # make input nonnegative numpy array
    f = np.array(f,dtype=np.float64)
    f = np.maximum(f,0)

    # definitions
    sz = f.shape
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

    # volume element
    dV = np.float64(1)/n**2



    # initial time for heat diffusion
    #tau = (np.log(1/eps2)+1)/np.float64(n**2)

    # make probability distribution
    fint = np.sum(f*dV,axis=(0,1))
    f = f/fint

    # Adaptive smoothing
    T = int(1/tau) + 1
    for itr in range(T):

        # add constant to f
        f1 = f + eps1*(dV*n**2)
        ff = np.sqrt(f1) 

        # smooth by running heat equation
        ff = weight_endpoints(ff,1/sqrt2,sz)
        d = heat_kernel_multipliers(tau,sz)
        ff = apply_mult(ff,d,sz)
        ff = np.reshape(ff,sz)

        # make probability distriution
        f = ff**2
        fint = np.sum(f*dV,axis=(0,1))
        ff = ff/np.sqrt(fint)

        # null space of laplacian
        v0 = np.ones(sz,dtype=np.float64)
        v0 = weight_endpoints(v0,1/sqrt2,sz)
        v0 = np.reshape(v0,-1)
        v0 = v0/np.sqrt(n**2)


        # compute potential
        ff = np.reshape(ff,-1)
        d = lap_multipliers(sz)
        q = apply_mult(ff,d,sz)/ff

        # prepare for another iteration  if needed
        tau = 2*tau
        eps1 = 2*eps1

        # if max(q) > n**2 we need more smoothing since 
        # this could cause numerical issues.

        # more smoothing since this could cause numerical issues
        if np.max(np.abs(q)) > n**2:
            continue

        # check that operator is Positive Semi Definite 
        thresh = 1e-7    
        s_min_approx = approx_smallest(q,v0,sz,ff,printing)
        if s_min_approx > -thresh:
            break

    if printing:
        print("adaptive smoothing required",itr,"iterations")

    # package output
    Q = {"potential": q, "fsqrt": ff, "sz": sz}
    
    return Q


def wittensolve(u,q,sz,printing,tol,maxiter):
    '''

    Parameters
    ----------
    u: array_like
        u
    q: array_like
        q
    sz: array_like
        two-dimensional array with size of f
    printing: bool
        print stuff
    tol: float
        accuracy of conjugate gradient solve
    maxiter: int
        maximum iterations of conjugate gradient

    Returns
    -------
    psi: array_like
        solution to pde

    '''

    # define shape 
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

    # Null space of lap
    v0 = np.ones(sz,dtype=np.float64)
    v0 = weight_endpoints(v0,1/sqrt2,sz)
    v0 = np.reshape(v0,-1)
    v0 = v0/np.sqrt(n**2)

    # multipliers
    d = lapinv_half_multipliers(sz)
    
    # operator A
    A = LinearOperator(dims, matvec=lambda x: Afun(x,q,d,v0,sz))

    # Solve A x = u with CG
    u = apply_mult(u,d,sz)
    x, cginfo = cg(A,u,tol=tol,maxiter=maxiter)
    psi = apply_mult(x,d,sz)

    if printing:
        print("cginfo",cginfo)
        err = np.linalg.norm(A*x - u)/np.linalg.norm(u)
        print("err in cg",err)

    return psi



def Afun(x,q,d,v0,sz):
    '''
    matrix apply used for conjugate gradient
    '''
    x = x - np.dot(v0,x)*v0 + apply_mult(q*apply_mult(x,d,sz),d,sz)
    return x


def Lfun(x,q,sz):
    sqrt2 = np.sqrt(np.float64(2))
    d = lap_multipliers(sz)
    x = weight_endpoints(x,1/sqrt2,sz)
    x = np.reshape(x,-1)
    x = -apply_mult(x,d,sz) + q*x
    x = weight_endpoints(x,sqrt2,sz)
    x = np.reshape(x,-1)
    return x



def apply_mult(x,d,sz):
    '''
    multiply a vector that contains the values of a function
    tabulated on a grid 
    '''
    x = np.reshape(x,sz)
    x = dct(x,axis=0,type=1,norm="ortho")
    x = dct(x,axis=1,type=1,norm="ortho")
    x = x*d
    x = idct(x,axis=0,type=1,norm="ortho")
    x = idct(x,axis=1,type=1,norm="ortho")
    x = np.reshape(x,-1)
    return x


def lap_multipliers(sz):
    '''
    construct array of factors used for applying the 
    laplacian operator to an expansion in complex exponentials
    '''
    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fqx, fqy = np.meshgrid(fq, fq, indexing='ij')
    d = -(fqx**2 + fqy**2)
    return d


def lapinv_multipliers(sz):
    '''
    construct array of factors used for applying the inverse
    laplacian operator to an expansion in complex exponentials
    '''
    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fqx, fqy = np.meshgrid(fq, fq, indexing='ij')
    # Avoid divide by zero
    fqx[0,0] = 1 
    fqy[0,0] = 1
    d = 1/(fqx**2 + fqy**2)
    d[0,0] = 1
    return d


def lapinv_half_multipliers(sz):
    '''
    construct array of factors used for applying the square root inverse
    laplacian operator to an expansion in complex exponentials
    '''
    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fqx, fqy = np.meshgrid(fq, fq, indexing='ij')
    # Avoid divide by zero
    fqx[0,0] = 1 
    fqy[0,0] = 1
    d = 1/np.sqrt(fqx**2 + fqy**2)
    d[0,0] = 1
    return d


def heat_kernel_multipliers(tau,sz):
    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fq1, fq2 = np.meshgrid(fq, fq, indexing='ij')
    d = np.exp(-tau*(fq1**2 + fq2**2))
    return d


def weight_endpoints(x,w,sz):
    szx = x.shape
    n = sz[0] - 1
    ia = np.array(range(n+1))
    ie = np.array([0,n])
    idx = np.ix_(ia,ie)
    idy = np.ix_(ie,ia)
    x = np.reshape(x,sz)
    x[idx] = x[idx]*w
    x[idy] = x[idy]*w
    return x


def approx_smallest(q,v0,sz,ff,printing):

    # define shape 
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    d = lapinv_half_multipliers(sz)
    A = LinearOperator(dims, matvec=lambda x: Afun(x,q,d,v0,sz))
    if printing:
        print("checking smallest eigenvalue")
    sapprox = eigsh(A,k=1,which='SA',return_eigenvectors=False,tol=10)
    if printing:
        prin2("sapprox",sapprox)

    return sapprox


    
if __name__ == "__main__":
    main()

