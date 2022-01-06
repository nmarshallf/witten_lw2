"""
---------------------------------------------------------------------

This module contains the following user-callable functions:

linear_w2:
    the primary function of this module, this function 
    computed the linearized w2 distance between two inputted
    functions tabulated on a grid in R^2

solve_pde: 
    solve the pde
    $$-\Delta \psi + q \psi = u$$

potential:
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
    the linearized w2 distance between two gaussians computed 
    using the functions of this module to the analytically 
    available formula. 
    '''
    
    # Parameters
    n = 2**9

    print("")
    print("Gaussian example")
    print("image dimensions = ",n+1,"x",n+1)
    pi = np.float64(np.pi)

    # first gaussian
    mu1 = np.array([pi/2, pi/2])
    std1 = np.array([pi/16, pi/16])

    # stds
    mu2 = np.array([pi/2 + 0.002, pi/2 + 0.003])
    std2 = np.array([pi/16 + 0.001, pi/16 + 0.002])

    X, Y, gcheck, dx1, dy1 = gaussian_2d_example(n, mu1, mu2, std1, std2)
    sol = gaussian_formula(mu1, mu2, std1, std2)
                                                   
    f = X

    t0 = time.time()
    q, ff = potential(f)
    t1 = time.time()
    print("potential time (seconds)=",t1 -t0)
    print("")

    t0 = time.time()
    v= linear_w2(X,Y,q,ff,tol=1e-4)
    t1 = time.time()
    print("linear_w2 time (seconds)=",t1 -t0)
    print("sol = ",sol)
    print("linear w2 = ",v)
    print("err = ",np.abs(sol - v))

    return


def gaussian_formula(mu1, mu2, std1, std2):
    '''
    This function exists for testing purposes. it computes
    the distance between two gaussians tabulated at equispaced
    nodes. 
    '''

    # unpack means
    sx, sy = std1
    sx1, sy1 = std2
    mx, my = mu1
    mx1, my1 = mu2

    # differences
    dmux, dmuy = mu2 - mu1
    dstdx, dstdy = std2 - std1

    # compute w2 distance using formula
    sol = np.sqrt(dmuy**2 + dmux**2 + dstdx**2 + dstdy**2)
    sol /= np.pi

    return sol 


def gaussian_2d_example(n, mu1, mu2, std1, std2):
    '''
    This function exists for testing purposes. it computes
    the distance between two gaussians tabulated at equispaced
    nodes. 
    '''

    x = np.pi*np.float64(range(n+1))/n
    y = np.pi*np.float64(range(n+1))/n
    xs, ys = np.meshgrid(x, y, indexing='ij')

    # unpack means
    sx, sy = std1
    sx1, sy1 = std2
    mx, my = mu1
    mx1, my1 = mu2
    dx1, dy1 = mu2 - mu1

    # function handles for tabulating gaussians
    fx = lambda x: 1/(np.sqrt(2*np.pi)*sx)*np.exp(-(x-mx)**2/(2*sx**2))
    hx = lambda x: 1/(np.sqrt(2*np.pi)*sx1)*np.exp(-(x-mx1)**2/(2*sx1**2))
    fxp = lambda x: -(x-mx)/sx**2*fx(x)
    fxpp = lambda x: -1/sx**2*fx(x) + (x-mx)**2/sx**4*fx(x)

    fy = lambda y: 1/(np.sqrt(2*np.pi)*sy)*np.exp(-(y-my)**2/(2*sy**2))
    hy = lambda y: 1/(np.sqrt(2*np.pi)*sy1)*np.exp(-(y-my1)**2/(2*sy1**2))
    fyp = lambda y: -(y-my)/sy**2*fy(y)
    fypp = lambda y: -1/sy**2*fy(y) + (y-my)**2/sy**4*fy(y)

    # evaulate functions at equispaced grid
    fs = fx(xs)*fy(ys)
    hs = hx(xs)*hy(ys)
    fgs = np.sqrt((fy(ys)*fxp(xs))**2 + (fx(xs)*fyp(ys))**2)
    fds = fy(ys)*fxpp(xs) + fy(xs)*fypp(ys)
    gcheck = -(1/4)*fs**(-2)*fgs**2 + (1/2)*fs**(-1)*fds
  
    return fs,hs,gcheck, dx1, dy1



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
def linear_w2(X, Y, q, ff, tol=1e-10, maxiter=100, verbose=False):
    '''
    computed the linearized W2 distance between the two-dimensional 
    array X and the two-dimensional array Y with witten potential
    q

    Parameters
    ----------
    X: array_like
        n x n array of function tabulations on a grid
    Y: array_like
        n x n array of function tabulations on a grid
    q: array_like
        n x n array of function tabulations of witten potential 
    tol: float
        accuracy of conjugate gradient solve
    maxiter: int
        maximum number of iterations of conjugate gradient 
    verbose: bool
        print more information if True

    Returns
    -------
    v: float
        the linearized W2 distance between X and Y
    '''

    # for unit square [0,1]^2

    # make input numpy array
    X = np.array(X,dtype=np.float64)
    Y = np.array(Y,dtype=np.float64)

    # X and Y are non-negatively valued
    assert(np.min(X) >= 0)
    assert(np.min(Y) >= 0)

    # definitions
    sz = X.shape
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

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
    psi = solve_pde(u,q,sz,verbose,tol,maxiter)

    # Compute utilde = L*psi (since L is p.s.d. this ensures v is non-negative
    # even when cg fails to convergence in maxiter)
    utilde = Lfun(psi,q,sz)
    
    # Compute sqrt(integral psi(x) u(x) dx)
    v = np.reshape(psi*utilde,-1)
    v = np.sum(v)*dV/(np.pi**2)
    v = np.sqrt(v)

    return v



def potential(f, eps1=1e-6, tau=1e-4, verbose=False):
    '''
    evaluate Witten potential by smoothing f via laplacian smoothing. 
    a small additive constant is added to f to avoid division by 0. 

    Parameters
    ----------
    f: array_like
        n x n array of function tabulations on a grid
        which is used to compute weighted sobolev norm
    eps1: float
        constant to add to f
    tau: float
        diffusion time (amount of smoothing)
    verbose: bool
        print stuff

    Returns
    -------
    q: array_like
        witten potential
    ff: array_like
        sqrt of f     

    '''

    # make input nonnegative numpy array
    f = np.array(f,dtype=np.float64)
    assert(np.min(f) >= 0)

    # definitions
    sz = f.shape
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

    # volume element
    dV = np.float64(1)/n**2

    # make probability distribution
    fint = np.sum(f*dV,axis=(0,1))
    f = f/fint

    # adaptive smoothing
    T = int(1 / tau) + 1
    for itr in range(T):

        # add constant to f
        f1 = f + eps1*(dV*n**2)
        ff = np.sqrt(f1) 

        # smooth by running heat equation
        ff = weight_endpoints(ff, 1/sqrt2, sz)
        d = heat_kernel_multipliers(tau, sz)
        ff = apply_mult(ff, d, sz)
        ff = np.reshape(ff, sz)

        # make probability distriution
        f = ff**2
        fint = np.sum(f * dV,axis=(0,1))
        ff = ff / np.sqrt(fint)

        # null space of laplacian
        v0 = np.ones(sz, dtype=np.float64)
        v0 = weight_endpoints(v0, 1/sqrt2, sz)
        v0 = np.reshape(v0, -1)
        v0 = v0/np.sqrt(n**2)

        # compute potential
        ff = np.reshape(ff, -1)
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
        s_min_approx = approx_smallest(q,v0,sz,verbose)
        if s_min_approx > -thresh:
            break

    if verbose:
        print("adaptive smoothing required", itr, "iterations")

    return q, ff


def solve_pde(u,q,sz,verbose,tol,maxiter):
    '''
    solve the pde 
    $$-\Delta \psi + q \psi = u$$
    

    Parameters
    ----------
    u: array_like
        right hand side
    q: array_like
        q
    sz: array_like
        two-dimensional array with size of f
    verbose: bool
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

    # multiply on left and right of operator by -laplacian^{-1/2}

    # Solve A x = u with CG
    u = apply_mult(u,d,sz)
    x, cginfo = cg(A,u,tol=tol,maxiter=maxiter)
    psi = apply_mult(x,d,sz)

    if verbose:
        print("cginfo",cginfo)
        err = np.linalg.norm(A*x - u)/np.linalg.norm(u)
        print("err in cg",err)

    return psi



def Afun(x,q,d,v0,sz):
    '''
    this function applies the matrix apply used for conjugate gradient
    
    v0 is null space vector
    
    '''
    x = x - np.dot(v0, x) * v0 + apply_mult(q * apply_mult(x, d, sz), d, sz)
    return x


def Lfun(x,q,sz):
    '''
    apply the operator $$-\Delta + q I$$ to x 
    '''

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
    Take x, an n x n array of function tabulations on a grid,
    perform a discrete cos transform, and perform pointwise 
    multiplication in coefficients of cos expansion with n x n
    array d. 
    '''

    # cos transform of x
    x = np.reshape(x,sz)
    x = dct(x,axis=0,type=1,norm="ortho")
    x = dct(x,axis=1,type=1,norm="ortho")
    
    # pointwise multiplication in dct space
    x = x*d

    # revert back to spatial domain
    x = idct(x,axis=0,type=1,norm="ortho")
    x = idct(x,axis=1,type=1,norm="ortho")
    x = np.reshape(x,-1)

    return x


def lap_multipliers(sz):
    '''
    construct n x n array of multiplicative factors used for applying the 
    laplacian operator to an expansion in complex exponentials
    '''

    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fqx, fqy = np.meshgrid(fq, fq, indexing='ij')
    d = -(fqx**2 + fqy**2)
    return d


def lapinv_multipliers(sz):
    '''
    construct n x n array of multiplicative factors used for applying the 
    inverse laplacian operator to an expansion in complex exponentials
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
    construct n x n array of multiplicative factors used for applying
    \(\Delta^{-1/2}\) to an expansion in complex exponentials
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
    '''
    construct n x n array of multiplicative factors used for applying
    heat kernel for time tau to an expansion in complex exponentials
    '''
    n = sz[0] - 1
    fq = np.array(range(n+1),dtype=np.float64)
    fq1, fq2 = np.meshgrid(fq, fq, indexing='ij')
    d = np.exp(-tau*(fq1**2 + fq2**2))
    return d


def weight_endpoints(x,w,sz):
    '''
    scale the boundary elements of the n x n array x. 
    scale each element of boundary of x 
    by w, expect for the corner points which are multiply by w^2
    
    '''
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


def approx_smallest(q,v0,sz,verbose):
    '''
    check smallest eigenvalue of matrix of Afun (up to a factor of 10) 
    '''

    # define shape 
    n = sz[0] - 1
    dims = ((n+1)**2,(n+1)**2)
    d = lapinv_half_multipliers(sz)
    A = LinearOperator(dims, matvec=lambda x: Afun(x,q,d,v0,sz))
    if verbose:
        print("checking smallest eigenvalue")
    sapprox = eigsh(A,k=1,which='SA',return_eigenvectors=False,tol=10)
    if verbose:
        prin2("sapprox",sapprox)

    return sapprox


    
if __name__ == "__main__":
    main()

