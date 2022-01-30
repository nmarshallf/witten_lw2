import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.linalg import eigh
from prini import prin2
import sys
sys.path.append("..") 
from witten_lw2 import potential, prepsqrt, embed, tabulate_gaussian_2d, \
    gaussian_lw2_formula
np.random.seed(123)


def main():

    # Parameters
    n = 2**6

    pi = np.float64(np.pi)

    # first gaussian
    mu1 = np.array([pi / 2, pi / 2])
    std1 = np.array([pi / 16, pi / 14])

    # second gaussian
    mu2 = mu1 + np.array([0.001, 0.002])
    std2 = std1 + np.array([0.001, 0.003])
    X, Y = tabulate_gaussian_2d(n, mu1, mu2, std1, std2)

    # Third
    mu3 = mu1 + np.array([0.003, -0.002])
    std3 = std1 + np.array([-0.001, 0.002])
    X, Z = tabulate_gaussian_2d(n, mu1, mu3, std1, std3)

    V, ff = potential(X)

    m = 20
    W = prepsqrt(X,V,m,a=1,verbose=True)

    phiX = embed(X, X, V, ff, W, tol=1e-10, maxiter=100, verbose=False)
    phiY = embed(X, Y, V, ff, W, tol=1e-10, maxiter=100, verbose=True)
    phiZ = embed(X, Z, V, ff, W, tol=1e-10, maxiter=100, verbose=True)


    v = np.linalg.norm(phiY- phiZ)

    prin2("v",v)
    sol = gaussian_lw2_formula(mu2, mu3, std2, std3)
    prin2("sol",sol)
    print("err = ", np.abs(sol - v))

    #t0 = time.time()
    #v = linear_w2(X, Y, V, ff, tol=1e-4)
    #t1 = time.time()
    #print("linear_w2 time (seconds)=", t1 - t0)
    #print("sol = ", sol)
    #print("linear w2 = ", v)



if __name__ == "__main__":
    main()
