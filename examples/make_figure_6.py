import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from prini import prin2
import sys
sys.path.append("..") 
from witten_lw2 import potential, prepsqrt, embed, tabulate_gaussian_2d, \
    gaussian_lw2_formula
np.random.seed(123)


def main():

    # Parameters
    n = 2**6
    m = 40 # degree of poly
    nn = 7
    ni = nn**2
    
    pi = np.pi

    # first gaussian
    mu1 = np.array([pi / 2, pi / 2])
    std1 = np.array([pi / 16, pi / 14])
    X = gaussian(n, mu1, std1)

    # second gaussian
    mu2 = mu1 + np.array([0.001, 0.002])
    std2 = std1 + np.array([0.001, 0.003])
    Y = gaussian(n, mu2, std2)

    # third gaussian
    mu3 = mu1 + np.array([0.03, -0.02])
    std3 = std1 + np.array([-0.01, 0.02])
    Z = gaussian(n, mu3, std3)

    V, ff = potential(X)
    W = prepsqrt(X,V,m,a=1)
    PhiY = embed(X, Y, V, ff, W, tol=1e-10, maxiter=100, verbose=False)
    PhiZ = embed(X, Z, V, ff, W, tol=1e-10, maxiter=100, verbose=False)

    plt.figure()
    color_map = plt.cm.get_cmap('gray').reversed()
    plt.imshow(PhiY,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()

    plt.figure()
    color_map = plt.cm.get_cmap('gray').reversed()
    plt.imshow(PhiZ,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()

 
 

    v = np.linalg.norm(PhiY - PhiZ)

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

    plt.show()

def gaussian(n, mu1, std1):

    x = np.pi * np.float64(range(n + 1)) / n
    y = np.pi * np.float64(range(n + 1)) / n
    xs, ys = np.meshgrid(x, y, indexing="ij")

    # unpack means
    sx, sy = std1
    mx, my = mu1

    # function handles for tabulating gaussians
    fx = (
        lambda x: 1
        / (np.sqrt(2 * np.pi) * sx)
        * np.exp(-((x - mx) ** 2) / (2 * sx**2))
    )
    fy = (
        lambda y: 1
        / (np.sqrt(2 * np.pi) * sy)
        * np.exp(-((y - my) ** 2) / (2 * sy**2))
    )

    # evaulate functions at equispaced grid
    fs = fx(xs) * fy(ys)

    return fs

def cmdscale(D):
    n = len(D)
    H = np.eye(n) - np.ones((n, n))/n
    B = -H.dot(D**2).dot(H)/2
    evals, evecs = np.linalg.eigh(B)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals



if __name__ == "__main__":
    main()
