import sys
from numpy import array as array
import numpy as np


def prini():
    fsock = open("out.log", "w")
    fsock.close()


def prin2(msg, x):

    x = array(x)
    sz = x.shape
    m = 1
    n = 0
    if len(sz) >= 1:
        m = sz[0]
    if len(sz) >= 2:
        n = sz[1]

    prin2_helper(msg, x, m, n)
    saveout = sys.stdout
    fsock = open("out.log", "a")
    sys.stdout = fsock
    prin2_helper(msg, x, m, n)
    sys.stdout = saveout
    fsock.close()
    return


def prin2_helper(msg, x, m, n):
    sys.stdout.write(" ")
    sys.stdout.write(msg)
    sys.stdout.write(" = ({:0}, {:1}) ".format(m, n))
    sys.stdout.write("{}".format(x.dtype))
    sys.stdout.write("\n")
    n = np.maximum(n,1)
    x = np.reshape(x, (m, n))
    for i in range(m):
        sys.stdout.write("    [{:11.4E}".format(x[i, 0]))
        for j in range(1, n):
            if j % 6 == 0:
                sys.stdout.write("\n    ")
            sys.stdout.write(" {:11.4E}".format(x[i, j]))
        sys.stdout.write("]\n")
    sys.stdout.write("\n")
    return

# Always Call
prini()
if __name__ == "__main__":
    x = np.random.randn(6, 6)
    prin2("x", x)
    x = np.random.randn(12, 12)
    prin2("x", x)
