# witten_lw2

This module provides functions for computing a linearized 
version of the 2-Wasserstein (W2) distance (see e.g. 
https://en.wikipedia.org/wiki/Wasserstein_metric) between
two nonnegative distributions of equal mass defined on the unit square ![equation](https://latex.codecogs.com/gif.latex?[0,1]^2%20\subset%20\mathbb{R}^2)

The primary numerical task for this local approximation of W2 
distance (which is a negative order homogeneous weighted Sobolev norm) is the numerical solution to a pde of the form

![equation](https://latex.codecogs.com/gif.latex?(-\Delta%20+%20V)%20\psi%20=%20u)

where ![equation](https://latex.codecogs.com/gif.latex?V) is a potential function.

A detailed description of the numerical algorithm used in this
module as well as the corresponding analysis is provided here 
[arXiv].



