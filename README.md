# witten_lw2

This module provides functions for computing the weighted negative homogeneous Sobolev norm, which 
is a linearization of the quadratic Wasserstein distance $W_2$ (see for example
https://en.wikipedia.org/wiki/Wasserstein_metric) between
two nonnegative distributions of equal mass defined on the unit square $[0,1]^2 \subset \mathbb{R}^2$

The primary numerical task for this local approximation of the $W_2$ metric
distance is the numerical solution to a pde of the form
$$
(-\Delta + V) \psi = u
$$
where $V$ is a potential function.

For a detailed description of the numerical algorithm used in this
module as well as the corresponding analysis, see [arXiv].



