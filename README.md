# witten_lw2

This module provides functions for linearization of the quadratic Wasserstein distance W₂ (see for example
https://en.wikipedia.org/wiki/Wasserstein_metric) between
two nonnegative distributions of equal mass defined on the unit square [0,1]² ⊂ ℝ².
In particular, this module provides a code to approximate the negative weighted homogeneous 
Sobolev norm using a formulation based on the Witten Laplacian H, which is a Schrödinger operator
of the form

H = -Δ + V,

where V is a potential function. The primary numerical task for this local approximation of the W₂ metric
distance is the numerical solution to the elliptic equation

H ψ = u

For a detailed description of the numerical algorithm used in this
module as well as the corresponding analysis, see [arXiv].



