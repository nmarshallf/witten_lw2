# witten_lw2

This module provides functions for computed a linearized 
version of the Wasserstein-2 (W2) distance (see e.g. 
https://en.wikipedia.org/wiki/Wasserstein_metric) between
two distributions defined on a rectangular region of $\mathbb{R}^2$.

The primary numerical task in approximating the linearized W2 
distance is the numerical solution to a pde of the form

$$ \Delta \psi + q \psi = u $$

where $u, q : \mathbb{R}^2 \to \mathbb{R}$. 

A detailed description of the numerical algorithm used in this
module as well as the corresponding analysis is provided here 
[arXiv].



