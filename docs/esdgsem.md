# The entropy-stable discontinuous Galerkin spectral element method {#esdgsem}

WarpII solves the five-moment system of fluid equations, also known as the compressible
Euler equations.
This is a nonlinear system of hyperbolic conservation laws, and is prone to the
development of discontinuous shocks even from smooth initial data.
High-order methods famously struggle with discontinuities and under-resolved
solutions, due to the well-known Gibbs phenomenon, spectral aliasing, and other issues.
