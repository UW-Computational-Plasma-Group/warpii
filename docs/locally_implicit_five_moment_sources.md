# Locally-implicit source terms for five-moment systems {#locally_implicit}

This page describes WarpII's source term handling for the FiveMoment application with Maxwell fields.
The scheme described here and implemented in the code is based on Wang et al. <sup>\ref wang2020 "[1]"</sup>

The normalized Ampère's law is
\f[ \partial_t \mathbf{E} - \frac{\omega_p \tau}{\omega_c \tau} \nabla \times \mathbf{B} = - \omega_p \tau \sum_a \mathbf{j}_a, \f]
while the momentum equation for each species \f( a \f) is
\f[ A_a \partial_t (n_a \mathbf{u}_a) + \nabla \cdot \left( A_a n_a \mathbf{u}_a \otimes \mathbf{u}_a + \mathbb{P} \right) = n_a Z_a ( \omega_p \tau \mathbf{E} + \omega_c \tau \mathbf{u}_a \otimes \mathbf{B}). \f]
The equation for current density \f$\mathbf{j}_a = Z_a n_a \mathbf{u}_a\f$ is easily obtained by multiplying the momentum equation by \f$Z_a / A_a\f$.

We apply a splitting scheme to separate the flux terms from the source terms.
The flux term update is applied using a high-order nodal Discontinuous Galerkin [scheme](@ref esdgsem).
The remaining source term update requires solving a system of equations at each DG node.
We present the equation for the case of two species for notational simplicity:
\f[ 
\partial_t \begin{pmatrix}
\mathbf{E} \\
\mathbf{j}_1 \\
\mathbf{j}_2
\end{pmatrix}
=
\begin{pmatrix}
0 & -\omega_p \tau & -\omega_p \tau \\
\frac{Z_1}{A_1}\omega_p \tau & \frac{Z_1}{A_1}\omega_c \tau \mathbb{I} \times \mathbf{B} & 0 \\
\frac{Z_2}{A_2}\omega_p \tau & 0 & \frac{Z_2}{A_2}\omega_c \tau \mathbb{I} \times \mathbf{B}
\end{pmatrix}
\begin{pmatrix}
\mathbf{E} \\
\mathbf{j}_1 \\
\mathbf{j}_2
\end{pmatrix}
=
L
\begin{pmatrix}
\mathbf{E} \\
\mathbf{j}_1 \\
\mathbf{j}_2
\end{pmatrix}
,
\f]
where \f( \mathbb{I} \f) is the \f( 3 \times 3 \f) identity tensor.
Each entry in the above matrix is a \f( 3 \times 3 \f) sub-block.
We solve this using an implicit midpoint method:
\f[
\begin{pmatrix}
\mathbf{E}^{n+1/2} \\
\mathbf{j}^{n+1/2}_1 \\
\mathbf{j}^{n+1/2}_2
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{E}^{n} \\
\mathbf{j}^{n}_1 \\
\mathbf{j}^{n}_2
\end{pmatrix}
+
\frac{\Delta t}{2} L 
\begin{pmatrix}
\mathbf{E}^{n+1/2} \\
\mathbf{j}^{n+1/2}_1 \\
\mathbf{j}^{n+1/2}_2
\end{pmatrix},
\f]
or
\f[
(I - \frac{\Delta t}{2}L) \begin{pmatrix}
\mathbf{E}^{n+1/2} \\
\mathbf{j}^{n+1/2}_1 \\
\mathbf{j}^{n+1/2}_2
\end{pmatrix}
= 
M \begin{pmatrix}
\mathbf{E}^{n+1/2} \\
\mathbf{j}^{n+1/2}_1 \\
\mathbf{j}^{n+1/2}_2
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{E}^{n} \\
\mathbf{j}^{n}_1 \\
\mathbf{j}^{n}_2
\end{pmatrix}.
\f]
This can be easily solved by inverting \f( L \f).
Then, the next timestep is obtained via
\f[
\mathbf{E}^{n+1} = 2 \mathbf{E}^{n+1/2} - \mathbf{E}^n,
\f]
and similarly for the species current densities.

\anchor wang2020 [1] Wang, Liang, Ammar H. Hakim, Jonathan Ng, Chuanfei Dong, and Kai Germaschewski. “Exact and Locally Implicit Source Term Solvers for Multifluid-Maxwell Systems.” Journal of Computational Physics 415 (August 2020): 109510. https://doi.org/10.1016/j.jcp.2020.109510.

