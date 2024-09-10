#pragma once

#include "deal.II/base/tensor.h"
#include "geometry.h"

using namespace dealii;

namespace warpii {

struct PHMaxwellConstants {
    PHMaxwellConstants(
            double c,
            double chi,
            double gamma):
        c(c), chi(chi), gamma(gamma) {}

    double c;
    double chi;
    double gamma;
};

template <int dim, typename Number>
Tensor<1, 8, Tensor<1, dim, Number>> ph_maxwell_flux(
    Tensor<1, 8, Number> state, PHMaxwellConstants constants) {
    const double c = constants.c;
    const double chi = constants.chi;
    const double gamma = constants.gamma;

    Tensor<1, 3, Number> E;
    Tensor<1, 3, Number> B;
    for (unsigned int d = 0; d < 3; d++) {
        E[d] = state[d];
        B[d] = state[d + 3];
    }
    const Number phi = state[6];
    const Number psi = state[7];

    Tensor<1, 8, Tensor<1, dim, Number>> flux;

    double c2 = c * c;

    // x-direction first
    flux[0][0] = chi * c2 * phi;
    flux[1][0] = c2 * B[2];
    flux[2][0] = -c2 * B[1];
    flux[3][0] = gamma * psi;
    flux[4][0] = -E[2];
    flux[5][0] = E[1];
    flux[6][0] = chi * E[0];
    flux[7][0] = gamma * c2 * B[0];

    // y-direction
    if (dim > 1) {
        flux[0][1] = -c2 * B[2];
        flux[1][1] = chi * c2 * phi;
        flux[2][1] = c2 * B[0];
        flux[3][1] = E[2];
        flux[4][1] = gamma * psi;
        flux[5][1] = -E[0];
        flux[6][1] = chi * E[1];
        flux[7][1] = gamma * c2 * B[1];
    }
    if (dim > 2) {
        flux[0][2] = c2 * B[1];
        flux[1][2] = -c2 * B[0];
        flux[2][2] = chi * c2 * phi;
        flux[3][2] = -E[1];
        flux[4][2] = E[0];
        flux[5][2] = gamma * psi;
        flux[6][2] = chi * E[2];
        flux[7][2] = gamma * c2 * B[2];
    }

    return flux;
}

template <int dim, typename Number>
Tensor<1, 8, Number> ph_maxwell_numerical_flux(
    Tensor<1, 8, Number> state_m, Tensor<1, 8, Number> state_p,
    Tensor<1, dim, Number> outward_normal, PHMaxwellConstants constants) {
    const double c = constants.c;
    const double chi = constants.chi;
    const double gamma = constants.gamma;

    const auto n = at_least_3d<dim, Number>(outward_normal);

    const double c2 = c * c;

    Tensor<1, 3, Number> E_m;
    Tensor<1, 3, Number> B_m;
    Tensor<1, 3, Number> E_p;
    Tensor<1, 3, Number> B_p;
    for (unsigned int d = 0; d < 3; d++) {
        E_m[d] = state_m[d];
        B_m[d] = state_m[d + 3];
        E_p[d] = state_p[d];
        B_p[d] = state_p[d + 3];
    }

    const auto n_cross_B_avg = cross_product_3d(n, 0.5 * (B_m + B_p));
    const auto B_jump_cross_n = cross_product_3d(B_p - B_m, n);
    const auto n_cross_E_avg = cross_product_3d(n, 0.5 * (E_m + E_p));
    const auto E_jump_cross_n = cross_product_3d(E_p - E_m, n);

    const auto E_avg_dot_n = 0.5 * (E_m + E_p) * n;
    const auto B_avg_dot_n = 0.5 * (B_m + B_p) * n;
    const auto E_jump_dot_n = (E_p - E_m) * n;
    const auto B_jump_dot_n = (B_p - B_m) * n;

    const auto phi_avg = 0.5 * (state_m[6] + state_p[6]);
    const auto psi_avg = 0.5 * (state_m[7] + state_p[7]);
    const auto phi_jump = state_p[6] - state_m[6];
    const auto psi_jump = state_p[7] - state_m[7];

    const auto E_flux = -c2 * n_cross_B_avg -
                        0.5 * c * cross_product_3d(n, E_jump_cross_n) +
                        (chi * c2 * phi_avg - 0.5 * chi * c * E_jump_dot_n) * n;
    const auto B_flux = n_cross_E_avg -
                        0.5 * c * cross_product_3d(n, B_jump_cross_n) +
                        (gamma * c2 * psi_avg - 0.5 * gamma * c * B_jump_dot_n) * n;

    const auto phi_flux = chi * E_avg_dot_n - 0.5 * chi * c * phi_jump;
    const auto psi_flux = gamma * c2 * B_avg_dot_n - 0.5 * gamma * c * psi_jump;

    Tensor<1, 8, Number> flux;
    for (unsigned int i = 0; i < 3; i++) {
        flux[i] = E_flux[i];
        flux[i+3] = B_flux[i];
    }
    flux[6] = phi_flux;
    flux[7] = psi_flux;

    return flux;
}

template <typename Number>
Tensor<1, 8, Number> ph_maxwell_rotate_to_frame(
        const Tensor<1, 8, Number> &q, 
        const Tensor<1, 3, Number> &normal,
        const Tensor<1, 3, Number> &tangent,
        const Tensor<1, 3, Number> &binormal
        ) {
    Tensor<1, 3, Number> E;
    Tensor<1, 3, Number> B;
    for (unsigned int d = 0; d < 3; d++) {
        E[d] = q[d];
        B[d] = q[d+3];
    }
    Tensor<1, 8, Number> result({
            E*normal, E*tangent, E*binormal,
            B*normal, B*tangent, B*binormal,
            q[6], q[7]
            });
    return result;
}

template <typename Number>
Tensor<1, 8, Number> ph_maxwell_antirotate_from_frame(
        const Tensor<1, 8, Number> &q_rotated,
        const Tensor<1, 3, Number> &normal,
        const Tensor<1, 3, Number> &tangent,
        const Tensor<1, 3, Number> &binormal
        ) {
    Tensor<1, 3, Number> E = q_rotated[0] * normal + q_rotated[1] * tangent + q_rotated[2] * binormal;
    Tensor<1, 3, Number> B = q_rotated[3] * normal + q_rotated[4] * tangent + q_rotated[5] * binormal;
    Tensor<1, 8, Number> result({
            E[0], E[1], E[2],
            B[0], B[1], B[2],
            q_rotated[6], q_rotated[7]
            });
    return result;
}

template <int dim, typename Number>
Tensor<1, 8, Number> ph_maxwell_numerical_flux_2(
    Tensor<1, 8, Number> state_m, Tensor<1, 8, Number> state_p,
    Tensor<1, dim, Number> outward_normal, PHMaxwellConstants constants) {
    const double c = constants.c;
    const double chi = constants.chi;
    const double gamma = constants.gamma;

    const auto n = at_least_3d(outward_normal);
    const auto tangent_binormal = tangent_and_binormal(n);
    const auto tangent = tangent_binormal.first;
    const auto binormal = tangent_binormal.second;

    const auto qR_m = ph_maxwell_rotate_to_frame(state_m, n, tangent, binormal);
    const auto qR_p = ph_maxwell_rotate_to_frame(state_p, n, tangent, binormal);

    const double c2 = c * c;

    Tensor<1, 3, Number> E_m;
    Tensor<1, 3, Number> B_m;
    Tensor<1, 3, Number> E_p;
    Tensor<1, 3, Number> B_p;
    for (unsigned int d = 0; d < 3; d++) {
        E_m[d] = qR_m[d];
        B_m[d] = qR_m[d + 3];
        E_p[d] = qR_p[d];
        B_p[d] = qR_p[d + 3];
    }
    const auto E_avg = 0.5 * (E_m + E_p);
    const auto B_avg = 0.5 * (B_m + B_p);
    Number phi_m = qR_m[6];
    Number phi_p = qR_p[6];
    Number psi_m = qR_m[7];
    Number psi_p = qR_p[7];
    const auto phi_avg = 0.5 * (phi_m + phi_p);
    const auto psi_avg = 0.5 * (psi_m + psi_p);

    Tensor<1, 8, Number> flux_rotated({
            chi * c2 * phi_avg + 0.5 * chi * c * (E_m[0] - E_p[0]),
            c2 * B_avg[2] + 0.5 * c * (E_m[1] - E_p[1]),
            -c2 * B_avg[1] + 0.5 * c * (E_m[2] - E_p[2]),
            gamma * psi_avg + 0.5 * gamma * c * (B_m[0] - B_p[0]),
            -E_avg[2] + 0.5 * c * (B_m[1] - B_p[1]),
            E_avg[1] + 0.5 * c * (B_m[2] - B_p[2]),
            chi * E_avg[0] + 0.5 * chi * c * (phi_m - phi_p),
            gamma * c2 * B_avg[0] + 0.5 * gamma * c * (psi_m - psi_p)
            });

    return ph_maxwell_antirotate_from_frame(flux_rotated, n, tangent, binormal);
}

}  // namespace warpii
