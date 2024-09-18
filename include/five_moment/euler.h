#pragma once
#include <deal.II/base/config.h>
#include <deal.II/base/tensor.h>
#include "../tensor_utils.h"
#include "../geometry.h"
#include "../utilities.h"

namespace warpii {
namespace five_moment {
using namespace dealii;

/**
 * Compute the velocity from the conserved variables.
 * Returns the first `dim` components of the velocity.
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> euler_velocity(
    const Tensor<1, 5, Number> &conserved_variables) {
    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for (unsigned int d = 0; d < dim; d++) {
        velocity[d] = conserved_variables[d + 1] * inverse_density;
    }

    return velocity;
}

/**
 * Compute Euler pressure from conserved variables
 *
 * TODO: remove dim template parameter
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number euler_pressure(
    const Tensor<1, 5, Number> &conserved_variables, double gamma) {
    const Number rho = conserved_variables[0];
    Number squared_momentum = Number(0.);
    for (unsigned int d = 0; d < 3; d++) {
        Number p_d = conserved_variables[d + 1];
        squared_momentum += p_d * p_d;
    }
    const Number kinetic_energy = squared_momentum / (2. * rho);
    const Number total_energy = conserved_variables[4];
    return (gamma - 1) * (total_energy - kinetic_energy);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, 5, dealii::Tensor<1, dim, Number>>
euler_flux(const Tensor<1, 5, Number> &q, double gamma) {
    const Tensor<1, dim, Number> velocity = euler_velocity<dim>(q);
    const Number pressure = euler_pressure<dim>(q, gamma);

    Tensor<1, 5, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; d++) {
        flux[0][d] = q[d + 1];
        for (unsigned int e = 0; e < 3; e++) {
            flux[e + 1][d] = q[e + 1] * velocity[d];
        }
        // Diagonal pressure tensor.
        flux[d + 1][d] += pressure;
        flux[4][d] = velocity[d] * (q[4] + pressure);
    }
    return flux;
}

/**
 * Lax-Friedrichs flux
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, 5, Number> euler_numerical_flux(
    const Tensor<1, 5, Number> &q_in,
    const Tensor<1, 5, Number> &q_out,
    const Tensor<1, dim, Number> &outward_normal, double gamma) {
    const auto u_in = euler_velocity<dim>(q_in);
    const auto u_out = euler_velocity<dim>(q_out);

    const auto pressure_in = euler_pressure<dim>(q_in, gamma);
    const auto pressure_out = euler_pressure<dim>(q_out, gamma);

    const auto flux_in = euler_flux<dim>(q_in, gamma);
    const auto flux_out = euler_flux<dim>(q_out, gamma);

    const auto lambda =
        0.5 * std::sqrt(std::max(
                  u_out.norm_square() + gamma * pressure_out / q_out[0],
                  u_in.norm_square() + gamma * pressure_in / q_in[0]));

    return 0.5 * (flux_in * outward_normal + flux_out * outward_normal) +
           0.5 * lambda * (q_in - q_out);
}

/**
 * Central flux
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, 5, Tensor<1, dim, Number>> euler_central_flux(
    const Tensor<1, 5, Number> &q_in,
    const Tensor<1, 5, Number> &q_out,
    double gamma) {
    return euler_flux<dim>((q_in + q_out) / 2.0, gamma);
}

/**
 * Computes the logarithmic average (b - a) / (ln(b) - ln(a)) in a robust manner.
 *
 * For ln(b) - ln(a) ~ epsilon, we want to switch to the formula (b+a)/2.
 * Multiplying both formulas through by the denominator, we have
 *
 * b - a = y(ln(b) - ln(a))
 * b + a = 2y.
 *
 * Now multiply the first by a large constant C, and take the max of both sides of
 * both equations:
 *
 * max(C(b-a), b+a) = max(2, C(ln(b) - ln(a))) * y,
 *
 * where y can come out because it is the average of two positive quantities.
 */
template <typename Number>
inline DEAL_II_ALWAYS_INLINE Number ln_avg(Number a, Number b) {
    Number diff_log = std::abs(std::log(b) - std::log(a));
    const double C = 1e6;
    Number lhs = std::max(C * std::abs(b - a), b+a);
    Number denom = std::max(C * diff_log, Number(2.0));
    return lhs / denom;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number euler_beta(
        const Tensor<1, 5, Number> q, 
        const Number p) {
    const Number rho = q[0];
    return rho / (2.0 * p);
}

/**
 * The specific entropy, s = ln(p / rho^gamma).
 */
template <int dim, typename Number>
Number euler_thermodynamic_specific_entropy(
        const Tensor<1, 5, Number> q, double gamma) {
    return std::log(euler_pressure<dim>(q, gamma)) - gamma * std::log(q[0]);
}

template <int dim, typename Number>
Number euler_mathematical_entropy(
        const Tensor<1, 5, Number> q, double gamma) {
    Number s = euler_thermodynamic_specific_entropy<dim>(q, gamma);
    return -s * q[0] / (gamma - 1.0);
}

template <int dim, typename Number>
Tensor<1, 5, Number> euler_entropy_variables(
        const Tensor<1, 5, Number> q, double gamma) {
    Number beta = euler_beta<dim>(q, euler_pressure<dim>(q, gamma));
    Number s = euler_thermodynamic_specific_entropy<dim>(q, gamma);

    Tensor<1, 3, Number> u = euler_velocity<3>(q);
    Number u2 = Number(0.0);
    for (unsigned int d = 0; d < 3; d++) {
        u2 += u[d]*u[d];
    }
    Tensor<1, 5, Number> w;
    w[0] = (gamma - s) / (gamma - 1.0) - beta * u2;
    for (unsigned int d = 0; d < 3; d++) {
        w[d+1] = 2.0*beta * u[d];
    }
    w[4] = -2.0*beta;
    return w;
}

template <int dim, typename Number>
Tensor<1, dim, Number> euler_entropy_flux(const Tensor<1, 5, Number> state, double gamma) {
    Tensor<1, dim, Number> q;
    Number rho = state[0];
    Number s = euler_thermodynamic_specific_entropy<dim>(state, gamma);
    Tensor<1, dim, Number> u = euler_velocity<dim>(state);
    for (unsigned int d = 0; d < dim; d++) {
        q[d] = -rho * u[d] * s / (gamma - 1.0);
    }
    return q;
}

/**
 * Chandrashekar
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, 5, Tensor<1, dim, Number>> euler_CH_EC_flux(
        const Tensor<1, 5, Number> &q_j,
        const Tensor<1, 5, Number> &q_l,
        double gamma) {
    const Number p_j = euler_pressure<dim>(q_j, gamma);
    const Number beta_j = euler_beta<dim>(q_j, p_j);
    const Number p_l = euler_pressure<dim>(q_l, gamma);
    const Number beta_l = euler_beta<dim>(q_l, p_l);
    const Number beta_avg = 0.5 * (beta_j + beta_l);

    const Number beta_ln = ln_avg(beta_j, beta_l);

    const Number rho_j = q_j[0];
    const auto u_j = euler_velocity<3>(q_j);
    const Number rho_l = q_l[0];
    const auto u_l = euler_velocity<3>(q_l);
    const Number rho_ln = ln_avg(rho_j, rho_l);

    const Number rho_avg = 0.5 * (rho_j + rho_l);
    const Tensor<1, 3, Number> u_avg = 0.5 * (u_j + u_l);

    const Tensor<1, 3, Number> u2_j = componentwise_product(u_j, u_j);
    const Tensor<1, 3, Number> u2_l = componentwise_product(u_l, u_l);
    const Tensor<1, 3, Number> u2_avg = 0.5 * (u2_j + u2_l);
    const Tensor<1, 3, Number> u_avg_2 = componentwise_product(u_avg, u_avg);

    const Number p_hat = rho_avg / (2.0 * beta_avg);
    const Number h_hat = 1.0 / (2.0 * beta_ln * (gamma - 1.0)) - 
        0.5 * sum(u2_avg) +
        p_hat / rho_ln + sum(u_avg_2);

    Tensor<1, 5, Tensor<1, dim, Number>> result;
    for (unsigned int d = 0; d < dim; d++) {
        result[0][d] = rho_ln * u_avg[d];
        for (unsigned int e = 0; e < 3; e++) {
            result[e+1][d] = rho_ln * u_avg[d] * u_avg[e];
        }
        result[d+1][d] += p_hat;
        result[4][d] = rho_ln * u_avg[d] * h_hat;
    }
    return result;
}

// q_l is the out state
// q_j is the in state
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, 5, Number> euler_CH_entropy_dissipating_flux(
        const Tensor<1, 5, Number> &q_j,
        const Tensor<1, 5, Number> &q_l,
        const Tensor<1, dim, Number> &outward_normal,
        double gamma) {
    const Number p_j = euler_pressure<dim>(q_j, gamma);
    const Number beta_j = euler_beta<dim>(q_j, p_j);
    const Number p_l = euler_pressure<dim>(q_l, gamma);
    const Number beta_l = euler_beta<dim>(q_l, p_l);

    const Number beta_ln = ln_avg(beta_j, beta_l);

    const Number rho_j = q_j[0];
    const auto u_j = euler_velocity<3>(q_j);
    const Number rho_l = q_l[0];
    const auto u_l = euler_velocity<3>(q_l);

    const Number rho_avg = 0.5 * (rho_j + rho_l);
    const Tensor<1, 3, Number> u_avg = 0.5 * (u_j + u_l);

    const Tensor<1, 3, Number> u2_j = componentwise_product(u_j, u_j);
    const Tensor<1, 3, Number> u2_l = componentwise_product(u_l, u_l);

    auto flux = euler_CH_EC_flux<dim>(q_j, q_l, gamma) * outward_normal;

    const Number rho_jump = rho_l - rho_j;
    const Tensor<1, 3, Number> rho_u_jump = rho_l * u_l - rho_j * u_j;
    const Tensor<1, 3, Number> u_jump = u_l - u_j;

    // Speeds of sound and local wavespeeds
    const Number c_j = std::sqrt(gamma * p_j / rho_j);
    const Number c_l = std::sqrt(gamma * p_l / rho_l);
    const Number u2_j_norm = std::sqrt(sum(u2_j));
    const Number u2_l_norm = std::sqrt(sum(u2_l));
    const Number lambda_max = std::max(u2_j_norm + c_j, u2_l_norm + c_l);

    const Number beta_inv_jump = 1.0 / beta_l - 1.0 / beta_j;
    const Tensor<1, 3, Number> u_prod = componentwise_product(u_j, u_l);

    const Tensor<1, 3, Number> u_jump_avg_prod = componentwise_product(u_jump, u_avg);

    const Number energy_stab = (1.0 / (2.0 * (gamma - 1.0) * beta_ln) + 0.5*sum(u_prod)) * rho_jump +
        rho_avg * sum(u_jump_avg_prod) + rho_avg / (2.0*(gamma - 1.0)) * beta_inv_jump;

    flux[0] -= 0.5 * lambda_max * rho_jump;
    for (unsigned int d = 0; d < 3; d++) {
        flux[d+1] -= 0.5 * lambda_max * rho_u_jump[d];
    }
    flux[4] -= 0.5 * lambda_max * energy_stab;

    return flux;
}

template <int dim, typename Number>
Tensor<1, 5, Number> euler_rotate_to_frame(
        const Tensor<1, 5, Number> &q, 
        const Tensor<1, 3, Number> &normal,
        const Tensor<1, 3, Number> &tangent,
        const Tensor<1, 3, Number> &binormal
        ) {

    Tensor<1, 3, Number> rho_u;
    rho_u[0] = q[1];
    rho_u[1] = q[2];
    rho_u[2] = q[3];

    Tensor<1, 5, Number> result;
    result[0] = q[0];

    result[1] = rho_u * normal;
    result[2] = rho_u * tangent;
    result[3] = rho_u * binormal;

    result[4] = q[4];
    return result;
}

template <int dim, typename Number> 
Tensor<1, dim, Number> only_dim(Tensor<1, 3, Number> n) {
    Tensor<1, dim, Number> result;
    for (unsigned int d = 0; d < dim; d++) {
        result[d] = n[d];
    }
    return result;
}

template <int dim, typename Number>
Tensor<1, 5, Number> euler_antirotate_from_frame(
        const Tensor<1, 5, Number> &q_rotated,
        const Tensor<1, 3, Number> &normal,
        const Tensor<1, 3, Number> &tangent,
        const Tensor<1, 3, Number> &binormal
        ) {
    Tensor<1, 5, Number> result;
    Tensor<1, 3, Number> rho_u = q_rotated[1] * normal + q_rotated[2] * tangent + q_rotated[3] * binormal;
    result[0] = q_rotated[0];
    result[1] = rho_u[0];
    result[2] = rho_u[1];
    result[3] = rho_u[2];
    result[4] = q_rotated[4];
    return result;
}

/**
 * Implements pages 362-366 from Roe's paper:
 *
 * Approximate Riemann solvers, parameter vectors, and difference
 * schemes, P.L. Roe, JCP Vol 43, Issue 2, Oct 1981, pg 357-372 
 * https://www.sciencedirect.com/science/article/pii/0021999181901285
 */
template <int dim, typename Number>
Tensor<1, 5, Number> euler_roe_flux(
        const Tensor<1, 5, Number> &q_in,
    const Tensor<1, 5, Number> &q_out,
    const Tensor<1, dim, Number> &outward_normal, double gamma, bool log) {

    const auto n = at_least_3d(outward_normal);
    const auto tangent_binormal = tangent_and_binormal(n);
    const auto tangent = tangent_binormal.first;
    const auto binormal = tangent_binormal.second;

    const auto qR_in = euler_rotate_to_frame<dim>(q_in, n, tangent, binormal);
    const auto qR_out = euler_rotate_to_frame<dim>(q_out, n, tangent, binormal);

    const auto u_in = euler_velocity<3>(qR_in);
    const auto u_out = euler_velocity<3>(qR_out);
    const auto rho_in = qR_in[0];
    const auto rho_out = qR_out[0];

    const auto rho_denom = std::sqrt(rho_in) + std::sqrt(rho_out);
    if (log) {
    SHOW(u_in);
    SHOW(u_out);
    SHOW(rho_denom);
    }
    const auto U_hat = (std::sqrt(rho_in) * u_in[0] + std::sqrt(rho_out) * u_out[0]) / rho_denom;
    const auto V_hat = (std::sqrt(rho_in) * u_in[1] + std::sqrt(rho_out) * u_out[1]) / rho_denom;
    const auto W_hat = (std::sqrt(rho_in) * u_in[2] + std::sqrt(rho_out) * u_out[2]) / rho_denom;

    const auto E_in = qR_in[4];
    const auto E_out = qR_out[4];
    const auto p_in = euler_pressure<3>(qR_in, gamma);
    const auto p_out = euler_pressure<3>(qR_out, gamma);

    const auto H_hat = ((E_in + p_in) / std::sqrt(rho_in) + (E_out + p_out) / std::sqrt(rho_out)) / rho_denom;

    if (log) {
    SHOW(U_hat);
    }

    const auto q2 = (U_hat * U_hat + V_hat * V_hat + W_hat * W_hat);
    const auto a_squared = (gamma - 1) * (H_hat - 0.5 * q2);
    const auto a = std::sqrt(a_squared);

    if (log) {
        SHOW(a);
    }

    const auto jump = qR_out - qR_in;

    // We're computing the eigendecomposition of the jump:
    // Equation (22a)
    const Number a4 = (gamma - 1.0) / a_squared * ((H_hat - q2) * jump[0] + U_hat * jump[1] + V_hat * jump[2] + W_hat * jump[3] - jump[4]);
    // (22b)
    const Number wa3 = jump[3] - W_hat * jump[1];
    // (22c)
    const Number va2 = jump[2] - V_hat * jump[0];
    // a * (22d) + (22e)
    const Number a5 = 1.0 / (2.0*a) * (a * jump[0] - U_hat * jump[0] + jump[1] - a * a4);
    // (22d)
    const Number a1 = jump[0] - a4 - a5;

    // w1 = a1 * e1
    Tensor<1, 5, Number> w1;
    w1[0] = a1;
    w1[1] = (U_hat - a) * a1;
    w1[2] = V_hat * a1;
    w1[3] = W_hat * a1;
    w1[4] = (H_hat - U_hat * a) * a1;

    if (log) {
        SHOW(w1);
    }

    // w234 = va2 * (e2/v) + wa3 * (e3/w) + a4*e4
    Tensor<1, 5, Number> w234;
    w234[0] = a4;
    w234[1] = U_hat * a4;
    w234[2] = va2 + V_hat * a4;
    w234[3] = wa3 + W_hat * a4;
    w234[4] = va2 * V_hat + wa3 * W_hat + 0.5 * a4 * q2;
    if (log) {
        SHOW(w234);
    }

    Tensor<1, 5, Number> w5;
    w5[0] = a5;
    w5[1] = (U_hat + a) * a5;
    w5[2] = V_hat * a5;
    w5[3] = W_hat * a5;
    w5[4] = (H_hat + U_hat * a) * a5;
    if (log) {
        SHOW(w5);
    }

    const auto zero = Number(0.0);
    // Left-going fluctuations
    Tensor<1, 5, Number> amdq = std::min(U_hat - a, zero) * w1 + std::min(U_hat, zero) * w234 + std::min(U_hat + a, zero) * w5;
    if (log) {
    SHOW(amdq);
    }
    // Right-going fluctuations
    Tensor<1, 5, Number> apdq = std::max(U_hat - a, zero) * w1 + std::max(U_hat, zero) * w234 + std::max(U_hat + a, zero) * w5;
    if (log) {
    SHOW(apdq);
    }

    const auto FL = euler_flux<dim>(q_in, gamma) * outward_normal;
    const auto FR = euler_flux<dim>(q_out, gamma) * outward_normal;
    if (log) {
        SHOW(FL);
    }

    auto flux = 0.5 * (FL + FR) - 0.5 * euler_antirotate_from_frame<dim>(apdq - amdq,
            n, tangent, binormal);

    //flux = euler_antirotate_from_frame<dim>(flux, n, tangent, binormal);
    if (log) {
        SHOW(flux);
    }
    return flux;
}

}  // namespace five_moment
}  // namespace warpii
