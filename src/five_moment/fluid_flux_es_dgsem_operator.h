#pragma once

#include <boost/variant/variant.hpp>
#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <algorithm>

#include "../dof_utils.h"
#include "five_moment/extension.h"
#include "utilities.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "five_moment/euler.h"
#include "fluxes/subcell_finite_volume_flux.h"
#include "solution_vec.h"
#include "five_moment/species.h"
#include "fluxes/split_form_volume_flux.h"
#include "fluxes/jacobian_utils.h"
#include "../dgsem/persson_peraire_shock_indicator.h"
#include "../rk.h"

namespace warpii {
namespace five_moment {

using namespace dealii;

template <int dim>
class FluidFluxESDGSEMOperator : ForwardEulerOperator<FiveMSolutionVec> {
   public:
    FluidFluxESDGSEMOperator(
        std::shared_ptr<five_moment::Extension<dim>> extension,
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        double gas_gamma, std::vector<std::shared_ptr<Species<dim>>> species,
        bool fields_enabled)
        : extension(extension),
          discretization(discretization),
          gas_gamma(gas_gamma),
          n_species(species.size()),
          species(species),
          split_form_volume_flux(discretization, gas_gamma),
          subcell_finite_volume_flux(*discretization, gas_gamma),
          shock_indicator(discretization),
          fields_enabled(fields_enabled)
    {
    }

    TimestepResult perform_forward_euler_step(
        FiveMSolutionVec &dst, const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers, 
        const TimestepRequest dt_request,
        const double t, 
        const double b=0.0, const double a=1.0, const double c=1.0) override;

    double recommend_dt(const MatrixFree<dim, double> &mf,
                        const FiveMSolutionVec &sol);

    void apply_positivity_limiter(FiveMSolutionVec& soln);

   private:
    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range);

    void local_apply_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    template <int n_species>
    void local_apply_boundary_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range,
        std::vector<FiveMBoundaryIntegratedFluxesVector> &boundary_integrated_fluxes,
        const double t) const;

    void local_apply_positivity_limiter(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    double compute_cell_transport_speed(
        const MatrixFree<dim, double> &mf,
        const LinearAlgebra::distributed::Vector<double> &sol) const;

    /**
     * Checks if the solution is positive:
     *    - Pressure and density are everywhere positive
     */
    bool check_if_solution_is_positive(
            const MatrixFree<dim, double> &mf,
            const LinearAlgebra::distributed::Vector<double> &sol) const;

    void calculate_high_order_EC_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, 5, double> &phi,
        const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
        const FullMatrix<double> &D, unsigned int d,
        VectorizedArray<double> alpha, bool log = false) const;

    void calculate_first_order_ES_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, 5, double> &phi,
        const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
        const std::vector<double> &quadrature_weights,
        const FullMatrix<double> &Q, unsigned int d,
        VectorizedArray<double> alpha, bool log = false) const;

    std::shared_ptr<five_moment::Extension<dim>> extension;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    double gas_gamma;
    unsigned int n_species;
    std::vector<std::shared_ptr<Species<dim>>> species;
    SplitFormVolumeFlux<dim> split_form_volume_flux;
    SubcellFiniteVolumeFlux<dim> subcell_finite_volume_flux;
    PerssonPeraireShockIndicator<dim> shock_indicator;
    bool fields_enabled;
};

template <int dim>
TimestepResult FluidFluxESDGSEMOperator<dim>::perform_forward_euler_step(
    FiveMSolutionVec &dst, const FiveMSolutionVec &u,
    std::vector<FiveMSolutionVec> &sol_registers, 
    const TimestepRequest dt_request,
    const double t, 
    const double b, const double a, const double c) {
    using Iterator = typename DoFHandler<1>::active_cell_iterator;

    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);
    auto sol_before_limiting = sol_registers.at(2);
    dudt_register.mesh_sol = 0.0;

    if (!check_if_solution_is_positive(discretization->get_matrix_free(), u.mesh_sol)) {
        std::cout << "u is already negative pressure" << std::endl;
    }
    /*
    if (!check_if_solution_is_positive(discretization->get_matrix_free(), dst.mesh_sol)) {
        std::cout << "dst is already negative pressure" << std::endl;
    }
    */

    {
        for (auto sp : species) {
            for (auto &i : sp->bc_map.inflow_boundaries()) {
                i.second->set_time(t);
            }
        }

        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            cell_operation =
                [&](const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void { this->local_apply_cell(mf, dst, src, cell_range); };
        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            face_operation =
                [&](const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void { this->local_apply_face(mf, dst, src, cell_range); };
        std::vector<FiveMBoundaryIntegratedFluxesVector> &d_dt_boundary_integrated_fluxes =
            dudt_register.boundary_integrated_fluxes;
        for (auto& boundary_fluxes : d_dt_boundary_integrated_fluxes) {
            boundary_fluxes.zero();
        }
        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            boundary_operation =
                [this, &d_dt_boundary_integrated_fluxes, t](
                    const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void {
                if (n_species == 1) {
                    this->template local_apply_boundary_face<1>(mf, dst, src, cell_range,
                                                    d_dt_boundary_integrated_fluxes, t);
                } else if (n_species == 2) {
                    this->template local_apply_boundary_face<2>(mf, dst, src, cell_range,
                                                    d_dt_boundary_integrated_fluxes, t);
                } else {
                    Assert(false, ExcMessage("We have only templated up to n_species = 2."));
                }
        };

        const bool zero_out_register = true;
        discretization->mf.loop(
            cell_operation, face_operation, boundary_operation,
            Mdudt_register.mesh_sol, u.mesh_sol, zero_out_register,
            MatrixFree<dim, double>::DataAccessOnFaces::values,
            MatrixFree<dim, double>::DataAccessOnFaces::values);
    }

    {
        double dt = dt_request.requested_dt;

        for (unsigned int attempt = 0; attempt < 10; attempt++) {
            discretization->mf.cell_loop(
                &FluidFluxESDGSEMOperator<dim>::local_apply_inverse_mass_matrix,
                this, dudt_register.mesh_sol, Mdudt_register.mesh_sol,
                std::function<void(const unsigned int, const unsigned int)>(),
                [&](const unsigned int start_range, const unsigned int end_range) {
                    /* DEAL_II_OPENMP_SIMD_PRAGMA */
                    for (unsigned int i = start_range; i < end_range; ++i) {
                        const double dudt_i =
                            dudt_register.mesh_sol.local_element(i);
                        const double dst_i = dst.mesh_sol.local_element(i);
                        const double u_i = u.mesh_sol.local_element(i);
                        dst.mesh_sol.local_element(i) =
                            b * dst_i + a * u_i + c * dt * dudt_i;
                    }
                });
            // dst = b * dest + a * u + c * dt * dudt
            for (unsigned int i = 0; i < n_species; i++) {
                if (!dst.boundary_integrated_fluxes.at(i).is_empty()) {
                    dst.boundary_integrated_fluxes.at(i).sadd(b, a, u.boundary_integrated_fluxes.at(i));
                    dst.boundary_integrated_fluxes.at(i).sadd(
                        1.0, c * dt, dudt_register.boundary_integrated_fluxes.at(i));
                }
                if (dst.boundary_integrated_normal_poynting_vectors.size() != 0) {
                    dst.boundary_integrated_normal_poynting_vectors.sadd(
                            b, a, u.boundary_integrated_normal_poynting_vectors);
                }
            }

            if (check_if_solution_is_positive(discretization->get_matrix_free(), dst.mesh_sol)) {
                return TimestepResult(dt_request.requested_dt, true, dt);
            } else if (!dt_request.is_flexible) {
                std::cout << "Failing step due to non-positivity" << std::endl;
                return TimestepResult::failure(dt_request.requested_dt);
            } else {
                std::cout << "Retrying step due to non-positivity" << std::endl;
                dt *= 0.75;
            }
        }
        std::cout << "Could not find a positivity-preserving timestep after 10 attempts, giving up" << std::endl;
        return TimestepResult::failure(dt_request.requested_dt);
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::apply_positivity_limiter(FiveMSolutionVec &soln) {
    discretization->mf.cell_loop(
            &FluidFluxESDGSEMOperator<dim>::local_apply_positivity_limiter,
            this, soln.mesh_sol, soln.mesh_sol);
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);
        FEEvaluation<dim, -1, 0, 5, double> phi(mf, 0, 1,
                                                      first_component);
        MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 5, double>
            inverse(phi);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi.read_dof_values(src);

            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

            phi.set_dof_values(dst);
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) {
    unsigned int fe_degree = discretization->get_fe_degree();
    unsigned int Np = fe_degree + 1;

    const auto& fe_values = discretization->get_fe_values();
    const std::vector<double> &quadrature_weights =
        fe_values.get_quadrature().get_weights();

    FullMatrix<double> D(Np, Np);
    FullMatrix<double> Q(Np, Np);
    for (unsigned int j = 0; j < Np; j++) {
        for (unsigned int l = 0; l < Np; l++) {
            Point<dim> j_pt = fe_values.get_quadrature().point(j);
            D(j, l) = fe_values.get_fe().shape_grad(l, j_pt)[0];
            Q(j, l) = quadrature_weights[j] * D(j, l);
        }
    }

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);
        FEEvaluation<dim, -1, 0, 5, double> phi(mf, 0, 1,
                                                      first_component);
        FEEvaluation<dim, -1, 0, 5, double> phi_reader(mf, 0, 1,
                                                             first_component);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi_reader.reinit(cell);
            phi_reader.gather_evaluate(src, EvaluationFlags::values);
            phi_reader.read_dof_values(src);

            VectorizedArray<double> alpha;
            Vector<double> p_times_rho;
            p_times_rho.reinit(phi.dofs_per_component);
            for (unsigned int lane = 0; lane < VectorizedArray<double>::size(); lane++) {
                for (unsigned int dof = 0; dof < phi_reader.dofs_per_component; dof++) {
                    const auto q_dof = phi_reader.get_dof_value(dof);
                    const auto rho = q_dof[0][lane];
                    const auto p = euler_pressure<dim>(q_dof, gas_gamma)[lane];
                    p_times_rho(dof) = p * rho;
                }
                alpha[lane] = shock_indicator.compute_shock_indicator(p_times_rho);
            }

            split_form_volume_flux.calculate_flux(dst, phi, phi_reader, alpha, false);
            if (alpha.sum() > 0.0) {
                subcell_finite_volume_flux.calculate_flux(dst, phi, phi_reader, alpha, false);
            }
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);
        FEFaceEvaluation<dim, -1, 0, 5, double> phi_m(mf, true, 0, 1,
                                                            first_component);
        FEFaceEvaluation<dim, -1, 0, 5, double> phi_p(mf, false, 0, 1,
                                                            first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src, EvaluationFlags::values);

            phi_m.reinit(face);
            phi_m.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : phi_m.quadrature_point_indices()) {
                const auto n = phi_m.normal_vector(q);
                const auto flux_m =
                    euler_flux<dim>(phi_m.get_value(q), gas_gamma) * n;
                const auto flux_p =
                    euler_flux<dim>(phi_p.get_value(q), gas_gamma) * n;
                const auto numerical_flux =
                    euler_CH_entropy_dissipating_flux<dim>(
                        phi_m.get_value(q), phi_p.get_value(q),
                        phi_m.get_normal_vector(q), gas_gamma);

                phi_m.submit_value(flux_m - numerical_flux, q);
                phi_p.submit_value(numerical_flux - flux_p, q);
            }

            phi_m.integrate_scatter(EvaluationFlags::values, dst);
            phi_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

template <int dim, int n_species_static>
std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species_static> construct_face_eval_array(
        const MatrixFree<dim> &mf) {
    if constexpr (n_species_static == 1) {
        return {{ {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 0)} }};
    } else if constexpr (n_species_static == 2) {
        return {{
            {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 0)},
            {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 5)}
        }};
    } else if constexpr (n_species_static == 3) {
        return {{
            {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 0)},
            {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 5)},
            {FEFaceEvaluation<dim, -1, 0, 5, double>(mf, true, 0, 1, 10)}
        }};
    } else {
        AssertThrow(false, ExcMessage("Only supports n_species 1, 2, 3"));
    }
}

template <int dim>
template <int n_species_static>
void FluidFluxESDGSEMOperator<dim>::local_apply_boundary_face(
    const MatrixFree<dim> &mf, LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range,
    std::vector<FiveMBoundaryIntegratedFluxesVector> &d_dt_boundary_integrated_fluxes,
    const double t)
    const {

    // Set up FE evaluators
    auto fluid_evals = construct_face_eval_array<dim, n_species_static>(mf);
    auto E_field_eval = fields_enabled
        ? std::make_optional<FEFaceEvaluation<dim, -1, 0, 3, double>>(mf, true, 0, 1, 5*n_species)
        : std::nullopt;
    auto B_field_eval = fields_enabled
        ? std::make_optional<FEFaceEvaluation<dim, -1, 0, 3, double>>(mf, true, 0, 1, 5*n_species + 3)
        : std::nullopt;

    FEFaceEvaluation<dim, -1, 0, 5, double> phi_basic(mf, true, 0, 1, 0);

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        EulerBCMap<dim> &bc_map = species.at(species_index)->bc_map;

        FEFaceEvaluation<dim, -1, 0, 5, double> phi(mf, true, 0, 1, 5*species_index);
        FEFaceEvaluation<dim, -1, 0, 5, double> phi_boundary_flux_integrator(mf, true, 0, 1, 5*species_index);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            const auto boundary_id = mf.get_boundary_id(face);

            if (bc_map.is_extension_bc(boundary_id)) {
                if (fields_enabled) {
                    extension->prepare_boundary_flux_evaluators(
                            face, species_index, src, fluid_evals, *E_field_eval, *B_field_eval);
                } else {
                    extension->prepare_boundary_flux_evaluators(face, species_index, src, fluid_evals);
                }
            }
            phi.reinit(face);
            phi.gather_evaluate(src, EvaluationFlags::values);
            phi_boundary_flux_integrator.reinit(face);

            for (const unsigned int q : phi.quadrature_point_indices()) {
                const Tensor<1, 5, VectorizedArray<double>> w_m =
                    phi.get_value(q);
                const Tensor<1, dim, VectorizedArray<double>> normal =
                    phi.normal_vector(q);

                auto rho_u_dot_n = w_m[1] * normal[0];
                for (unsigned int d = 1; d < dim; d++) {
                    rho_u_dot_n += w_m[1 + d] * normal[d];
                }

                // bool at_outflow = false;
                bool compute_from_ghost = true;
                Tensor<1, 5, VectorizedArray<double>> w_p;
                Tensor<1, 5, VectorizedArray<double>> numerical_flux;
                if (bc_map.is_inflow(boundary_id)) {
                    w_p = evaluate_function<dim, 5>(
                        *bc_map.get_inflow(boundary_id),
                        phi.quadrature_point(q));
                } else if (bc_map.is_subsonic_outflow(boundary_id)) {
                    w_p = w_m;
                    w_p[4] = evaluate_function<dim>(
                        *bc_map.get_subsonic_outflow_energy(boundary_id),
                        phi.quadrature_point(q), 4);
                    // at_outflow = true;
                } else if (bc_map.is_supersonic_outflow(boundary_id)) {
                    w_p = w_m;
                    // at_outflow = true;
                } else if (bc_map.is_wall(boundary_id)) {
                    // Copy out density
                    w_p[0] = w_m[0];
                    for (unsigned int d = 0; d < dim; d++) {
                        w_p[d + 1] = w_m[d + 1] - 2.0 * rho_u_dot_n * normal[d];
                    }
                    // The velocity component in the direction of symmetry shouldn't matter,
                    // i.e. u_z for a 2d simulation, but set it to zero just in case.
                    for (unsigned int d = dim; d < 3; d++) {
                        w_p[d + 1] = 0.0;
                    }
                    w_p[4] = w_m[4];
                } else if (bc_map.is_extension_bc(boundary_id)) {
                    compute_from_ghost = false;
                    if (fields_enabled) {
                        numerical_flux = extension->boundary_flux(
                                boundary_id, q, t, species_index, 
                                fluid_evals, *E_field_eval, *B_field_eval);
                    } else {
                        numerical_flux = extension->boundary_flux(
                                boundary_id, q, t, species_index, 
                                fluid_evals);
                    }
                } else {
                    AssertThrow(
                        false,
                        ExcMessage("Unknown boundary id, did you set a "
                                   "boundary condition "
                                   "for this part of the domain boundary?"));
                }

                auto analytic_flux = euler_flux<dim>(w_m, gas_gamma) * normal;
                if (compute_from_ghost) {
                    numerical_flux = euler_roe_flux<dim>(w_m, w_p, normal, gas_gamma, false);
                }

                phi.submit_value(analytic_flux - numerical_flux, q);
                phi_boundary_flux_integrator.submit_value(numerical_flux, q);
            }
            phi.integrate_scatter(EvaluationFlags::values, dst);

            /**
             * While we are here at this face, integrate the numerical flux
             * across it for use in diagnostics.
             */
            Tensor<1, 5, VectorizedArray<double>>
                integrated_boundary_flux =
                    phi_boundary_flux_integrator.integrate_value();
            for (unsigned int lane = 0;
                 lane < mf.n_active_entries_per_face_batch(face); lane++) {
                Tensor<1, 5, double> tensor;
                for (unsigned int comp = 0; comp < 5; comp++) {
                    tensor[comp] = integrated_boundary_flux[comp][lane];
                }
                d_dt_boundary_integrated_fluxes.at(species_index).add<dim>(boundary_id, tensor);
            }
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_positivity_limiter(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    using VA = VectorizedArray<double>;

    // Used only for the area calculation
    FEEvaluation<dim, -1, 0, 1, double> phi_scalar(mf, 0, 1);

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);
        FEEvaluation<dim, -1, 0, 5, double> phi(mf, 0, 1,
                                                      first_component);
        MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 5, double>
            inverse(phi);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi_scalar.reinit(cell);
            phi.reinit(cell);

            phi.gather_evaluate(src, EvaluationFlags::values);
            VA rho_min = VA(std::numeric_limits<double>::infinity());

            for (const unsigned int q : phi_scalar.quadrature_point_indices()) {
                phi_scalar.submit_value(VA(1.0), q);
            }
            auto area = phi_scalar.integrate_value();
            for (const unsigned int q : phi.quadrature_point_indices()) {
                auto v = phi.get_value(q);
                rho_min = std::min(v[0], rho_min);
                phi.submit_value(v, q);
            }
            auto cell_avg = phi.integrate_value() / area;

            auto rho_bar = cell_avg[0];
            auto p_bar = euler_pressure<dim>(cell_avg, gas_gamma);

            //std::cout << "cell avg = " << cell_avg << std::endl;
            //std::cout << "p_bar = " << p_bar << std::endl;

            for (unsigned int v = 0; v < VA::size(); ++v) {
                if (rho_bar[v] < 0.0) {
                    AssertThrow(false,
                                ExcMessage("Cell average density was negative"));
                }
                if (p_bar[v] <= 0.0) {
                    AssertThrow(false,
                                ExcMessage("Cell average pressure was negative"));
                }
            }

            /*
             * Theta_rho calculation
             */
            auto num_rho =
                std::max(rho_bar - (1e-12 * std::max(rho_bar, VA(1.0))), VA(0.0));
            auto denom_rho = rho_bar - rho_min;
            for (unsigned int v = 0; v < VA::size(); ++v) {
                denom_rho[v] = denom_rho[v] <= 0.0 ? 1.0 : denom_rho[v];
            }
            auto theta_rho = std::min(VA(1.0), num_rho / denom_rho);

            /*
             * Theta E calculation
             *
             * First we calculate the min energy after the theta_rho scaling
             */
            phi.gather_evaluate(src, EvaluationFlags::values);
            VA p_min = VA(std::numeric_limits<double>::infinity());
            for (const unsigned int q : phi.quadrature_point_indices()) {
                auto v = phi.get_value(q);
                v[0] = theta_rho * (v[0] - rho_bar) + rho_bar;
                p_min = std::min(euler_pressure<dim>(v, gas_gamma), p_min);
            }
            auto num_E = p_bar - (1e-12 * std::max(p_bar, VA(1.0)));
            auto denom_E = p_bar - p_min;
            for (unsigned int v = 0; v < VA::size(); ++v) {
                denom_E[v] = denom_E[v] <= 0.0 ? 1.0 : denom_E[v];
            }
            auto theta_E = std::min(VA(1.0), num_E / denom_E);

            // Finally, scale the quadrature point values by theta_rho and theta_E.
            phi.gather_evaluate(src, EvaluationFlags::values);
            for (const unsigned int q : phi.quadrature_point_indices()) {
                auto v = phi.get_value(q);
                // std::cout << "v: " << v << std::endl;
                auto rho = theta_rho * (v[0] - rho_bar) + rho_bar;
                rho = theta_E * (rho - rho_bar) + rho_bar;
                v[0] = rho;
                for (unsigned int c = 1; c < 5; c++) {
                    v[c] = theta_E * (v[c] - cell_avg[c]) + cell_avg[c];
                }
                auto pressure = euler_pressure<dim>(v, gas_gamma);
                for (unsigned int vec_i = 0; vec_i < VA::size(); ++vec_i) {
                    // AssertThrow(v[dim+2][vec_i] > 1e-12, ExcMessage("Submitting
                    // negative density to quad point"));
                    if (pressure[vec_i] <= 1e-12) {
                        std::cout << "problem with: " << vec_i << std::endl;
                        std::cout << "cell avg: " << cell_avg << std::endl;
                        std::cout << "area: " << area << std::endl;
                        std::cout << "p bar: " << p_bar << std::endl;
                        std::cout << "p min: " << p_min << std::endl;
                        std::cout << "theta rho: " << theta_rho << std::endl;
                        std::cout << "theta rho: " << num_rho << std::endl;
                        std::cout << "theta rho: " << denom_rho << std::endl;
                        std::cout << "rho min: " << rho_min << std::endl;
                        std::cout << "theta E: " << theta_E << std::endl;
                        std::cout << "Submitting value: " << v << std::endl;
                    }
                    /*
                    AssertThrow(
                        rho[vec_i] > 1e-12,
                        ExcMessage("Submitting negative density to quad point"));
                    AssertThrow(
                        pressure[vec_i] > 1e-12,
                        ExcMessage("Submitting negative pressure to quad point"));
                        */
                }
                //std::cout << "v_submitted: " << v << std::endl;
                //  This overwrites the value previously submitted.
                //  See fe_evaluation.h:4995
                phi.submit_value(v, q);
            }
            phi.integrate(EvaluationFlags::values);
            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
            phi.set_dof_values(dst);

            for (unsigned int dof = 0; dof < phi.dofs_per_component; dof++) {
                const auto val = phi.get_dof_value(dof);
                const auto p = euler_pressure<1>(val, gas_gamma);
                for (unsigned int lane = 0; lane < VectorizedArray<double>::size(); lane++) {
                    if (p[lane] <= 1e-12) {
                        std::cout << "Found negative pressure!" << std::endl;
                        std::cout << "val = " << val << std::endl;
                    }
                }
            }
        }
    }
}


template <int dim>
double FluidFluxESDGSEMOperator<dim>::recommend_dt(
    const MatrixFree<dim, double> &mf, const FiveMSolutionVec &sol) {
    if (!check_if_solution_is_positive(discretization->get_matrix_free(), sol.mesh_sol)) {
        std::cout << "sol is already negative pressure" << std::endl;
    }
    double max_transport_speed = compute_cell_transport_speed(mf, sol.mesh_sol);
    unsigned int fe_degree = discretization->get_fe_degree();
    return 0.5 / (max_transport_speed * (fe_degree + 1) * (fe_degree + 1));
}

template <int dim>
double FluidFluxESDGSEMOperator<dim>::compute_cell_transport_speed(
    const MatrixFree<dim, double> &mf,
    const LinearAlgebra::distributed::Vector<double> &solution) const {
    using VA = VectorizedArray<Number>;

    Number max_transport = 0;

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);

        FEEvaluation<dim, -1, 0, 5, Number> phi(mf, 0, 1,
                                                      first_component);

        for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(solution, EvaluationFlags::values);
            VA local_max = 0.;
            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto solution = phi.get_value(q);
                const auto velocity = euler_velocity<dim>(solution);
                const auto pressure = euler_pressure<dim>(solution, gas_gamma);

                const auto inverse_jacobian = phi.inverse_jacobian(q);
                const auto convective_speed = inverse_jacobian * velocity;
                VA convective_limit = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                    convective_limit = std::max(convective_limit,
                                                std::abs(convective_speed[d]));

                const auto speed_of_sound =
                    std::sqrt(gas_gamma * pressure * (1. / floor(solution[0])));

                Tensor<1, dim, VA> eigenvector;
                for (unsigned int d = 0; d < dim; ++d) eigenvector[d] = 1.;
                for (unsigned int i = 0; i < 5; ++i) {
                    eigenvector = transpose(inverse_jacobian) *
                                  (inverse_jacobian * eigenvector);
                    VA eigenvector_norm = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                        eigenvector_norm = std::max(eigenvector_norm,
                                                    std::abs(eigenvector[d]));
                    eigenvector /= eigenvector_norm;
                }
                const auto jac_times_ev = inverse_jacobian * eigenvector;
                const auto max_eigenvalue =
                    std::sqrt((jac_times_ev * jac_times_ev) /
                              (eigenvector * eigenvector));
                local_max =
                    std::max(local_max, max_eigenvalue * speed_of_sound +
                                            convective_limit);
            }

            for (unsigned int v = 0;
                 v < mf.n_active_entries_per_cell_batch(cell); ++v) {
                for (unsigned int d = 0; d < 3; ++d)
                    max_transport = std::max(max_transport, local_max[v]);
            }
        }
    }
    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
}

template <int dim>
bool FluidFluxESDGSEMOperator<dim>::check_if_solution_is_positive(
    const MatrixFree<dim, double> &mf,
    const LinearAlgebra::distributed::Vector<double> &sol) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (5);

        FEEvaluation<dim, -1, 0, 5, Number> phi(mf, 0, 1,
                                                      first_component);

        for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(sol, EvaluationFlags::values);
            Tensor<1, 5, VectorizedArray<double>> cell_avg;
            VectorizedArray<double> area(0.);
            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto solution = phi.get_value(q);
                cell_avg += solution * phi.JxW(q);
                area += phi.JxW(q);
            }
            cell_avg = cell_avg / area;
            const auto pressure = euler_pressure<dim>(cell_avg, gas_gamma);
            for (unsigned int lane = 0; lane < VectorizedArray<double>::size(); lane++) {
                if (cell_avg[0][lane] < 0.0 || pressure[lane] < 1e-13) {
                    std::cout << "cell = " << cell << std::endl;
                    std::cout << "location = ";
                    for (unsigned int d = 0; d < dim; d++) {
                        std::cout << phi.quadrature_point(0)[d][lane] << ", ";
                    }
                    std::cout << "cell_avg = " << cell_avg << std::endl;
                    std::cout << "pressure = " << pressure << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace five_moment
}  // namespace warpii
