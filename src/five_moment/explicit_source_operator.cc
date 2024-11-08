#include "explicit_source_operator.h"
#include <deal.II/base/tensor.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <limits>
#include "function_eval.h"
#include "explicit_operator.h"
#include "five_moment/cell_evaluators.h"

namespace warpii {
namespace five_moment {

template <int dim>
TimestepResult FiveMomentExplicitSourceOperator<dim>::perform_forward_euler_step(
        FiveMSolutionVec &dst, const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers, 
        const TimestepRequest dt_request,
        const double t, 
        const double b, const double a, const double c) {

    auto& Mdudt_register = sol_registers.at(0);
    auto& dudt_register = sol_registers.at(1);
    dudt_register.mesh_sol = 0.0;

    for (auto& sp : species) {
        sp->general_source_term->set_time(t);
    }
    fields->get_general_source_term().func->set_time(t);

    const bool zero_out_register = true;
    discretization->mf.cell_loop(
            &FiveMomentExplicitSourceOperator<dim>::local_apply_cell, this,
            Mdudt_register.mesh_sol, u.mesh_sol, zero_out_register);

    {
        discretization->mf.cell_loop(
            &FiveMomentExplicitSourceOperator<dim>::local_apply_inverse_mass_matrix,
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
                        b * dst_i + a * u_i + c * dt_request.requested_dt * dudt_i;
                }
            });
        // dst = beta * dest + a * u + c * dt * dudt
        if (!dst.boundary_integrated_fluxes.is_empty()) {
            dst.boundary_integrated_fluxes.sadd(b, a, u.boundary_integrated_fluxes);
            dst.boundary_integrated_fluxes.sadd(
                1.0, c * dt_request.requested_dt, 
                dudt_register.boundary_integrated_fluxes);
        }
    }

    return TimestepResult::success(dt_request);
}

template <int dim>
void FiveMomentExplicitSourceOperator<dim>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    for (unsigned int species_index = 0; species_index < species.size();
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

    if (fields_enabled) {
        FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1,
                                                      5*species.size());
        MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 8, double>
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
void FiveMomentExplicitSourceOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) {

    FiveMomentCellEvaluators<dim> evaluators(mf, src, species.size(), fields_enabled);
    FiveMomentCellEvaluators<dim> writers(mf, src, species.size(), fields_enabled);

    std::vector<bool> nonzero_fluid_general_source_term;
    for (unsigned int i = 0; i < species.size(); i++) {
        nonzero_fluid_general_source_term.push_back(
                !(*species[i]->general_source_term).is_zero);
    }
    bool nonzero_field_general_source_term = !fields->get_general_source_term().is_zero;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; cell++) {
        bool call_extension_prepare = false;
        for (unsigned int i = 0; i < species.size(); i++) {
            // We need to have species value evaluations for rho_c.
            evaluators.ensure_species_evaluated(i, cell, EvaluationFlags::values);
            writers.ensure_species_evaluated(i, cell, EvaluationFlags::values);
            if (species.at(i)->has_extension_source_term) {
                call_extension_prepare = true;
            }
        }
        if (fields_enabled) {
            evaluators.ensure_fields_evaluated(cell, EvaluationFlags::values);
        }
        if (call_extension_prepare) {
            extension->prepare_source_term_evaluators(cell, evaluators);
        }

        for (unsigned int q : evaluators.species_eval(0).quadrature_point_indices()) {
            auto rho_c = VectorizedArray<double>(0.0);
            auto j = Tensor<1, 3, VectorizedArray<double>>({0.0, 0.0, 0.0});

            Tensor<1, 8, VectorizedArray<double>> field_vals({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            if (fields_enabled) {
                field_vals = evaluators.field_eval()->get_value(q);
            }
            const auto E = Tensor<1, 3, VectorizedArray<double>>({field_vals[0], field_vals[1], field_vals[2]});
            const auto B = Tensor<1, 3, VectorizedArray<double>>({field_vals[3], field_vals[4], field_vals[5]});

            for (unsigned int i = 0; i < species.size(); i++) {
                auto& phi = evaluators.species_eval(i);
                double Z = species[i]->charge;
                double A = species[i]->mass;

                if (fields_enabled) {
                    rho_c += phi.get_value(q)[0] * Z / A;
                    j[0] += phi.get_value(q)[1] * Z / A;
                    j[1] += phi.get_value(q)[2] * Z / A;
                    j[2] += phi.get_value(q)[3] * Z / A;
                }

                Tensor<1, 5, VectorizedArray<double>> source_val;
                if (species.at(i)->has_extension_source_term) {
                    source_val += extension->source_term(q, i, evaluators);
                } else if (nonzero_fluid_general_source_term[i]) {
                    const auto p = phi.quadrature_point(q);
                    source_val += evaluate_function<dim, 5>(*species[i]->general_source_term, p);
                }

                if (fields_enabled && explicit_fluid_field_coupling) {
                    const auto n = phi.get_value(q)[0] / A;
                    const auto u = euler_velocity<3>(phi.get_value(q));
                    const auto momentum_source = plasma_norm.omega_c_tau * Z * n * (E + cross_product_3d(u, B));
                    const auto energy_source = plasma_norm.omega_c_tau * Z * n * (E * u);
                    for (unsigned int d = 0; d < 3; d++) {
                        source_val[d+1] += momentum_source[d];
                    }
                    source_val[4] += energy_source;
                }
                writers.species_eval(i).submit_value(source_val, q);
            }

            if (fields_enabled) {
                Tensor<1, 8, VectorizedArray<double>> field_source;
                field_source = 0.0;
                if (nonzero_field_general_source_term) {
                    const auto p = evaluators.field_eval()->quadrature_point(q);
                    field_source += evaluate_function<dim, 8>(*(fields->get_general_source_term().func), p);
                }
                if (explicit_fluid_field_coupling) {
                    const auto fac = plasma_norm.omega_p_tau * plasma_norm.omega_p_tau / plasma_norm.omega_c_tau;
                    for (unsigned int d = 0; d < 3; d++) {
                        field_source[d] -= fac * j[d];
                    }
                }
                const double chi = fields->phmaxwell_constants().chi;
                field_source[6] += chi * plasma_norm.omega_p_tau * rho_c;
                evaluators.field_eval()->submit_value(field_source, q);
            }
        }

        for (unsigned int i = 0; i < species.size(); i++) {
            if (species.at(i)->has_extension_source_term || nonzero_fluid_general_source_term[i]
                    || explicit_fluid_field_coupling) {
                auto& phi = writers.species_eval(i);
                phi.integrate_scatter(EvaluationFlags::values, dst);
            }
        }
        if (fields_enabled) {
            evaluators.field_eval()->integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

template <int dim>
double FiveMomentExplicitSourceOperator<dim>::recommend_dt(const MatrixFree<dim> &mf, const FiveMSolutionVec &soln) {
    if (!explicit_fluid_field_coupling) {
        return std::numeric_limits<double>::infinity();
    }

    VectorizedArray<double> max_freq = 0.0;
    FEEvaluation<dim, -1, 0, 8, Number> fields(mf, 0, 1, 5*species.size());
    for (unsigned int species_index = 0; species_index < species.size(); species_index++) {
        unsigned int first_component = species_index * (5);

        FEEvaluation<dim, -1, 0, 5, Number> phi(mf, 0, 1,
                                                      first_component);

        for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(soln.mesh_sol, EvaluationFlags::values);
            fields.reinit(cell);
            fields.gather_evaluate(soln.mesh_sol, EvaluationFlags::values);
            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto solution = phi.get_value(q);
                const auto A = species[species_index]->mass;
                const auto Z = species[species_index]->charge;
                const auto n = solution[0] / A;

                const auto field_vals = fields.get_value(q);
                const auto B = Tensor<1, 3, VectorizedArray<double>>({
                        field_vals[3], field_vals[4], field_vals[5]});
                const auto B_norm = B.norm();

                const auto proton_plasma_freq = plasma_norm.omega_p_tau;
                const auto species_plasma_freq = proton_plasma_freq * std::abs(Z) * std::sqrt(n / A);

                const auto proton_cyclotron_freq = plasma_norm.omega_c_tau;
                const auto species_cyclotron_freq = proton_cyclotron_freq * B_norm * std::abs(Z) / A;

                max_freq = std::max(max_freq, species_plasma_freq);
                max_freq = std::max(max_freq, species_cyclotron_freq);
            }
        }
    }

    double max_freq_single = 0.0;
    for (unsigned int lane = 0; lane < VectorizedArray<double>::size(); lane++) {
        max_freq_single = std::max(max_freq_single, max_freq[lane]);
    }
    double max_freq_all = Utilities::MPI::max(max_freq_single, MPI_COMM_WORLD);

    return 0.2 / max_freq_all;
}

template class FiveMomentExplicitSourceOperator<1>;
template class FiveMomentExplicitSourceOperator<2>;

}
}  // namespace warpii
