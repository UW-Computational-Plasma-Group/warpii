#include "explicit_source_operator.h"
#include <deal.II/matrix_free/fe_evaluation.h>
#include "function_eval.h"
#include "explicit_operator.h"

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentExplicitSourceOperator<dim>::perform_forward_euler_step(
        FiveMSolutionVec &dst, const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers, const double dt,
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
                        b * dst_i + a * u_i + c * dt * dudt_i;
                }
            });
        // dst = beta * dest + a * u + c * dt * dudt
        if (!dst.boundary_integrated_fluxes.is_empty()) {
            dst.boundary_integrated_fluxes.sadd(b, a, u.boundary_integrated_fluxes);
            dst.boundary_integrated_fluxes.sadd(
                1.0, c * dt, dudt_register.boundary_integrated_fluxes);
        }
    }
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

    if (fields_enabled && !fields->get_general_source_term().is_zero) {
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

    std::vector<FEEvaluation<dim, -1, 0, 5, double>> fluid_evals;
    for (unsigned int i = 0; i < species.size(); i++) {
        fluid_evals.emplace_back(mf, 0, 1, 5*i);
    }

    std::optional<FEEvaluation<dim, -1, 0, 8, double>> field_eval = fields_enabled
        ? std::make_optional<FEEvaluation<dim, -1, 0, 8, double>>(mf, 0, 1, 5*species.size())
        : std::nullopt;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; cell++) {
        for (unsigned int i = 0; i < species.size(); i++) {
            auto& phi = fluid_evals[i];
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values);
        }
        if (fields_enabled) {
            field_eval->reinit(cell);
            field_eval->gather_evaluate(src, EvaluationFlags::values);
        }

        for (unsigned int q : fluid_evals[0].quadrature_point_indices()) {
            auto rho_c = VectorizedArray<double>(0.0);

            for (unsigned int i = 0; i < species.size(); i++) {
                auto& phi = fluid_evals[i];

                if (fields_enabled) {
                    double Zi = species[i]->charge;
                    double Ai = species[i]->mass;
                    rho_c += phi.get_value(q)[0] * Zi / Ai;
                }

                if (!(*species[i]->general_source_term).is_zero) {
                    const auto p = phi.quadrature_point(q);
                    const auto source_val = evaluate_function<dim, 5>(*species[i]->general_source_term, p);
                    phi.submit_value(source_val, q);
                }
            }

            if (fields_enabled) {
                Tensor<1, 8, VectorizedArray<double>> field_source;
                field_source = 0.0;
                if (!fields->get_general_source_term().is_zero) {
                    const auto p = field_eval->quadrature_point(q);
                    field_source += evaluate_function<dim, 8>(*(fields->get_general_source_term().func), p);
                }
                const double chi = fields->phmaxwell_constants().chi;
                field_source[6] += chi * plasma_norm.omega_p_tau * rho_c;
                field_eval->submit_value(field_source, q);
            }
        }

        for (unsigned int i = 0; i < species.size(); i++) {
            if (!(*species[i]->general_source_term).is_zero) {
                auto& phi = fluid_evals[i];
                phi.integrate_scatter(EvaluationFlags::values, dst);
            }
        }
        if (fields_enabled) {
            field_eval->integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

template class FiveMomentExplicitSourceOperator<1>;
template class FiveMomentExplicitSourceOperator<2>;

}
}  // namespace warpii
