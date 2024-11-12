#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/communication_pattern_base.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <mpi.h>
#include "dg_solution_helper.h"

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentDGSolutionHelper<dim>::project_fluid_quantities(
    const Function<dim> &function,
    LinearAlgebra::distributed::Vector<double> &solution,
    unsigned int species_index) const {
    const auto& mf = discretization->mf;

    unsigned int first_component = species_index * 5;
    FEEvaluation<dim, -1, 0, 5, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 5, double>
        inverse(phi);

    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices()) {
            auto value = evaluate_function<dim, 5>(
                                     function, phi.quadrature_point(q));
            phi.submit_dof_value(value, q);
        }
        inverse.transform_from_q_points_to_basis(
            5, phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(solution);
    }
}

template <int dim>
void FiveMomentDGSolutionHelper<dim>::project_field_quantities(
    const Function<dim> &function,
    LinearAlgebra::distributed::Vector<double> &solution) const {
    const auto& mf = discretization->mf;

    unsigned int first_component = n_species * 5;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 8, double>
        inverse(phi);

    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices()) {
            auto value = evaluate_function<dim, 8>(
                                     function, phi.quadrature_point(q));
            phi.submit_dof_value(value, q);
        }
        inverse.transform_from_q_points_to_basis(
            8, phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(solution);
    }
}

template <int dim>
double FiveMomentDGSolutionHelper<dim>::compute_global_error(
    LinearAlgebra::distributed::Vector<double>& solution, 
    Function<dim>& f,
    unsigned int component) {
    Vector<double> difference;
    auto select = ComponentSelectFunction<dim, double>(component, discretization->get_n_components());
    VectorTools::integrate_difference(
            discretization->get_mapping(), discretization->get_dof_handler(),
            solution, f, difference,
            QGauss<dim>(discretization->get_fe_degree()), 
            VectorTools::NormType::L2_norm,
            &select);
    return VectorTools::compute_global_error(
            discretization->get_grid().triangulation,
            difference,
            VectorTools::NormType::L2_norm);
}

template <int dim>
Tensor<1, 5, double> FiveMomentDGSolutionHelper<dim>::compute_global_integral(
        LinearAlgebra::distributed::Vector<double>&solution,
        unsigned int species_index) {
    const auto& mf = discretization->mf;
    unsigned int first_component = species_index * 5;
    FEEvaluation<dim, -1, 0, 5, double> phi(mf, 0, 1, first_component);

    Tensor<1, 5, double> sum;
    for (unsigned int comp = 0; comp < 5; comp++) {
        sum[comp] = 0.0;
    }
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        for (unsigned int q : phi.quadrature_point_indices()) {
            phi.submit_value(phi.get_value(q), q);
        }
        auto cell_integral = phi.integrate_value();
        for (unsigned int lane = 0; lane < mf.n_active_entries_per_cell_batch(cell); lane++) {
            for (unsigned int comp = 0; comp < 5; comp++) {
                sum[comp] += cell_integral[comp][lane];
            }
        }
    }
    sum = Utilities::MPI::sum(sum, MPI_COMM_WORLD);
    return sum;
}

template <int dim>
Tensor<1, 2, double> FiveMomentDGSolutionHelper<dim>::compute_global_electromagnetic_energy(
        LinearAlgebra::distributed::Vector<double>&solution) {
    const auto& mf = discretization->mf;
    unsigned int first_component = n_species * 5;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);

    Tensor<1, 2, double> sum({0.0, 0.0});

    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        for (unsigned int q : phi.quadrature_point_indices()) {
            const auto val = phi.get_value(q);
            auto squared = Tensor<1, 8, VectorizedArray<double>>();
            for (unsigned int comp = 0; comp < 8; comp++) {
                squared[comp] = val[comp] * val[comp];
            }
            phi.submit_value(squared, q);
        }
        auto cell_integral = phi.integrate_value();
        for (unsigned int lane = 0; lane < mf.n_active_entries_per_cell_batch(cell); lane++) {
            const auto E2 = cell_integral[0][lane] + cell_integral[1][lane] + cell_integral[2][lane];
            const auto B2 = cell_integral[3][lane] + cell_integral[4][lane] + cell_integral[5][lane];
            const auto c = plasma_norm.omega_p_tau / plasma_norm.omega_c_tau;
            sum[0] += 0.5 * E2 / (c * c);
            sum[1] += 0.5 * B2;
        }
    }
    sum = Utilities::MPI::sum(sum, MPI_COMM_WORLD);
    return sum;
}

template <int dim>
double FiveMomentDGSolutionHelper<dim>::compute_global_electrostatic_energy(
    LinearAlgebra::distributed::Vector<double>& solution) {
    auto zero = Functions::ZeroFunction<dim>(5*n_species + 8);
    const double sqrt_result = compute_global_error(solution, zero, 5*n_species)
        + compute_global_error(solution, zero, 5*n_species + 1)
        + compute_global_error(solution, zero, 5*n_species + 2);
    return sqrt_result * sqrt_result;
}

template class FiveMomentDGSolutionHelper<1>;
template class FiveMomentDGSolutionHelper<2>;

}
}  // namespace warpii


// Explicit template instantiation is required for this because dealii only instantiates
// for up to dim=3
template Tensor<1, 4, double> dealii::Utilities::MPI::sum(
        const Tensor<1, 4, double>&,
        const MPI_Comm comm);

