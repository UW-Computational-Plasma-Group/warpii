#include "nodal_dg_discretization.h"
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>

namespace warpii {

template <int dim>
void NodalDGDiscretization<dim>::reinit() {
    dof_handler.distribute_dofs(fe);

    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const AffineConstraints<double> dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
    const std::vector<Quadrature<1>> quadratures = {
        QGauss<1>(fe_degree + 2), QGaussLobatto<1>(fe_degree + 1)};

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points |
         update_values);
    additional_data.mapping_update_flags_inner_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.mapping_update_flags_boundary_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;

    mf.reinit(mapping, dof_handlers, constraints, quadratures, additional_data);
}

template <int dim>
void NodalDGDiscretization<dim>::perform_allocation(
    LinearAlgebra::distributed::Vector<double> &solution) {
    mf.initialize_dof_vector(solution);
}

template <int dim>
template <int m>
Tensor<1, m, double> NodalDGDiscretization<dim>::global_integral_squared(
        int first_component,
        const LinearAlgebra::distributed::Vector<double>&solution) {
    int exact_quadratic_integration_quadrature_no = 0;
    FEEvaluation<dim, -1, 0, m, double> phi(mf, 0, 
            exact_quadratic_integration_quadrature_no, first_component);

    Tensor<1, m, double> sum;

    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        for (unsigned int q : phi.quadrature_point_indices()) {
            const auto val = phi.get_value(q);
            auto squared = Tensor<1, m, VectorizedArray<double>>();
            for (unsigned int comp = 0; comp < m; comp++) {
                squared[comp] = val[comp] * val[comp];
            }
            phi.submit_value(squared, q);
        }
        auto cell_integral = phi.integrate_value();
        for (unsigned int lane = 0; lane < mf.n_active_entries_per_cell_batch(cell); lane++) {
            for (unsigned int comp = 0; comp < m; comp++) {
                sum[comp] += cell_integral[comp][lane];
            }
        }
    }
    sum = Utilities::MPI::sum(sum, MPI_COMM_WORLD);
    return sum;
}

template <int dim>
template <int m>
Tensor<1, m, double> NodalDGDiscretization<dim>::global_integral(
        int first_component,
        const LinearAlgebra::distributed::Vector<double>&solution) {
    // Quadrature rule no. 1, which is the QGaussLobatto(fe_degree + 1),
    // is only exact for polynomials up to 2*(fe_degree + 1) - 3 = 2*fe_degree - 1.
    // This is at least equal to fe_degree for linear and above.
    int exact_integration_quadrature_no = 1;
    FEEvaluation<dim, -1, 0, m, double> phi(mf, 0, 
            exact_integration_quadrature_no, first_component);

    Tensor<1, m, double> sum;

    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        for (unsigned int q : phi.quadrature_point_indices()) {
            phi.submit_value(phi.get_value(q), q);
        }
        auto cell_integral = phi.integrate_value();
        for (unsigned int lane = 0; lane < mf.n_active_entries_per_cell_batch(cell); lane++) {
            for (unsigned int comp = 0; comp < m; comp++) {
                sum[comp] += cell_integral[comp][lane];
            }
        }
    }
    sum = Utilities::MPI::sum(sum, MPI_COMM_WORLD);
    return sum;
}

template class NodalDGDiscretization<1>;
template class NodalDGDiscretization<2>;

template Tensor<1, 8, double> NodalDGDiscretization<1>::global_integral_squared<8>(int, const LinearAlgebra::distributed::Vector<double> &);
template Tensor<1, 8, double> NodalDGDiscretization<2>::global_integral_squared<8>(int, const LinearAlgebra::distributed::Vector<double> &);

template Tensor<1, 5, double> NodalDGDiscretization<1>::global_integral<5>(int, const LinearAlgebra::distributed::Vector<double> &);
template Tensor<1, 5, double> NodalDGDiscretization<2>::global_integral<5>(int, const LinearAlgebra::distributed::Vector<double> &);

}
