#include "solution_helper.h"

#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "function_eval.h"
namespace warpii {
namespace maxwell {

template <int dim>
void PHMaxwellSolutionHelper<dim>::project_field_quantities(
    const PHMaxwellFunc<dim> &func,
    LinearAlgebra::distributed::Vector<double> &solution) const {
    using VA = VectorizedArray<double>;

    const auto &mf = discretization->mf;

    unsigned int first_component = 0;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 8, double> inverse(
        phi);

    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices()) {
            const auto p = phi.quadrature_point(q);
            Tensor<1, 8, VectorizedArray<double>> val = evaluate_function<dim, 8>(*func.func, p);
            phi.submit_dof_value(val, q);
        }
        inverse.transform_from_q_points_to_basis(8, phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution);
    }
}

template <int dim>
double PHMaxwellSolutionHelper<dim>::compute_global_error(
    const LinearAlgebra::distributed::Vector<double>& solution, 
    Function<dim>& f,
    unsigned int component) const {
    AssertThrow(f.n_components == 8, 
            ExcMessage("The function provided to compare against must have 5 components."));
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


template class PHMaxwellSolutionHelper<1>;
template class PHMaxwellSolutionHelper<2>;

}  // namespace maxwell
}  // namespace warpii
