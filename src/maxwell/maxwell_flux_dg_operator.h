#pragma once
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <boost/mpl/pair.hpp>
#include <vector>

#include "../dgsem/nodal_dg_discretization.h"
#include "../dof_utils.h"
#include "../rk.h"
#include "bc_map.h"
#include "fields.h"
#include "function_eval.h"
#include "geometry.h"
#include "maxwell.h"
#include "maxwell/extension.h"

using namespace dealii;

namespace warpii {

template <int dim, typename SolutionVec>
class MaxwellFluxDGOperator : ForwardEulerOperator<SolutionVec> {
   public:
    MaxwellFluxDGOperator(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        unsigned int first_component_index,
        std::shared_ptr<PHMaxwellFields<dim>> fields        )
        : discretization(discretization),
          first_component_index(first_component_index),
          fields(fields),
          constants(fields->phmaxwell_constants())
    {}

    TimestepResult perform_forward_euler_step(SolutionVec &dst, const SolutionVec &u,
                                    std::vector<SolutionVec> &sol_registers,
                                    const TimestepRequest dt, const double t,
                                    const double b = 0.0, const double a = 1.0,
                                    const double c = 1.0) override;

    double recommend_dt(const MatrixFree<dim, double> &mf,
                        const SolutionVec &sol);

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    unsigned int first_component_index;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    PHMaxwellConstants constants;
    double time;

    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_apply_boundary_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    double compute_cell_transport_speed(
        const MatrixFree<dim, double> &mf,
        const LinearAlgebra::distributed::Vector<double> &sol) const;
};

template <int dim, typename SolutionVec>
TimestepResult MaxwellFluxDGOperator<dim, SolutionVec>::perform_forward_euler_step(
    SolutionVec &dst, const SolutionVec &u,
    std::vector<SolutionVec> &sol_registers, 
    const TimestepRequest dt_request, const double t,
    const double b, const double a, const double c) {
    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);

    // bool zero_out_register = true;
    discretization->mf.loop(
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_cell,
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_face,
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_boundary_face,
        this, Mdudt_register.mesh_sol, u.mesh_sol, true,
        MatrixFree<dim, double>::DataAccessOnFaces::values,
        MatrixFree<dim, double>::DataAccessOnFaces::values);

    for (auto &bc : fields->get_bc_map().get_function_bcs()) {
        bc.second.func->set_time(t);
    }

    {
        discretization->mf.cell_loop(
            &MaxwellFluxDGOperator<
                dim, SolutionVec>::local_apply_inverse_mass_matrix,
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
    }

    return TimestepResult::success(dt_request);
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int first_component = first_component_index;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 8, double> inverse(
        phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEvaluation<dim, -1, 0, 8, double> fe_eval(mf, 0, 1,
                                                first_component_index);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         cell++) {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(src, EvaluationFlags::values);
        for (unsigned int q : fe_eval.quadrature_point_indices()) {
            const auto val = fe_eval.get_value(q);
            Tensor<1, 8, Tensor<1, dim, VectorizedArray<double>>> flux =
                ph_maxwell_flux<dim>(val, constants);
            fe_eval.submit_gradient(flux, q);
        }
        fe_eval.integrate_scatter(EvaluationFlags::gradients, dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_m(mf, true, 0, 1,
                                                      first_component_index);
    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_p(mf, false, 0, 1,
                                                      first_component_index);
    for (unsigned int face = face_range.first; face < face_range.second;
         face++) {
        fe_eval_p.reinit(face);
        fe_eval_p.gather_evaluate(src, EvaluationFlags::values);

        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src, EvaluationFlags::values);
        for (const unsigned int q : fe_eval_m.quadrature_point_indices()) {
            const auto n = fe_eval_m.normal_vector(q);

            const auto val_m = fe_eval_m.get_value(q);
            const auto val_p = fe_eval_p.get_value(q);

            Tensor<1, 8, VectorizedArray<double>> numerical_flux =
                ph_maxwell_numerical_flux<dim>(val_m, val_p, n, constants);

            fe_eval_m.submit_value(-numerical_flux, q);
            fe_eval_p.submit_value(numerical_flux, q);
        }

        fe_eval_m.integrate_scatter(EvaluationFlags::values, dst);
        fe_eval_p.integrate_scatter(EvaluationFlags::values, dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_boundary_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {

    using VA = VectorizedArray<double>;

    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_m(mf, true, 0, 1,
                                                      first_component_index);

    for (unsigned int face = face_range.first; face < face_range.second;
         face++) {
        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src, EvaluationFlags::values);

        const types::boundary_id boundary_id = mf.get_boundary_id(face);
        MaxwellBCType bc_type = fields->get_bc_map().get_bc_type(boundary_id);

        for (const unsigned int q : fe_eval_m.quadrature_point_indices()) {
            const auto n = fe_eval_m.normal_vector(q);
            const Tensor<1, 3, VA> n3d = at_least_3d<dim, VA>(n);
            const auto val_m = fe_eval_m.get_value(q);
            //const auto t_and_b = tangent_and_binormal<VA>(n3d);

            Tensor<1, 3, VA> E_m;
            Tensor<1, 3, VA> B_m;
            VA phi_m;
            VA psi_m;
            for (unsigned int d = 0; d < 3; d++) {
                E_m[d] = val_m[d];
                B_m[d] = val_m[d+3];
            }
            phi_m = val_m[6];
            psi_m = val_m[7];

            Tensor<1, 8, VA> val_bdy;
            Tensor<1, 8, VA> val_p;

            if (bc_type == MaxwellBCType::COPY_OUT) {
                val_p = val_m;
                val_p[6] = -val_m[6];
                val_p[7] = -val_m[7];
            } else if (bc_type == MaxwellBCType::PERFECT_CONDUCTOR) {
                // Just the normal component of E
                const auto E_bdy = (n3d * E_m) * n3d;
                // Just the tangential components of B
                const auto B_bdy = B_m - (n3d * B_m) * n3d;
                const auto phi_bdy = VA(0.0);
                const auto psi_bdy = VA(0.0);
                for (unsigned int d = 0; d < 3; d++) {
                    val_bdy[d] = E_bdy[d];
                    val_bdy[d+3] = B_bdy[d];
                }
                val_bdy[6] = phi_bdy;
                val_bdy[7] = psi_bdy;
                val_p = 2.0*val_bdy - val_m;
            } else if (bc_type == MaxwellBCType::DIRICHLET) {
                const auto p = fe_eval_m.quadrature_point(q);
                const auto& func = fields->get_bc_map().get_dirichlet_func(boundary_id);
                val_bdy = evaluate_function<dim, 8>(*func.func, p);
                val_p = 2.0*val_bdy - val_m;
            } else if (bc_type == MaxwellBCType::FLUX_INJECTION) {
                const auto p = fe_eval_m.quadrature_point(q);
                const auto& func = fields->get_bc_map().get_flux_injection_func(boundary_id);
                val_bdy = evaluate_function<dim, 8>(*func.func, p);
                auto B_z = val_bdy[5];

                auto c = constants.c;

                auto Ey_in = val_m[1];
                auto Bz_in = val_m[5];
                //auto alpha_1 = Bz_in / 2.0 + Ey_in / (2.0*c);
                auto alpha_2 = Bz_in / 2.0 - Ey_in / (2.0*c);
                // Copy out the outgoing characteristic variable
                auto beta_2 = alpha_2;
                // Solve for the incoming characteristic variable that makes the total
                // equal to the desired B_z
                auto beta_1 = B_z - alpha_2;
                auto Ey_out = beta_1 * c - beta_2 * c;

                // Copy out all components
                val_p = val_m;
                // Except for B
                val_p[0] = val_m[0];
                // Set E_y = cB_z, according to the eigendecomposition.
                // This corresponds to setting the right-going wave to a certain
                // value, and the left-going wave to zero.
                val_p[1] = Ey_out;
                // E_z = -cB_y, corresponding to the right-going wave.
                val_p[2] = 0.0;
                val_p[3] = val_bdy[3];
                val_p[4] = val_bdy[4];
                val_p[5] = B_z;
            }

            const auto numerical_flux = ph_maxwell_numerical_flux(val_m, val_p, n, constants);
            fe_eval_m.submit_value(-numerical_flux, q);
        }

        fe_eval_m.integrate_scatter(EvaluationFlags::values, dst);
    }
}

template <int dim, typename SolutionVec>
double MaxwellFluxDGOperator<dim, SolutionVec>::recommend_dt(
    const MatrixFree<dim, double> &mf, const SolutionVec &sol) {
    double max_transport_speed = compute_cell_transport_speed(mf, sol.mesh_sol);
    unsigned int fe_degree = discretization->get_fe_degree();
    return 0.5 / (max_transport_speed * (fe_degree + 1) * (fe_degree + 1));
}

template <int dim, typename SolutionVec>
double MaxwellFluxDGOperator<dim, SolutionVec>::compute_cell_transport_speed(
    const MatrixFree<dim, double> &mf,
    const LinearAlgebra::distributed::Vector<double> &solution) const {
    using VA = VectorizedArray<Number>;

    Number max_transport = 0;

    unsigned int first_component = first_component_index;

    FEEvaluation<dim, -1, 0, 8, Number> phi(mf, 0, 1, first_component);

    for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VA local_max = 0.;
        for (const unsigned int q : phi.quadrature_point_indices()) {
            const double max_speed =
                std::max(constants.c, std::max(constants.c * constants.chi,
                         constants.c * constants.gamma));

            const auto inverse_jacobian = phi.inverse_jacobian(q);
            Tensor<1, dim, VA> eigenvector;
            for (unsigned int d = 0; d < dim; ++d) eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5; ++i) {
                eigenvector = transpose(inverse_jacobian) *
                              (inverse_jacobian * eigenvector);
                VA eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                    eigenvector_norm =
                        std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
            }
            const auto jac_times_ev = inverse_jacobian * eigenvector;
            const auto max_eigenvalue = std::sqrt(
                (jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max = std::max(local_max, max_eigenvalue * max_speed);
        }

        for (unsigned int v = 0; v < mf.n_active_entries_per_cell_batch(cell);
             ++v) {
            for (unsigned int d = 0; d < 3; ++d)
                max_transport = std::max(max_transport, local_max[v]);
        }
    }
    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
}

}  // namespace warpii
