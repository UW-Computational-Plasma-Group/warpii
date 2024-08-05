#include "source_operator.h"

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentSourceOperator<dim>::reinit(
    const LinearAlgebra::distributed::Vector<double> &solution) {
    soln_register.reinit(solution);
}

template <int dim>
void FiveMomentSourceOperator<dim>::evolve_one_time_step(
    LinearAlgebra::distributed::Vector<double> &solution,
    const double dt) {
    this->dt = dt;

    if (fields_enabled) {
        discretization->mf.cell_loop(
                &FiveMomentSourceOperator<dim>::local_apply_cell,
                this,
                solution,
                solution);
    }
}


template <int dim>
void FiveMomentSourceOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int n_species = species.size();
    const double omega_c_tau = plasma_norm.omega_c_tau;

    std::vector<FEEvaluation<dim, -1, 0, 5, double>> species_evals;
    for (unsigned int i = 0; i < n_species; i++) {
        species_evals.emplace_back(mf, 0, 0, 5 * i);
    }
    FEEvaluation<dim, -1, 0, 8, double> fields_eval(mf, 0, 0, 5 * n_species);

    LAPACKFullMatrix<double> M(3 * n_species + 3);
    M.reset_values();

    FullMatrix<double> IxB(3);

    FullMatrix<double> omega_p_tau_scaling(3);
    for (unsigned int i = 0; i < 3; i++) {
        omega_p_tau_scaling(i, i) = plasma_norm.omega_p_tau;
        IxB(i, i) = 0.0;
    }
    // Populate constant parts of the local matrix
    for (unsigned int i = 0; i < n_species; i++) {
        M.fill(omega_p_tau_scaling, 0, 3 + 3 * i, 0, 0, -dt / 2.0 * -1.0);
        const double Z_i = species[i]->charge;
        const double A_i = species[i]->mass;
        M.fill(omega_p_tau_scaling, 3 + 3 * i, 0, 0, 0,
               -dt / 2.0 * Z_i * Z_i / A_i);
    }

    Vector<double> RHS(3 * n_species + 3);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        // Evaluate values at DOFs
        fields_eval.reinit(cell);
        fields_eval.read_dof_values(src);
        for (unsigned int i = 0; i < n_species; i++) {
            species_evals[i].reinit(cell);
            species_evals[i].read_dof_values(src);
        }

        for (unsigned int dof = 0; dof < fields_eval.dofs_per_component;
             dof++) {
            Tensor<1, 8, VectorizedArray<double>> fields_soln =
                fields_eval.get_dof_value(dof);
            std::vector<Tensor<1, 5, VectorizedArray<double>>> fluids_soln;
            for (unsigned int i = 0; i < n_species; i++) {
                fluids_soln.push_back(species_evals[i].get_dof_value(dof));
            }

            for (unsigned int lane = 0; lane < VectorizedArray<double>::size();
                 lane++) {
                // Construct spatially varying entries of matrix
                const auto field_vals = fields_eval.get_dof_value(dof);
                const double B_x = field_vals[3][lane];
                const double B_y = field_vals[4][lane];
                const double B_z = field_vals[5][lane];

                IxB(0, 1) = B_z;
                IxB(0, 2) = -B_y;
                IxB(1, 0) = -B_z;
                IxB(1, 2) = B_x;
                IxB(2, 0) = B_y;
                IxB(2, 2) = -B_x;
                for (unsigned int i = 0; i < n_species; i++) {
                    const double Z_i = species[i]->charge;
                    const double A_i = species[i]->mass;
                    M.fill(IxB, 3 + 3 * i, 3 + 3 * i, 0, 0,
                           -dt / 2.0 * omega_c_tau * Z_i * Z_i / A_i);
                }

                for (unsigned int i = 0; i < 3 * n_species + 3; i++) {
                    M(i, i) += 1.0;
                }
                M.compute_lu_factorization();

                // Form Vector that contains the RHS
                for (unsigned int d = 0; d < 3; d++) {
                    // Populate E
                    RHS[d] = field_vals[d][lane];

                    for (unsigned int i = 0; i < n_species; i++) {
                        const double Z_i = species[i]->charge;
                        const double A_i = species[i]->mass;
                        const auto species_val =
                            species_evals[i].get_dof_value(dof);
                        const double j_i_d =
                            Z_i / A_i * species_val[d + 1][lane];
                        RHS(3 + 3 * i + d) = j_i_d;
                    }
                }

                M.solve(RHS);
                auto &LHS = RHS;

                // Write solution to dst
                for (unsigned int d = 0; d < 3; d++) {
                    fields_soln[d][lane] = LHS(d);
                    for (unsigned int i = 0; i < n_species; i++) {
                        fluids_soln[i][d + 1][lane] = LHS(3 + 3 * i + d);
                    }
                }
            }
            fields_eval.submit_dof_value(fields_soln, dof);
            for (unsigned int i = 0; i < n_species; i++) {
                species_evals[i].submit_dof_value(fluids_soln[i], dof);
            }
        }

        fields_eval.set_dof_values(dst);
        for (unsigned int i = 0; i < n_species; i++) {
            species_evals[i].set_dof_values(dst);
        }
    }
}

template class FiveMomentSourceOperator<1>;
template class FiveMomentSourceOperator<2>;

}  // namespace five_moment
}  // namespace warpii
