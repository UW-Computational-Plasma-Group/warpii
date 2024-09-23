#include "implicit_source_operator.h"

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "utilities.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentImplicitSourceOperator<dim>::reinit(
    const LinearAlgebra::distributed::Vector<double> &solution) {
    soln_register.reinit(solution);
}

template <int dim>
void FiveMomentImplicitSourceOperator<dim>::evolve_one_time_step(
    LinearAlgebra::distributed::Vector<double> &solution, const double dt) {
    this->dt = dt;

    if (fields_enabled) {
        discretization->mf.cell_loop(
            &FiveMomentImplicitSourceOperator<dim>::local_apply_cell, this, solution,
            solution);
    }
}

template <int dim>
void FiveMomentImplicitSourceOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int n_species = species.size();

    const double omega_p_tau = plasma_norm.omega_p_tau;
    const double omega_c_tau = plasma_norm.omega_c_tau;

    std::vector<FEEvaluation<dim, -1, 0, 5, double>> species_evals;
    for (unsigned int i = 0; i < n_species; i++) {
        species_evals.emplace_back(mf, 0, 0, 5 * i);
    }
    FEEvaluation<dim, -1, 0, 8, double> fields_eval(mf, 0, 0, 5 * n_species);
    AssertThrow(species_evals[0].dofs_per_component == fields_eval.dofs_per_component,
            ExcMessage("DOFs per component do not match"));

    LAPACKFullMatrix<double> M(3 * n_species + 3);
    LAPACKFullMatrix<double> L(3 * n_species + 3);
    L.reset_values();

    FullMatrix<double> IxB(3);

    FullMatrix<double> identity_3(3);
    for (unsigned int i = 0; i < 3; i++) {
        identity_3(i, i) = 1.0;
        IxB(i, i) = 0.0;
    }
    // Populate constant parts of the local matrix
    // Top row of the matrix.
    for (unsigned int i = 0; i < n_species; i++) {
        L.fill(identity_3, 0, 3 + 3 * i, 0, 0, -omega_p_tau * omega_p_tau / omega_c_tau);
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
            auto field_vals = fields_eval.get_dof_value(dof);

            Tensor<1, 3, VectorizedArray<double>> E_n_plus_1_2;
            std::vector<Tensor<1, 3, VectorizedArray<double>>> rhou_n_plus_1_2;
            for (unsigned int i = 0; i < n_species; i++) {
                rhou_n_plus_1_2.emplace_back();
            }

            for (unsigned int lane = 0; lane < VectorizedArray<double>::size();
                 lane++) {

                // Construct entries of the local matrix

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
                    auto rho_i = species_evals[i].get_dof_value(dof)[0][lane];
                    const auto n_i = rho_i / A_i;

                    L.fill(identity_3, 3 + 3 * i, 0, 0, 0, n_i * Z_i * Z_i / A_i * omega_c_tau);
                    L.fill(IxB, 3 + 3 * i, 3 + 3 * i, 0, 0,
                           omega_c_tau * Z_i / A_i);
                }

                M.reinit(3 * n_species + 3);
                for (unsigned int i = 0; i < 3 * n_species + 3; i++) {
                    M(i, i) += 1.0;
                }
                M.add(-dt/2.0, L);
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

                for (unsigned int d = 0; d < 3; d++) {
                    E_n_plus_1_2[d][lane] = LHS(d);
                    for (unsigned int i = 0; i < n_species; i++) {
                        const double Z_i = species[i]->charge;
                        const double A_i = species[i]->mass;
                        rhou_n_plus_1_2[i][d][lane] = A_i / Z_i * LHS(3 + 3 * i + d);
                    }
                }
            }

            field_vals = fields_eval.get_dof_value(dof);
            for (unsigned int d = 0; d < 3; d++) {
                field_vals[d] = 2.0 * E_n_plus_1_2[d] - 1.0*field_vals[d];
            }
            fields_eval.submit_dof_value(field_vals, dof);

            for (unsigned int i = 0; i < n_species; i++) {
                auto species_vals = species_evals[i].get_dof_value(dof);
                
                VectorizedArray<double> old_KE = VectorizedArray(0.0);
                VectorizedArray<double> new_KE = VectorizedArray(0.0);
                for (unsigned int d = 0; d < 3; d++) {
                    old_KE += 0.5 * species_vals[d+1] * species_vals[d+1] / species_vals[0];
                    species_vals[d+1] = 2.0 * rhou_n_plus_1_2[i][d] - 1.0*species_vals[d+1];
                    new_KE += 0.5 * species_vals[d+1] * species_vals[d+1] / species_vals[0];
                }
                VectorizedArray<double> internal_energy = species_vals[4] - old_KE;
                species_vals[4] = internal_energy + new_KE;

                species_evals[i].submit_dof_value(species_vals, dof);
            }
        }

        fields_eval.set_dof_values(dst);
        for (unsigned int i = 0; i < n_species; i++) {
            species_evals[i].set_dof_values(dst);
        }
    }
}

template class FiveMomentImplicitSourceOperator<1>;
template class FiveMomentImplicitSourceOperator<2>;

}  // namespace five_moment
}  // namespace warpii
