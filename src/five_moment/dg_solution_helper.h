#pragma once

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

#include "function_eval.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "grid.h"
#include "../normalization.h"

namespace warpii {
namespace five_moment {
template <int dim>
class FiveMomentDGSolutionHelper {
   public:
    FiveMomentDGSolutionHelper(
            unsigned int n_species,
            std::shared_ptr<NodalDGDiscretization<dim>> discretization,
            PlasmaNormalization plasma_norm):
        n_species(n_species),
        discretization(discretization),
        plasma_norm(plasma_norm)
    {}

    void project_fluid_quantities(
        const Function<dim> &function,
        LinearAlgebra::distributed::Vector<double> &solution,
        unsigned int species_index) const;

    void project_field_quantities(
        const Function<dim> &function,
        LinearAlgebra::distributed::Vector<double> &solution) const;

    /**
     * Compute the L^2 norm of the difference between `solution` and `f`,
     * integrated over the domain.
     */
    double compute_global_error(
        LinearAlgebra::distributed::Vector<double>& solution, 
        Function<dim>& f,
        unsigned int component);

    /**
     * Computes the global integral of the solution vector for the given species.
     */
    Tensor<1, 5, double> compute_global_integral(
        LinearAlgebra::distributed::Vector<double>& solution,
        unsigned int species_index);

    /**
     * Returns a tensor of the global integrals of |E|^2/(2c^2) and |B|^2/2,
     * which are the global electric and magnetic energy, respectively.
     */
    Tensor<1, 2, double> compute_global_electromagnetic_energy(
        LinearAlgebra::distributed::Vector<double>& solution);

    double compute_global_electrostatic_energy(
        LinearAlgebra::distributed::Vector<double>& solution);

   private:
    unsigned int n_species;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    PlasmaNormalization plasma_norm;
};

}  // namespace five_moment
}  // namespace warpii
