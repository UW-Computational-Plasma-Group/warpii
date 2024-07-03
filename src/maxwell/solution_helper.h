#pragma once
#include "../dgsem/nodal_dg_discretization.h"
#include "phmaxwell_func.h"

namespace warpii {
namespace maxwell {
/**
 * Contains solution processing routines which are specific
 * to the perfectly hyperbolic Maxwell's system of equations.
 */
template <int dim>
class PHMaxwellSolutionHelper {
   public:
    PHMaxwellSolutionHelper(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization)
        : discretization(discretization) {}

    void project_field_quantities(
        const PHMaxwellFunc<dim> &func,
        LinearAlgebra::distributed::Vector<double> &solution) const;

    /**
     * Compute the L^2 norm of the difference between `solution` and `f`,
     * integrated over the domain.
     */
    double compute_global_error(
        const LinearAlgebra::distributed::Vector<double>& solution, 
        Function<dim>& f,
        unsigned int component) const;

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
};

}  // namespace maxwell
}  // namespace warpii
