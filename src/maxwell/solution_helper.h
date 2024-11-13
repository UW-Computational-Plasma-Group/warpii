#pragma once
#include "../dgsem/nodal_dg_discretization.h"
#include "phmaxwell_func.h"
#include "../normalization.h"

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
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        PlasmaNormalization plasma_norm)
        : discretization(discretization),
    plasma_norm(plasma_norm){}

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

    /**
     * Returns a tensor of the global integrals of |E|^2/(2c^2) and |B|^2/2,
     * which are the global electric and magnetic energy, respectively.
     */
    Tensor<1, 2, double> compute_global_electromagnetic_energy(
        const LinearAlgebra::distributed::Vector<double>& solution) const;


   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    PlasmaNormalization plasma_norm;
};

}  // namespace maxwell
}  // namespace warpii
