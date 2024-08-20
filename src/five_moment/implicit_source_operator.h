#pragma once

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "species.h"
#include "../normalization.h"
#include "../dgsem/nodal_dg_discretization.h"

using namespace dealii;
namespace warpii {
namespace five_moment {

/**
 * Performs an implicit midpoint solve of the Maxwell-fluid source terms.
 */
template <int dim>
class FiveMomentImplicitSourceOperator {
    public:
    FiveMomentImplicitSourceOperator(
            PlasmaNormalization plasma_norm,
            std::vector<std::shared_ptr<Species<dim>>> species,
            std::shared_ptr<NodalDGDiscretization<dim>> discretization,
            bool fields_enabled):
        plasma_norm(plasma_norm),
        dt(0.0),
        species(species),
        discretization(discretization),
        fields_enabled(fields_enabled)
    {}

    void reinit(
           const LinearAlgebra::distributed::Vector<double>& solution);

   void evolve_one_time_step(
           LinearAlgebra::distributed::Vector<double>& solution,
           const double dt);

   void local_apply_cell(
           const MatrixFree<dim, double> &mf,
           LinearAlgebra::distributed::Vector<double> &dst,
           const LinearAlgebra::distributed::Vector<double> &src,
           const std::pair<unsigned int, unsigned int> &cell_range) const;

   private:
     PlasmaNormalization plasma_norm;
     double dt;
     LinearAlgebra::distributed::Vector<double> soln_register;
     std::vector<std::shared_ptr<Species<dim>>> species;
     std::shared_ptr<NodalDGDiscretization<dim>> discretization;
     bool fields_enabled;
};

}  // namespace five_moment
}  // namespace warpii
