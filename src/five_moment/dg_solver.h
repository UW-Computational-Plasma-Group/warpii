#pragma once

#include <boost/math/policies/policy.hpp>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <functional>

#include "../grid.h"
#include "../rk.h"
#include "../timestepper.h"
#include "solution_vec.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "fluid_flux_dg_operator.h"
#include "fluid_flux_es_dgsem_operator.h"
#include "flux_operator.h"
#include "source_operator.h"
#include "species.h"
#include "../maxwell/maxwell.h"
#include "../maxwell/fields.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

/**
 * The DGSolver for the Five-Moment application.
 *
 * This class's job is to perform the whole time integration of the five-moment
 * system of equations, using the supplied DGOperator to calculate right-hand
 * sides.
 */
template <int dim>
class FiveMomentDGSolver {
   public:
    FiveMomentDGSolver(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species, 
        std::shared_ptr<PHMaxwellFields<dim>> fields,
        PlasmaNormalization plasma_norm,
        double gas_gamma,
        double t_end,
        unsigned int n_boundaries,
        bool fields_enabled)
        : t_end(t_end),
          discretization(discretization),
          solution_helper(species.size(), discretization),
          species(species),
          fields(fields),
          fluid_flux_operator(discretization, gas_gamma, species),
          flux_operator(discretization, gas_gamma, species, 
                  fields, fields_enabled),
          source_operator(plasma_norm, species, discretization, fields_enabled),
          n_boundaries(n_boundaries),
          fields_enabled(fields_enabled)
        {}

    void reinit();

    void project_initial_condition();

    void solve(TimestepCallback callback);

    FiveMomentDGSolutionHelper<dim>& get_solution_helper();

    FiveMSolutionVec& get_solution();

    FluidFluxESDGSEMOperator<dim>& get_fluid_flux_operator() {
        return fluid_flux_operator;
    }

   private:
    double t_end;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    FiveMomentDGSolutionHelper<dim> solution_helper;
    std::vector<std::shared_ptr<Species<dim>>> species;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    FiveMSolutionVec solution;

    SSPRK2Integrator<double, FiveMSolutionVec, FiveMomentFluxOperator<dim>> ssp_integrator;
    FluidFluxESDGSEMOperator<dim> fluid_flux_operator;
    FiveMomentFluxOperator<dim> flux_operator;
    FiveMomentSourceOperator<dim> source_operator;
    unsigned int n_boundaries;
    bool fields_enabled;
};

}  // namespace five_moment
}  // namespace warpii
