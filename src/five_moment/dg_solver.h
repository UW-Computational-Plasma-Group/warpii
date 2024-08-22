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
#include "explicit_operator.h"
#include "implicit_source_operator.h"
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
        std::shared_ptr<five_moment::Extension<dim>> extension,
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
          explicit_operator(extension, discretization, gas_gamma, species, 
                  fields, plasma_norm, fields_enabled),
          implicit_source_operator(plasma_norm, species, discretization, fields_enabled),
          n_boundaries(n_boundaries),
          fields_enabled(fields_enabled)
        {}

    void reinit();

    void project_initial_condition();

    void solve(TimestepCallback callback);

    FiveMomentDGSolutionHelper<dim>& get_solution_helper();

    FiveMSolutionVec& get_solution();

   private:
    double t_end;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    FiveMomentDGSolutionHelper<dim> solution_helper;
    std::vector<std::shared_ptr<Species<dim>>> species;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    FiveMSolutionVec solution;

    SSPRK2Integrator<double, FiveMSolutionVec, FiveMomentExplicitOperator<dim>> ssp_integrator;
    FiveMomentExplicitOperator<dim> explicit_operator;
    FiveMomentImplicitSourceOperator<dim> implicit_source_operator;
    unsigned int n_boundaries;
    bool fields_enabled;
};

}  // namespace five_moment
}  // namespace warpii
