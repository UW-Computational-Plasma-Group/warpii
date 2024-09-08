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

#include "grid.h"
#include "../rk.h"
#include "../timestepper.h"
#include "solution_vec.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "explicit_operator.h"
#include "implicit_source_operator.h"
#include "five_moment/species.h"
#include "../maxwell/maxwell.h"
#include "../maxwell/fields.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
std::shared_ptr<SSPRKIntegrator<FiveMSolutionVec, FiveMomentExplicitOperator<dim>>>
create_integrator(const std::string& type) {
    if (type == "RK1") {
        return std::make_shared<RK1Integrator<FiveMSolutionVec, FiveMomentExplicitOperator<dim>>>();
    } else if (type == "SSPRK2") {
        return std::make_shared<SSPRK2Integrator<FiveMSolutionVec, FiveMomentExplicitOperator<dim>>>();
    } else {
        AssertThrow(false, ExcMessage("Unknown type of integrator: " + type));
    }
}

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
        bool fields_enabled,
        const std::string& integrator_type)
        : t_end(t_end),
          discretization(discretization),
          solution_helper(species.size(), discretization),
          species(species),
          fields(fields),
          explicit_operator(extension, discretization, gas_gamma, species, 
                  fields, plasma_norm, fields_enabled),
          implicit_source_operator(plasma_norm, species, discretization, fields_enabled),
          n_boundaries(n_boundaries),
          fields_enabled(fields_enabled),
          ssp_integrator(create_integrator<dim>(integrator_type))
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

    FiveMomentExplicitOperator<dim> explicit_operator;
    FiveMomentImplicitSourceOperator<dim> implicit_source_operator;
    unsigned int n_boundaries;
    bool fields_enabled;
    std::shared_ptr<SSPRKIntegrator<FiveMSolutionVec, FiveMomentExplicitOperator<dim>>> ssp_integrator;
};

}  // namespace five_moment
}  // namespace warpii
