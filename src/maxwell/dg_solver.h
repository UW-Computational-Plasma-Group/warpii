#pragma once
#include "../dgsem/nodal_dg_discretization.h"
#include "fields.h"
#include "solution_vec.h"
#include "maxwell_flux_dg_operator.h"
#include "../timestepper.h"
#include "solution_helper.h"

namespace warpii {

namespace maxwell {
template <int dim>
class PHMaxwellDGSolver {
   public:
    PHMaxwellDGSolver(
        double t_end,
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        std::shared_ptr<PHMaxwellFields<dim>> fields, unsigned int n_boundaries)
        : t_end(t_end),
          discretization(discretization),
          fields(fields),
          n_boundaries(n_boundaries),
          flux_operator(discretization, 0, fields),
          solution_helper(discretization)
    {}

    void reinit();
    
    void project_initial_condition();

    void solve(TimestepCallback callback);

    const MaxwellSolutionVec& get_solution() {
        return solution;
    }
    const PHMaxwellSolutionHelper<dim>& get_solution_helper() {
        return solution_helper;
    }

   private:
    double t_end;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    unsigned int n_boundaries;
    MaxwellSolutionVec solution;

    SSPRK2Integrator<double, MaxwellSolutionVec, MaxwellFluxDGOperator<dim, MaxwellSolutionVec>> ssp_integrator;
    MaxwellFluxDGOperator<dim, MaxwellSolutionVec> flux_operator;
    PHMaxwellSolutionHelper<dim> solution_helper;
};
}  // namespace maxwell
}  // namespace warpii
