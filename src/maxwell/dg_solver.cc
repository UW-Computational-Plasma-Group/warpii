#include "dg_solver.h"

namespace warpii {
namespace maxwell {

template <int dim>
void PHMaxwellDGSolver<dim>::reinit() {
    discretization->reinit();
    discretization->perform_allocation(solution.mesh_sol);
    ssp_integrator.reinit(solution, 2);
}

template <int dim>
void PHMaxwellDGSolver<dim>::project_initial_condition() {
    solution_helper.project_field_quantities(
            fields->get_initial_condition(),
            solution.mesh_sol);
}

template <int dim>
void PHMaxwellDGSolver<dim>::solve(TimestepCallback writeout_callback) {
    auto step = [&](double t, double dt) -> bool {
        ssp_integrator.evolve_one_time_step(flux_operator, solution, dt, t);
        std::cout << "t = " << t << std::endl;
        return true;
    };
    auto recommend_dt = [&]() -> double {
        return 0.001;
    };
    std::vector<TimestepCallback> callbacks = {writeout_callback};
    advance(step, t_end, recommend_dt, callbacks);
}

template class PHMaxwellDGSolver<1>;
template class PHMaxwellDGSolver<2>;

}  // namespace maxwell
}  // namespace warpii
