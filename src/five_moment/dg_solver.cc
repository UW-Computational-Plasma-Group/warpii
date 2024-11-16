#include "dg_solver.h"
#include "solution_vec.h"
#include <iomanip>

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentDGSolver<dim>::reinit() {
    discretization->reinit();
    discretization->perform_allocation(solution.mesh_sol);
    for (unsigned int i = 0; i < species.size(); i++) {
        solution.boundary_integrated_fluxes.emplace_back();
        solution.boundary_integrated_fluxes.at(i).reinit(n_boundaries, dim);
    }
    solution.boundary_integrated_normal_poynting_vectors.reinit(n_boundaries);
    ssp_integrator->reinit(solution, 3);
}

template <int dim>
void FiveMomentDGSolver<dim>::project_initial_condition() {
    for (unsigned int i = 0; i < species.size(); i++) {
        solution_helper.project_fluid_quantities(
            *species.at(i)->initial_condition, solution.mesh_sol, i);
    }
    if (fields_enabled) {
        solution_helper.project_field_quantities(
                *fields->get_initial_condition().func,
                solution.mesh_sol);
    }
}

template <int dim>
void FiveMomentDGSolver<dim>::solve(TimestepCallback writeout_callback, 
        TimestepCallback diagnostic_callback) {
    auto step = [&](double t, double dt) -> bool {

        if (explicit_fluid_field_coupling) {
            extension->set_time(t);
            TimestepRequest request(dt, true);
            const TimestepResult result = ssp_integrator->evolve_one_time_step(
                    explicit_operator, solution, request, t);
            if (!result.successful) {
                return false;
            }
            std::cout << "t = " << t << std::endl;
            return true;
        } else if (splitting_scheme == "LieTrotter") {
            extension->set_time(t);
            TimestepRequest request(dt, true);
            const TimestepResult result = ssp_integrator->evolve_one_time_step(
                    explicit_operator, solution, request, t);
            if (!result.successful) {
                return false;
            }
            implicit_source_operator.evolve_one_time_step(solution.mesh_sol, result.achieved_dt);

            std::cout << "t = " << t << std::endl;
            return true;
        } else if (splitting_scheme == "Strang") {
            extension->set_time(t);
            TimestepRequest half_request(dt/2, true);
            TimestepResult result = ssp_integrator->evolve_one_time_step(
                    explicit_operator, solution, half_request, t);
            if (!result.successful) {
                return false;
            }
            implicit_source_operator.evolve_one_time_step(solution.mesh_sol, result.achieved_dt * 2.0);
            half_request = TimestepRequest(dt/2, false);
            result = ssp_integrator->evolve_one_time_step(explicit_operator, solution, half_request, t);
            if (!result.successful) {
                return false;
            }

            std::cout << "t = " << t << std::endl;
            return true;
        }
        AssertThrow(false, ExcMessage("Must have either explicit coupling or a valid splitting scheme"));
    };
    auto recommend_dt = [&]() -> double {
        return explicit_operator.recommend_dt(
            discretization->get_matrix_free(), solution);
    };

    std::vector<TimestepCallback> callbacks = {writeout_callback, diagnostic_callback};
    advance(step, t_end, recommend_dt, callbacks);
}

template<int dim>
void FiveMomentDGSolver<dim>::print_out_energy_inventory(const FiveMSolutionVec& soln) {
    const auto emag_energy = solution_helper.compute_global_electromagnetic_energy(soln.mesh_sol);
    std::cout << std::setprecision(12) << "E2 = " << emag_energy[0] << ", B2 = " << emag_energy[1] << std::endl;
    double total_energy = emag_energy[0] + emag_energy[1];
    for (unsigned int i = 0; i < species.size(); i++) {
        const auto integral = solution_helper.compute_global_integral(soln.mesh_sol, i);
        std::cout << species[i]->name << " energy = " << integral[4] << std::endl;
        total_energy += integral[4];
    }
    std::cout << "Total energy = " << total_energy << std::endl;
}

template <int dim>
FiveMSolutionVec& FiveMomentDGSolver<dim>::get_solution() {
    return solution;
}

template <int dim>
FiveMomentDGSolutionHelper<dim>& FiveMomentDGSolver<dim>::get_solution_helper() {
    return solution_helper;
}

template <int dim>
FiveMomentExplicitOperator<dim>& FiveMomentDGSolver<dim>::get_explicit_operator() {
    return explicit_operator;
}

template class FiveMomentDGSolver<1>;
template class FiveMomentDGSolver<2>;

}  // namespace five_moment
}  // namespace warpii
