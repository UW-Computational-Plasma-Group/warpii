#include "explicit_operator.h"

#include "solution_vec.h"

namespace warpii {
namespace five_moment {

template <int dim>
TimestepResult FiveMomentExplicitOperator<dim>::perform_forward_euler_step(
    FiveMSolutionVec &dst, const FiveMSolutionVec &u,
    std::vector<FiveMSolutionVec> &sol_registers, 
    const TimestepRequest dt_request,
    const double t, 
    const double b, const double a, const double c) {
    //AssertThrow(&dst != &u, ExcMessage("dst and u must not alias each other."));

    // dst1 = b*dst + a*u + c*dt*f1(u)
    const auto fluid_flux_result = fluid_flux.perform_forward_euler_step(
            dst, u, sol_registers, dt_request, t, b, a, c);
    if (!fluid_flux_result.successful) {
        return TimestepResult::failure(dt_request.requested_dt);
    }

    const auto inflexible_request = TimestepRequest(fluid_flux_result.achieved_dt, false);
    if (fields_enabled) {
        // dst2 = 1*dst1 + 0*u + c*dt*f2(u)
        //      = b*dst + a*u + c*dt*(f1(u) + f2(u))
        const auto result_2 = maxwell_flux.perform_forward_euler_step(
                dst, u, sol_registers, inflexible_request, t, 1.0, 0.0, c);
        if (!result_2.successful) {
            return TimestepResult::failure(dt_request.requested_dt);
        }
    }

    // dst = 1*dst2 + 0*u + c*dt*f3(u)
    //     = b*dst + a*u + c*dt*(f1(u) + f2(u) + f3(u))
    const auto result_3 = explicit_sources.perform_forward_euler_step(dst, u, sol_registers, 
            inflexible_request, t, 1.0, 0.0, c);
    if (!result_3.successful) {
        return TimestepResult::failure(dt_request.requested_dt);
    }

    fluid_flux.apply_positivity_limiter(dst);

    return TimestepResult(dt_request.requested_dt, true, fluid_flux_result.achieved_dt);
}


template <int dim>
double FiveMomentExplicitOperator<dim>::recommend_dt(const MatrixFree<dim>& mf, const FiveMSolutionVec& soln) {
    auto dt = fluid_flux.recommend_dt(mf, soln);
    if (fields_enabled) {
        dt = std::min(dt, maxwell_flux.recommend_dt(mf, soln));
    }
    return dt;
}

template class FiveMomentExplicitOperator<1>;
template class FiveMomentExplicitOperator<2>;

}  // namespace five_moment
}  // namespace warpii
