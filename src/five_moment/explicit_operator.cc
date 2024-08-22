#include "explicit_operator.h"

#include "solution_vec.h"

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentExplicitOperator<dim>::perform_forward_euler_step(
    FiveMSolutionVec &dst, const FiveMSolutionVec &u,
    std::vector<FiveMSolutionVec> &sol_registers, const double dt,
    const double t, const double b, const double a, const double c) {
    // dst1 = b*dst + a*u + c*dt*f1(u)
    fluid_flux.perform_forward_euler_step(dst, u, sol_registers, dt, t, b, a, c);

    if (fields_enabled) {
        // dst2 = 1*dst1 + 0*u + c*dt*f2(u)
        //      = b*dst + a*u + c*dt*(f1(u) + f2(u))
        maxwell_flux.perform_forward_euler_step(
                dst, u, sol_registers, dt, t, 1.0, 0.0, c);
    }

    // dst = 1*dst2 + 0*u + c*dt*f3(u)
    //     = b*dst + a*u + c*dt*(f1(u) + f2(u) + f3(u))
    explicit_sources.perform_forward_euler_step(dst, u, sol_registers, dt, t, 1.0, 0.0, c);
}


template <int dim>
double FiveMomentExplicitOperator<dim>::recommend_dt(const MatrixFree<dim>& mf, const FiveMSolutionVec& soln) {
    return fluid_flux.recommend_dt(mf, soln);
}

template class FiveMomentExplicitOperator<1>;
template class FiveMomentExplicitOperator<2>;

}  // namespace five_moment
}  // namespace warpii
