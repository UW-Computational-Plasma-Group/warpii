#pragma once

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "defs.h"
#include "dof_utils.h"
#include "timestep_request.h"
#include "timestep_result.h"

using namespace dealii;
// TODO put these in the warpii namespace!
using namespace warpii;

enum LowStorageRungeKuttaScheme {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
};

class LowStorageRungeKuttaIntegrator {
   public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme) {
        TimeStepping::runge_kutta_method lsrk;
        switch (scheme) {
            case stage_3_order_3: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
                break;
            }

            case stage_5_order_4: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
                break;
            }

            case stage_7_order_4: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
                break;
            }

            case stage_9_order_5: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
                break;
            }

            default:
                AssertThrow(false, ExcNotImplemented());
        }
        TimeStepping::LowStorageRungeKutta<
            LinearAlgebra::distributed::Vector<real>>
            rk_integrator(lsrk);
        rk_integrator.get_coefficients(ai, bi, ci);
    }

    unsigned int n_stages() const { return bi.size(); }

    template <typename VectorType, typename Operator>
    void perform_time_step(const Operator& pde_operator,
                           const double current_time, const double time_step,
                           VectorType& solution, VectorType& vec_ri,
                           VectorType& vec_ki) const {
        AssertDimension(ai.size() + 1, bi.size());

        pde_operator.perform_stage(current_time, bi[0] * time_step,
                                   ai[0] * time_step, solution, vec_ri,
                                   solution, vec_ri);

        for (unsigned int stage = 1; stage < bi.size(); ++stage) {
            const double c_i = ci[stage];
            pde_operator.perform_stage(
                current_time + c_i * time_step, bi[stage] * time_step,
                (stage == bi.size() - 1 ? 0 : ai[stage] * time_step), vec_ri,
                vec_ki, solution, vec_ri);
        }
    }

   private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
};

template <typename SolutionVec>
class ForwardEulerOperator {
    public:
        ~ForwardEulerOperator() = default;

        /**
         * Compute the single step
         *
         * ```
         * dst = b*dst + a*u + c*dt*f(u)),
         * ```
         *
         * where f is the RHS function provided by this operator.
         *
         * If `alpha` and `beta` take their default values, this reduces
         * to the simple Forward Euler step
         *
         * ```
         * dst = u + dt * f(u)
         * ```
         */
        virtual TimestepResult perform_forward_euler_step(
                SolutionVec &dst,
                const SolutionVec &u,
                std::vector<SolutionVec> &sol_registers,
                const TimestepRequest dt,
                const double t,
                const double b = 0.0,
                const double a = 1.0,
                const double c = 1.0) = 0;
};

/**
 * Interface for SSP-RK (Strong-stability-preserving Runge-Kutta) integrators.
 *
 * These integrators are based on the idea of a "Forward Euler operator",
 * by which we mean an operator that calculates
 *
 * ```
 * sol = sol + dt * F(sol).
 * ```
 */
template <typename SolutionVec, typename Operator>
class SSPRKIntegrator {
    public:
        virtual ~SSPRKIntegrator() = default;

    virtual TimestepResult evolve_one_time_step(Operator& forward_euler_operator,
                                      // Destination
                                      SolutionVec& solution,
                                      const TimestepRequest dt_request,
                                      const double t) = 0;

    virtual void reinit(const SolutionVec& sol, int sol_register_count) = 0;
};

/**
 * The Forward Euler or RK1 integrator.
 *
 * ```
 * sol = sol + dt * F(sol)
 * ```
 */
template <typename SolutionVec, typename Operator>
class RK1Integrator : public SSPRKIntegrator<SolutionVec, Operator> {
   public:
    RK1Integrator() {}

    TimestepResult evolve_one_time_step(Operator& forward_euler_operator,
                              // Destination
                              SolutionVec& solution,
                              const TimestepRequest dt_request,
                              const double t) override;

    void reinit(const SolutionVec& sol, int sol_register_count) override;

   private:
    SolutionVec soln_scratch;
    std::vector<SolutionVec> sol_registers;
};

template <typename SolutionVec, typename Operator>
TimestepResult RK1Integrator<SolutionVec, Operator>::evolve_one_time_step(
    Operator& forward_euler_operator,
    SolutionVec& solution,
    const TimestepRequest dt_request, 
    const double t) {
    // soln = soln + dt * f(soln)
    const auto result = forward_euler_operator.perform_forward_euler_step(
        soln_scratch, solution, sol_registers, dt_request, t);
    if (result.successful) {
        solution.swap(soln_scratch);
    }
    return result;
}

template <typename SolutionVec, typename Operator>
void RK1Integrator<SolutionVec, Operator>::reinit(
    const SolutionVec& sol,
    int sol_register_count) {
    soln_scratch.reinit(sol);
    for (int i = 0; i < sol_register_count; i++) {
        sol_registers.emplace_back();
        sol_registers[i].reinit(sol);
    }
}

/**
 * The 2-stage second-order SSP-RK integrator.
 *
 * ```
 * sol_1 = sol + dt * F(sol)
 * sol = 0.5 * sol + 0.5 * sol_1 + 0.5 * dt * F(sol_1)
 * ```
 */
template <typename SolutionVec, typename Operator>
class SSPRK2Integrator : public SSPRKIntegrator<SolutionVec, Operator> {
   public:
    SSPRK2Integrator() {}

    TimestepResult evolve_one_time_step(Operator& forward_euler_operator,
                              // Destination
                              SolutionVec& solution,
                              const TimestepRequest dt_request,
                              const double t) override;

    void reinit(const SolutionVec& sol, int sol_register_count) override;

   private:
    SolutionVec f_1;
    SolutionVec solution_scratch;
    std::vector<SolutionVec> sol_registers;
};

template <typename SolutionVec, typename Operator>
TimestepResult SSPRK2Integrator<SolutionVec, Operator>::evolve_one_time_step(
    Operator& forward_euler_operator,
    SolutionVec& solution,
    const TimestepRequest dt_request,
    const double t) {
    solution_scratch.sadd(0.0, 1.0, solution);

    const TimestepRequest request_1(dt_request.requested_dt, dt_request.is_flexible);

    // f_1 = soln + dt * f(soln)
    std::cout << "Taking stage 1" << std::endl;
    const auto result_1 = forward_euler_operator.perform_forward_euler_step(
        f_1, solution_scratch, sol_registers, request_1, t);
    if (!result_1.successful) {
        std::cout << "Stage 1 failed" << std::endl;
        return TimestepResult::failure(dt_request.requested_dt);
    }

    const TimestepRequest request_2(result_1.achieved_dt, false);

    // soln = 0.5*soln + 0.5*f_1 + 0.5*dt*f(f_1)
    std::cout << "Taking stage 2" << std::endl;
    const auto result_2 = forward_euler_operator.perform_forward_euler_step(
        solution_scratch, f_1, sol_registers, request_2, t + result_1.achieved_dt, 0.5, 0.5, 0.5);
    if (!result_2.successful) {
        std::cout << "Stage 2 failed" << std::endl;
        return TimestepResult::failure(dt_request.requested_dt);
    }

    solution_scratch.swap(solution);
    return TimestepResult(dt_request.requested_dt, true, result_1.achieved_dt);
}

template <typename SolutionVec, typename Operator>
void SSPRK2Integrator<SolutionVec, Operator>::reinit(
    const SolutionVec& sol,
    int sol_register_count) {
    f_1.reinit(sol);
    solution_scratch.reinit(sol);
    for (int i = 0; i < sol_register_count; i++) {
        sol_registers.emplace_back();
        sol_registers[i].reinit(sol);
    }
}

