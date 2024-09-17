#include "src/rk.h"
#include "src/timestep_result.h"
#include <gtest/gtest.h>
#include <deal.II/lac/vector.h>
#include <vector>
#include "src/timestepper.h"

using namespace dealii;

// Computes du/dt = lambda*u, whose solution is u = e^{lambda * t} * u0
class ExponentialGrowthOperator : public ForwardEulerOperator<Vector<double>> {
    public:
    ExponentialGrowthOperator(double lambda): lambda(lambda) {}

    TimestepResult perform_forward_euler_step(
            Vector<double> &dst,
            const Vector<double> &u,
            std::vector<Vector<double>>& ,
            const TimestepRequest dt_request,
            const double ,
            const double b=0.0,
            const double a=1.0,
            const double c=1.0) override {
        std::cout << "dst = " << dst(0) << std::endl;
        std::cout << "sol = " << u(0) << std::endl;
        dst(0) = b*dst(0) + a*u(0) + c*dt_request.requested_dt * lambda*u(0);
        return TimestepResult::success(dt_request);
    }

    private:
        double lambda;
};

TEST(RKTest, RK1Test) {
    RK1Integrator<Vector<double>, ExponentialGrowthOperator> integrator;
    double lambda = 3.4;
    ExponentialGrowthOperator op(lambda);

    double u0 = .12;
    Vector<double> solution(1);

    integrator.reinit(solution, 0);

    auto step = [&](double t, double dt) -> bool {
        TimestepRequest request(dt, false);
        integrator.evolve_one_time_step(op, solution, request, t);
        return true;
    };

    std::vector<double> dts = {0.01, 0.01/2, 0.01/4, 0.01/8, 0.01/16};
    std::vector<double> errors;
    for (auto dt : dts) {
        solution(0) = u0;
        auto recommend_dt = [&]() -> double {
            return dt;
        };

        std::vector<TimestepCallback> callbacks;
        advance(step, 3.0, recommend_dt, callbacks);

        double error = std::abs(std::exp(3.0*lambda) * u0 - solution(0));
        errors.push_back(error);
    }

    ASSERT_LT(errors[1], errors[0] / 1.8);
    ASSERT_LT(errors[2], errors[1] / 1.8);
    ASSERT_LT(errors[3], errors[2] / 1.8);
    ASSERT_LT(errors[4], errors[3] / 1.8);
}

TEST(RKTest, SSPRK2Test) {
    SSPRK2Integrator<Vector<double>, ExponentialGrowthOperator> integrator;
    double lambda = 3.4;
    ExponentialGrowthOperator op(lambda);

    double u0 = .12;
    Vector<double> solution(1);

    integrator.reinit(solution, 0);

    auto step = [&](double t, double dt) -> bool {
        TimestepRequest request(dt, false);
        integrator.evolve_one_time_step(op, solution, request, t);
        std::cout << "Done with step." << std::endl;
        return true;
    };

    std::vector<double> dts = {0.01, 0.01/2, 0.01/4, 0.01/8, 0.01/16};
    std::vector<double> errors;
    for (auto dt : dts) {
        solution(0) = u0;
        auto recommend_dt = [&]() -> double {
            return dt;
        };

        std::vector<TimestepCallback> callbacks;
        advance(step, 0.3, recommend_dt, callbacks);

        double error = std::abs(std::exp(0.3*lambda) * u0 - solution(0));
        errors.push_back(error);
        std::cout << "error = " << error;
    }

    ASSERT_LT(errors[1], errors[0] / 3.6);
    ASSERT_LT(errors[2], errors[1] / 3.6);
    ASSERT_LT(errors[3], errors[2] / 3.6);
    ASSERT_LT(errors[4], errors[3] / 3.6);
}
