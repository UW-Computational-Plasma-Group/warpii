#include <gtest/gtest.h>
#include "warpii.h"
#include "src/maxwell/maxwell_app.h"

using namespace dealii;
using namespace warpii;

TEST(ElectromagneticEnergyTest, GlobalIntegralsTest) {
    std::string input = R"(
set Application = PerfectlyHyperbolicMaxwell
set n_dims = 2
set t_end = 0.01
set write_output = false

set fe_degree = 4

subsection geometry 
set left = 0.0,0.0
set right = 1.0,1.0
set nx = 10,10
end

subsection Normalization
set omega_p_tau = 4200
set omega_c_tau = 13
end

subsection PHMaxwellFields
subsection InitialCondition
    set components = 4.5*sin(5.6*x); 4.5*sin(5.6*x); 4.5*sin(5.6*x); \
                     3.4*sin(4.9*y); 3.4*sin(4.9*y); 3.4*sin(4.9*y); \
                     0.0; 0.0
end
end)";

    Warpii warpii_obj;
    warpii_obj.input = input;

    warpii_obj.opts.fpe = true;

    warpii_obj.setup();
    auto& app = warpii_obj.get_app<maxwell::PHMaxwellApp<2>>();
    const auto& soln = app.get_solution();
    auto& helper = app.get_solution_helper();

    auto emag_energy = helper.compute_global_electromagnetic_energy(soln.mesh_sol);

    const double c = 4200.0 / 13.0;
    double a = 4.5;
    double k = 5.6;
    const double E2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));

    EXPECT_NEAR(emag_energy[0] * c * c, 3*E2_integral / 2, 1e-8);

    a = 3.4;
    k = 4.9;
    const double B2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));

    EXPECT_NEAR(emag_energy[1], 3*B2_integral / 2, 1e-8);
}

TEST(ElectromagneticEnergyTest, ConservationTest) {
    std::string input = R"(
set Application = PerfectlyHyperbolicMaxwell
set n_dims = 2
set t_end = 0.015
set write_output = false

set fe_degree = 4

subsection geometry 
set left = 0.0,0.0
set right = 1.0,1.0
set nx = 40,40
end

subsection Normalization
set omega_p_tau = 42
set omega_c_tau = 13
end

subsection PHMaxwellFields
    set phmaxwell_chi = 0.0
    set phmaxwell_gamma = 0.0
subsection InitialCondition
    set components = 4.5*sin(6*pi*x); 0.0; 0.0; \
                     0.0; 0.0; 3.4*sin(2*pi*y); \
                     0.0; 0.0
end
end)";

    Warpii warpii_obj;
    warpii_obj.input = input;

    warpii_obj.opts.fpe = true;

    warpii_obj.setup();
    auto& app = warpii_obj.get_app<maxwell::PHMaxwellApp<2>>();
    auto& soln = app.get_solution();
    auto& helper = app.get_solution_helper();

    auto emag_energy = helper.compute_global_electromagnetic_energy(soln.mesh_sol);

    const double pi = 3.1415926535;
    const double c = 42.0 / 13.0;
    double a = 4.5;
    double k = 6.0*pi;
    double E2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));

    EXPECT_NEAR(emag_energy[0], E2_integral / (2*c*c), 1e-8);

    a = 3.4;
    k = 2.0*pi;
    double B2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));

    EXPECT_NEAR(emag_energy[1], B2_integral / 2, 1e-8);
    const double total_energy = emag_energy[0] + emag_energy[1];

    warpii_obj.run();

    emag_energy = helper.compute_global_electromagnetic_energy(soln.mesh_sol);

    a = 4.5;
    k = 6.0*pi;
    E2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));
    a = 3.4;
    k = 2.0*pi;
    B2_integral = a*a/2 * (1.0 - 0.0) - a*a/(4*k) * (std::sin(2*k*1.0) - std::sin(2*k*0.0));

    EXPECT_NEAR(emag_energy[0] + emag_energy[1], total_energy, 1e-8);
}
