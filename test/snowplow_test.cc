#include <deal.II/base/function_parser.h>
#include <gtest/gtest.h>
#include "deal.II/base/parameter_handler.h"
#include <deal.II/numerics/vector_tools.h>
#include "src/five_moment/five_moment.h"
#include "five_moment_test_helpers.h"
#include "warpii.h"

using namespace dealii;
using namespace warpii;

TEST(SnowplowTest, CreatesSnowplowFromFluxInjection) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input = FiveMoment1DBuilder()
        .with_geometry(50, 10.0, false)
        .with_fe_degree(3)
        .until_time(1.0)
        .with_species(electrons(0.04)
                .with_ic_rho_u_p("0.04; 0; 0; 0; 1")
                .with_bc(inflow_bc("0.04; 2.7; 0; 0; 1")))
        .with_species(ions()
                .with_ic_rho_u_p("1.0; 0; 0; 0; 1")
                .with_bc(inflow_bc("1.0; 2.7; 0; 0; 1")))
        .with_omega_p_tau(10.0)
        .with_fields(MaxwellFieldsBuilder()
                .with_bc(maxwell_flux_injection_bc("0; 0; 0; 0; 0; 4; 0; 0")))
        .to_input();

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.run();

    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& soln = app.get_solution();
    //auto& helper = app.get_solution_helper();
    auto& disc = app.get_discretization();

    Utilities::MPI::RemotePointEvaluation<1> cache;
    std::vector<Point<1>> eval_points = {Point<1>(2.5), Point<1>(3.5)};
    auto point_vals = VectorTools::point_values<18>(disc.get_mapping(),
            disc.get_dof_handler(), soln.mesh_sol,
            eval_points, cache, VectorTools::EvaluationFlags::avg);

    // Check the region of greatest current
    auto first_pt = point_vals[0];
    std::cout << "First point: " << first_pt << std::endl;
    auto j_e_y = (-1.0 / 0.04) * first_pt[2];
    ASSERT_GT(j_e_y, 0.0);
    auto j_i_y = (1.0 / 1.0) * first_pt[7];
    ASSERT_GT(j_i_y, 0.0);
    // The ion and electron y momentum are nearly equal and opposite, differing by no more than 40% in
    // absolute value.
    //ASSERT_LT(std::abs(first_pt[2] + first_pt[7]), 0.4 * std::max(first_pt[2], first_pt[7]));

    auto Bz = first_pt[15];
    ASSERT_GT(Bz, 0.0);
    ASSERT_LT(Bz, 4.0);
    // Densities
    auto rho_e = first_pt[0];
    std::cout << "rho_e = " << rho_e << std::endl;
    auto rho_i = first_pt[5];
    ASSERT_GT(rho_e / 0.04, 1.8);
    ASSERT_GT(rho_i, 1.8);
    // X-momentum is positive at this point for both species
    auto rhou_e_x = first_pt[1];
    auto rhou_i_x = first_pt[6];
    ASSERT_GT(rhou_e_x / 0.04, 2.0);
    ASSERT_GT(rhou_i_x, 2.0);

    // Second point, inside the snowplow
    auto second_pt = point_vals[1];
    rho_e = second_pt[0];
    rho_i = second_pt[5];
    ASSERT_GT(rho_e / 0.04, 2.2);
    ASSERT_GT(rho_i, 2.2);
    rhou_e_x = second_pt[1];
    rhou_i_x = second_pt[6];
    ASSERT_GT(rhou_e_x / 0.04, 5.0);
    ASSERT_GT(rhou_i_x, 5.5);
}

