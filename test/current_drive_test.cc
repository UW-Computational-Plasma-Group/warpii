#include <gtest/gtest.h>
#include "warpii.h"
#include "src/five_moment/five_moment.h"

using namespace warpii;
using namespace warpii::five_moment;

TEST(CurrentDriveTest, RampsUpToCurrent) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input = R"(
set Application = FiveMoment
set n_dims = 1
set fe_degree = 1

set ExplicitIntegrator = RK1

subsection geometry
    set left = 0
    set right = 1
    set nx = 3
end

set n_species = 2
set n_boundaries = 0

subsection Species_0
    set name = electron
    set mass = 0.04
    set charge = -1.0
    subsection InitialCondition
        set components = 4; 0; 0; 0; .01
    end
end

subsection Species_1
    set name = ion
    set mass = 1.0
    set charge = 1.0
    subsection InitialCondition
        set components = 100; 0; 0; 0; .01
    end
end

subsection Normalization
    set omega_p_tau = 100
end

subsection PHMaxwellFields
    set phmaxwell_chi = 0
    set phmaxwell_gamma = 0
    subsection GeneralSourceTerm
        set components = 100 * min(.78, .78 * t / 40); 0; 0; 0; 0; 0; 0; 0
    end
end

set t_end = 60.0
)";

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.run();

    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& soln = app.get_solution();
    auto& helper = app.get_solution_helper();

    FunctionParser<1> rho_u_expected = FunctionParser<1>(
    "0; 0.75 * 0.04 / -1; 0; 0; 0;"
    "0; 0.03 * 1 / 1; 0; 0; 0;"
    "0; 0; 0; 0; 0; 0; 0; 0"
    );
    
    double rho_u_e_error = helper.compute_global_error(soln.mesh_sol, 
            rho_u_expected, 1);
    double rho_u_i_error = helper.compute_global_error(soln.mesh_sol, 
            rho_u_expected, 6);

    EXPECT_NEAR(rho_u_e_error, 0.0, 1e-14);
    EXPECT_NEAR(rho_u_i_error, 0.0, 1e-14);
}
