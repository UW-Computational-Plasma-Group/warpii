#include "src/warpii.h"
#include <gtest/gtest.h>
#include "src/maxwell/maxwell_app.h"

using namespace dealii;
using namespace warpii;

TEST(PHMaxwellTest, LightPropPeriodic1D) {
    std::string input_template = R"(
set Application = PerfectlyHyperbolicMaxwell
set n_dims = 1
set t_end = 0.01
set write_output = false

set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 1.0
end

subsection Normalization
    set omega_c_tau = 2.0
end

subsection PHMaxwellFields
    subsection InitialCondition
        subsection E field
            set Function constants = pi=3.1415926535
            set Function expression = 0.0; sin(2*pi*x); 0.0
        end
        subsection B field
            set Function constants = pi=3.1415926535
            set Function expression = 0.0; 0.0; 2*sin(2*pi*x)
        end
    end
end
    )";

    FunctionParser<1> expected = FunctionParser<1>(
            "0; sin(2*pi*(x-c*T)); 0; 0; 0; 2*sin(2*pi*(x-c*T)); 0; 0", "c=0.5, pi=3.1415926535, T=0.01");

    std::vector<unsigned int> Nxs = { 20, 30 };
    std::vector<double> errors;
    for (unsigned int i = 0; i < Nxs.size(); i++) {
        Warpii warpii_obj;
        std::stringstream input;
        input << input_template;
        input << "subsection geometry\n set nx = " << Nxs[i] << "\n end";
        warpii_obj.opts.fpe = true;
        warpii_obj.input = input.str();
        warpii_obj.run();
        auto& app = warpii_obj.get_app<maxwell::PHMaxwellApp<1>>();
        const auto& soln = app.get_solution();
        const auto& helper = app.get_solution_helper();

        double Ey_error = helper.compute_global_error(soln.mesh_sol, expected, 1);
        std::cout << "error = " << Ey_error << std::endl;
        errors.push_back(Ey_error);
    }
    EXPECT_NEAR(errors[1], 0.0, 1e-4);
    EXPECT_NEAR(errors[0] / errors[1], pow(30.0/20.0, 3), 1.0);
}
