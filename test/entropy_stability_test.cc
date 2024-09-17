#include <deal.II/base/function_parser.h>
#include <gtest/gtest.h>
#include "deal.II/base/parameter_handler.h"
#include <deal.II/numerics/vector_tools.h>
#include "five_moment/euler.h"
#include "src/dgsem/nodal_dg_discretization.h"
#include "src/five_moment/five_moment.h"
#include "five_moment_test_helpers.h"
#include "src/five_moment/solution_vec.h"
#include "warpii.h"

using namespace dealii;
using namespace warpii;
using namespace warpii::five_moment;

void check_entropy_increment_negative(
        const FiveMSolutionVec& soln,
        const FiveMSolutionVec& dudt,
        NodalDGDiscretization<1>& disc) {

    FEEvaluation<1, -1, 0, 5, Number> phi_dudt(disc.get_matrix_free(), 0, 1, 0);
    FEEvaluation<1, -1, 0, 5, Number> phi_soln(disc.get_matrix_free(), 0, 1, 0);
    double d_eta_dt_total = 0.0;
    for (unsigned int cell = 0; cell < disc.get_matrix_free().n_cell_batches(); ++cell) {
        phi_dudt.reinit(cell);
        phi_dudt.gather_evaluate(dudt.mesh_sol, EvaluationFlags::values);
        phi_soln.reinit(cell);
        phi_soln.gather_evaluate(soln.mesh_sol, EvaluationFlags::values);
        for (const unsigned int q : phi_dudt.quadrature_point_indices()) {
            const auto Q = phi_soln.get_value(q);
            const auto dQdt = phi_dudt.get_value(q);
            const auto W = euler_entropy_variables<1>(Q, 5.0 / 3.0);
            const auto p = euler_pressure<1>(Q, 5.0 / 3.0);

            const auto d_eta_dt = dQdt * W;
            d_eta_dt_total += (phi_soln.JxW(q) * d_eta_dt).sum();
            for (unsigned int lane = 0; lane < VectorizedArray<double>::size(); lane++) {
                ASSERT_GE(p[lane], 0.0);
            }
            SHOW(d_eta_dt);
            SHOW(dQdt);
        }
    }
    ASSERT_LE(d_eta_dt_total, 1e-8);
}

std::vector<FiveMSolutionVec> solution_registers(const FiveMSolutionVec& soln) {
    FiveMSolutionVec reg1;
    FiveMSolutionVec reg2;
    FiveMSolutionVec reg3;
    std::vector<FiveMSolutionVec> registers = {reg1, reg2, reg3};
    registers[0].reinit(soln);
    registers[1].reinit(soln);
    registers[2].reinit(soln);

    return registers;
}

TEST(EntropyStabilityTest, PeriodicShockTube) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input = FiveMoment1DBuilder()
        .with_geometry(100, 10.0, true)
        .until_time(0.2)
        .with_species(neutrals()
                .with_ic_rho_u_p(
                    "if(abs(x - 5) < 1, 1, 1/8);"
                    "0; 0; 0;"
                    "if(abs(x - 5) < 1, 1, 0.1)"))
        .to_input();

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();

    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    const auto& soln = app.get_solution();
    auto& disc = app.get_discretization();
    auto& solver = app.get_solver();
    auto& explicit_op = solver.get_explicit_operator();

    FiveMSolutionVec dst;
    dst.reinit(soln);

    auto registers = solution_registers(soln);
    const double dt = 1e-4;
    const TimestepRequest request = TimestepRequest(dt, false);
    explicit_op.perform_forward_euler_step(dst, soln, registers, request, 0.0);

    FiveMSolutionVec dudt;
    dudt.reinit(soln);
    dudt.mesh_sol.sadd(0.0, dst.mesh_sol); // dudt = dst
    dudt.mesh_sol.add(-1.0, soln.mesh_sol); // dudt = dst - soln
    dudt.mesh_sol /= dt; // dudt = (dst - soln) / dt

    check_entropy_increment_negative(soln, dudt, disc);
}

TEST(EntropyStabilityTest, LargeVelocity) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input = FiveMoment1DBuilder()
        .with_fe_degree(1)
        .with_geometry(2, 1.0, true)
        .until_time(0.2)
        .with_species(neutrals()
                .with_ic_rho_u_p(
                    "1;"
                    "1000*sin(pi*x); 400; 0;"
                    "200"))
        .to_input();

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();

    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    const auto& soln = app.get_solution();
    auto& disc = app.get_discretization();
    auto& solver = app.get_solver();
    auto& explicit_op = solver.get_explicit_operator();

    FiveMSolutionVec dst;
    dst.reinit(soln);

    auto registers = solution_registers(soln);
    const double dt = 1e-4;
    const TimestepRequest request = TimestepRequest(dt, false);
    explicit_op.perform_forward_euler_step(dst, soln, registers, request, 0.0);

    FiveMSolutionVec dudt;
    dudt.reinit(soln);
    dudt.mesh_sol.sadd(0.0, dst.mesh_sol); // dudt = dst
    dudt.mesh_sol.add(-1.0, soln.mesh_sol); // dudt = dst - soln
    dudt.mesh_sol /= dt; // dudt = (dst - soln) / dt

    check_entropy_increment_negative(soln, dudt, disc);
}

