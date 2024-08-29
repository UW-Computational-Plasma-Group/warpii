#include <deal.II/matrix_free/fe_evaluation.h>
#include "five_moment/extension.h"
#include "five_moment/euler.h"
#include "warpii.h"
#include <gtest/gtest.h>
#include "src/five_moment/five_moment.h"

using namespace warpii;

class SingleSpeciesBCExtension : public five_moment::Extension<1> {
    void prepare_boundary_flux_evaluators(
            const unsigned int face,
            const unsigned int ,
            const LinearAlgebra::distributed::Vector<double> &src,
            std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 1> &fluid_evals) override {
        auto& electrons = fluid_evals[0];
        electrons.reinit(face);
        electrons.gather_evaluate(src, EvaluationFlags::values);
    }

    Tensor<1, 5, VectorizedArray<double>> boundary_flux(
            const types::boundary_id , const unsigned int q,
        const unsigned int,
        const std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 1> &fluid_evals) override {
        auto& electrons = fluid_evals[0];
        const auto q_in = electrons.get_value(q);

        Tensor<1, 5, VectorizedArray<double>> ghost_state;
        ghost_state[0] = VectorizedArray<double>(1e-6);
        ghost_state[1] = VectorizedArray<double>(0.0);
        ghost_state[2] = VectorizedArray<double>(0.0);
        ghost_state[3] = VectorizedArray<double>(0.0);
        ghost_state[4] = VectorizedArray<double>(1e-6 * 1.5);

        return five_moment::euler_numerical_flux<1, VectorizedArray<double>>(
                q_in, ghost_state, electrons.normal_vector(q), 5.0 / 3.0);
    }
};

TEST(BCExtensionTest, SingleSpeciesTest) {
    Warpii warpii_obj;
    std::string input_template = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.05

set write_output = false
set fe_degree = 2

set fields_enabled = false

set n_boundaries = 2
set n_species = 1

subsection geometry
    set left = 0.0
    set right = 100.0
    set nx = 100
    set periodic_dimensions =
end

subsection Species_0
    set name = electron
    set charge = -1.0
    set mass = 0.04
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 0.04; 0.0; 0.0; 0.0; 10.0
    end

    subsection BoundaryCondition_0
        set Type = Inflow
        subsection InflowFunction
            set VariablesType = Primitive
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
    subsection BoundaryCondition_1
        set Type = Inflow
        subsection InflowFunction
            set VariablesType = Primitive
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
end
    )";

    // Obtain reference solution
    std::stringstream reference_input;
    reference_input << input_template;
    warpii_obj.opts.fpe = true;
    warpii_obj.input = reference_input.str();
    warpii_obj.run();
    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& reference_soln = app.get_solution();

    // Obtain test solution
    std::stringstream input;
    input << input_template;
    input << R"(
subsection Species_0
    subsection BoundaryCondition_0
        set Type = Extension
    end
    subsection BoundaryCondition_1
        set Type = Extension
    end
end
    )";
    WarpiiOpts opts;
    std::shared_ptr<SingleSpeciesBCExtension> ext = std::make_shared<SingleSpeciesBCExtension>();
    Warpii warpii_obj2(opts, ext);
    warpii_obj2.opts.fpe = true;
    warpii_obj2.input = input.str();
    warpii_obj2.run();
    auto& actual_soln = warpii_obj.get_app<five_moment::FiveMomentApp<1>>().get_solution();

    LinearAlgebra::distributed::Vector<double> difference(actual_soln.mesh_sol);
    difference.add(-1.0, reference_soln.mesh_sol);
    EXPECT_LE(difference.l2_norm(), 1e-8);
}

class TwoSpeciesBCExtension : public five_moment::Extension<1> {
    void prepare_boundary_flux_evaluators(
            const unsigned int face,
            const unsigned int,
            const LinearAlgebra::distributed::Vector<double> &src,
            std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 2> &fluid_evals,
        FEFaceEvaluation<1, -1, 0, 3, double>&,
        FEFaceEvaluation<1, -1, 0, 3, double>&) override {
        auto& electrons = fluid_evals[0];
        electrons.reinit(face);
        electrons.gather_evaluate(src, EvaluationFlags::values);
    }

    Tensor<1, 5, VectorizedArray<double>> boundary_flux(
            const types::boundary_id , const unsigned int q,
        const unsigned int,
        const std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 2> &fluid_evals,
        const FEFaceEvaluation<1, -1, 0, 3, double>&,
        const FEFaceEvaluation<1, -1, 0, 3, double>&
        ) override {
        auto& electrons = fluid_evals[0];
        const auto q_in = electrons.get_value(q);

        Tensor<1, 5, VectorizedArray<double>> ghost_state;
        ghost_state[0] = VectorizedArray<double>(1e-6);
        ghost_state[1] = VectorizedArray<double>(0.0);
        ghost_state[2] = VectorizedArray<double>(0.0);
        ghost_state[3] = VectorizedArray<double>(0.0);
        ghost_state[4] = VectorizedArray<double>(1e-6 * 1.5);

        return five_moment::euler_numerical_flux<1, VectorizedArray<double>>(
                q_in, ghost_state, electrons.normal_vector(q), 5.0 / 3.0);
    }
};

TEST(BCExtensionTest, TwoSpeciesTest) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    Warpii warpii_obj;
    std::string input_template = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.05

set write_output = true
set n_writeout_frames = 100

set fe_degree = 2

set fields_enabled = true

set n_boundaries = 2
set n_species = 2

subsection geometry
    set left = 0.0
    set right = 100.0
    set nx = 100
    set periodic_dimensions = y
end

subsection PHMaxwellFields
    subsection InitialCondition
        set components = 0; 0.0; 0; \
                         0; 0; 0; \
                         0; 0
    end
    subsection BoundaryCondition_0
        set Type = PerfectConductor
    end
    subsection BoundaryCondition_1
        set Type = PerfectConductor
    end
end

subsection Species_0
    set name = electron
    set charge = -1.0
    set mass = 0.04
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 0.04; 0.0; 0.0; 0.0; 10.0
    end

    subsection BoundaryCondition_0
        set Type = Inflow
        subsection InflowFunction
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
    subsection BoundaryCondition_1
        set Type = Inflow
        subsection InflowFunction
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
end

subsection Species_1
    set name = ion
    set charge = 1.0
    set mass = 1.0
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 1.0; 0.0; 0.0; 0.0; 10.0
    end

    subsection BoundaryCondition_0
        set Type = Inflow
        subsection InflowFunction
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
    subsection BoundaryCondition_1
        set Type = Inflow
        subsection InflowFunction
            set components = 1e-6; 0; 0; 0; 1e-6
        end
    end
end
    )";

    // Obtain reference solution
    std::stringstream reference_input;
    reference_input << input_template;
    warpii_obj.opts.fpe = true;
    warpii_obj.input = reference_input.str();
    warpii_obj.run();
    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& reference_soln = app.get_solution();

    // Obtain test solution
    std::stringstream input;
    input << input_template;
    input << R"(
subsection Species_0
    subsection BoundaryCondition_0
        set Type = Extension
    end
    subsection BoundaryCondition_1
        set Type = Extension
    end
end
    )";
    WarpiiOpts opts;
    std::shared_ptr<TwoSpeciesBCExtension> ext = std::make_shared<TwoSpeciesBCExtension>();
    Warpii warpii_obj2(opts, ext);
    warpii_obj2.opts.fpe = true;
    warpii_obj2.input = input.str();
    warpii_obj2.run();
    auto& actual_soln = warpii_obj.get_app<five_moment::FiveMomentApp<1>>().get_solution();

    LinearAlgebra::distributed::Vector<double> difference(actual_soln.mesh_sol);
    difference.add(-1.0, reference_soln.mesh_sol);
    EXPECT_LE(difference.l2_norm(), 1e-8);
}
