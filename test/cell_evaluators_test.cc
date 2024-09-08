#include "five_moment/cell_evaluators.h"
#include "grid.h"
#include "warpii.h"
#include "src/five_moment/five_moment.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <gtest/gtest.h>

using namespace dealii;
using namespace warpii;
using namespace warpii::five_moment;

TEST(CellEvaluatorsTest, SimpleTest) {
    int argc = 0;
    char** argv = nullptr;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input = R"(
set n_dims = 2
set fields_enabled = true
set n_species = 2

set fe_degree = 3

subsection geometry
    set left = 0.0,0.0
    set right = 1.0,1.0
    set nx = 10,10
end

subsection Species_0
    set name = electron
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 1 + 0.6 * sin(2*pi*x); 1.0; 0.0; 0.0; 1.0
    end
end

subsection Species_1
    set name = ion
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 1 + 0.6 * sin(2*pi*x); 1.0; 0.0; 0.0; 1.0
    end
end
    )";

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();

    auto& app = warpii_obj.get_app<FiveMomentApp<2>>();
    
    FiveMomentCellEvaluators<2> evaluators(
            app.get_discretization().get_matrix_free(),
            app.get_solution().mesh_sol,
            2, true);

    evaluators.ensure_species_evaluated(0, 0, EvaluationFlags::gradients);
    ASSERT_NO_THROW(evaluators.species_eval(0).get_gradient(0));
    evaluators.ensure_species_evaluated(0, 0, EvaluationFlags::values);
    ASSERT_NO_THROW(evaluators.species_eval(0).get_value(0));

    evaluators.ensure_species_evaluated(1, 0, EvaluationFlags::values);
    ASSERT_NO_THROW(evaluators.species_eval(1).get_value(0));

    evaluators.ensure_fields_evaluated(0, EvaluationFlags::values);
    ASSERT_NO_THROW(evaluators.field_eval().get_value(0));

    evaluators.ensure_species_evaluated(1, 0, EvaluationFlags::values);
    ASSERT_NO_THROW(evaluators.species_eval(1).get_value(0));

    evaluators.ensure_fields_evaluated(1, EvaluationFlags::values);
    ASSERT_NO_THROW(evaluators.field_eval().get_value(0));
}
