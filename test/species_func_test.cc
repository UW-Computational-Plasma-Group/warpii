#include <gtest/gtest.h>
#include <deal.II/base/parameter_handler.h>
#include "five_moment/species_func.h"

using namespace dealii;
using namespace warpii::five_moment;

TEST(SpeciesFuncTest, PrimitiveVariableTest) {
    ParameterHandler prm;
    SpeciesFunc<2>::declare_parameters(prm);
    SimulationInput input = SimulationInput(prm, R"(
set VariablesType = Primitive
set components = 4.0; 0.2; 0.3; 0.1; 1.0
)");
    input.reparse(false);
    auto func = SpeciesFunc<2>::create_from_parameters(input, 5.0 / 3.0);
    Point<2> pt = Point<2>(0.0, 0.0);
    Tensor<1, 5, double> val = func->evaluate(pt, 0.0);
    EXPECT_NEAR(val[0], 4.0, 1e-12);
    EXPECT_NEAR(val[1], 0.8, 1e-12);
    EXPECT_NEAR(val[2], 1.2, 1e-12);
    EXPECT_NEAR(val[3], 0.4, 1e-12);
    EXPECT_NEAR(val[4], 1.78, 1e-12);
}

TEST(SpeciesFuncTest, EvaluateTVectorized) {
    ParameterHandler prm;
    SpeciesFunc<2>::declare_parameters(prm);
    SimulationInput input = SimulationInput(prm, R"(
set VariablesType = Conserved
set components = 2*x; 2*y - t; 0; 0; 0
)");
    input.reparse(false);
    auto func = SpeciesFunc<2>::create_from_parameters(input, 5.0 / 3.0);
    Point<2, VectorizedArray<double>> pt = Point<2, VectorizedArray<double>>(
            VectorizedArray(1.0),
            VectorizedArray(2.0));
    Tensor<1, 5, VectorizedArray<double>> val = func->evaluate(pt, 0.32);
    EXPECT_EQ(val[0], VectorizedArray(2.0));
    EXPECT_EQ(val[1], VectorizedArray(4.0 - 0.32));
}

TEST(SpeciesFuncTest, EvaluateTDoubleDim2) {
    ParameterHandler prm;
    SpeciesFunc<2>::declare_parameters(prm);
    SimulationInput input = SimulationInput(prm, R"(
set VariablesType = Conserved
set components = 2*x; 2*y - t; 0; 0; 0
)");
    input.reparse(false);
    auto func = SpeciesFunc<2>::create_from_parameters(input, 5.0 / 3.0);
    Point<2, double> pt = Point<2, double>(1.0, 2.0);
    Tensor<1, 5, double> val = func->evaluate(pt, 0.32);
    EXPECT_EQ(val[0], 2.0);
    EXPECT_EQ(val[1], 4.0 - 0.32);
}

TEST(SpeciesFuncTest, EvaluateTDoubleDim1) {
    ParameterHandler prm;
    SpeciesFunc<2>::declare_parameters(prm);
    SimulationInput input = SimulationInput(prm, R"(
set VariablesType = Conserved
set components = 2*x; 2 - t; 0; 0; 0
)");
    input.reparse(false);
    auto func = SpeciesFunc<1>::create_from_parameters(input, 5.0 / 3.0);
    Point<1, double> pt = Point<1, double>(1.0);
    Tensor<1, 5, double> val = func->evaluate(pt, 0.32);
    EXPECT_EQ(val[0], 2.0);
    EXPECT_EQ(val[1], 2.0 - 0.32);
}
