#pragma once
#include <deal.II/base/parsed_function.h>
#include "simulation_input.h"

using namespace dealii;

namespace warpii {

template <int dim>
class PHMaxwellFunc {
   public:
    PHMaxwellFunc(std::shared_ptr<FunctionParser<dim>> func, bool is_zero):
        func(func), is_zero(is_zero) {}

    static void declare_parameters(ParameterHandler &prm);
    static PHMaxwellFunc<dim> create_from_parameters(SimulationInput &input);

    std::shared_ptr<FunctionParser<dim>> func;
    bool is_zero;
};

}  // namespace warpii
