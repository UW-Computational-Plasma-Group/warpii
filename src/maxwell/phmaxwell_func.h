#pragma once
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace warpii {

template <int dim>
class PHMaxwellFunc {
   public:
    PHMaxwellFunc(std::shared_ptr<FunctionParser<dim>> func):
        func(func) {}

    static void declare_parameters(ParameterHandler &prm);
    static PHMaxwellFunc<dim> create_from_parameters(ParameterHandler &prm);

    std::shared_ptr<FunctionParser<dim>> func;
};

}  // namespace warpii
