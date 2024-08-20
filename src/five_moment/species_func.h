#pragma once
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace warpii {
namespace five_moment {

enum SpeciesFuncVariablesType {
    CONSERVED,
    PRIMITIVE,
};

template <int dim>
class SpeciesFunc : public Function<dim> {
   public:
    SpeciesFunc(std::unique_ptr<FunctionParser<dim>> func,
                SpeciesFuncVariablesType variables_type, double gas_gamma,
                bool is_zero)
        : Function<dim>(5),
          func(std::move(func)),
          variables_type(variables_type),
          gas_gamma(gas_gamma),
        is_zero(is_zero) {}

    double value(const Point<dim> &pt,
                 const unsigned int component) const override;

    static void declare_parameters(ParameterHandler &prm);

    static std::unique_ptr<SpeciesFunc<dim>> create_from_parameters(ParameterHandler &prm,
                                                   double gas_gamma);


   private:
    std::unique_ptr<FunctionParser<dim>> func;
    SpeciesFuncVariablesType variables_type;
    double gas_gamma;

   public:
        bool is_zero;
};
}  // namespace five_moment
}  // namespace warpii
