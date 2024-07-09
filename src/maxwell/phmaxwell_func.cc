#include "phmaxwell_func.h"

#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/patterns.h>

using namespace dealii;

namespace warpii {

template <int dim>
void PHMaxwellFunc<dim>::declare_parameters(ParameterHandler& prm) {
    prm.declare_entry(
        "components",
        R"(0; 0; 0; \
0; 0; 0; \
0; 0)",
        Patterns::Anything(),
        R"(Expressions for the components of the perfectly-hyperbolic Maxwell system. These must be supplied in the following order:
```
Ex, Ey, Ez, Bx, By, Bz, phi, psi
```
The expressions may use the formula syntax from the [FunctionParser](https://www.dealii.org/current/doxygen/deal.II/classFunctionParser.html) class.

The spatial coordinates are defined as variables `x, y, z`, and time as the variable `t`.
)");

    prm.declare_entry(
        "constants", "", Patterns::Anything(),
        "Constants to use in the expression syntax. pi is defined by default.");
}

template <int dim>
PHMaxwellFunc<dim> PHMaxwellFunc<dim>::create_from_parameters(
    ParameterHandler& prm) {
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;

    std::string vnames;
    switch (dim) {
        case 1:
            vnames = "x,t";
            break;
        case 2:
            vnames = "x,y,t";
            break;
        case 3:
            vnames = "x,y,z,t";
            break;
        default:
            AssertThrow(false, ExcNotImplemented());
            break;
    }
    std::string expression = prm.get("components");
    std::string constants_list = prm.get("constants");

    std::vector<std::string> const_list =
        Utilities::split_string_list(constants_list, ',');
    for (const auto& constant : const_list) {
        std::vector<std::string> this_c =
            Utilities::split_string_list(constant, '=');
        AssertThrow(this_c.size() == 2,
                    ExcMessage("The list of constants, <" + constants_list +
                               ">, is not a comma-separated list of "
                               "entries of the form 'name=value'."));
        constants[this_c[0]] = Utilities::string_to_double(this_c[1]);
    }

    std::shared_ptr<FunctionParser<dim>> func = std::make_shared<FunctionParser<dim>>(8);
    func->initialize(vnames, expression, constants, true);

    return PHMaxwellFunc(func);
}

template class PHMaxwellFunc<1>;
template class PHMaxwellFunc<2>;

}  // namespace warpii
