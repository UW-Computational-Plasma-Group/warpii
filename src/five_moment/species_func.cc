#include "five_moment/species_func.h"
#include <deal.II/base/parameter_handler.h>
#include <variant>

namespace warpii {
namespace five_moment {

const std::string SPECIES_FUNC_DEFAULT = R"(0; 0; 0; 0; 0)";

template <int dim>
double SpeciesFunc<dim>::value(const Point<dim> &pt,
                               const unsigned int component) const {
    if (variables_type == CONSERVED) {
        return func->value(pt, component);
    } else {
        double rho = func->value(pt, 0);
        if (component == 0) {
            return rho;
        } else if (component <= 3) {
            return rho * func->value(pt, component);
        } else {
            double kinetic_energy = 0.0;
            for (unsigned int d = 0; d < 3; d++) {
                double u_d = func->value(pt, d + 1);
                kinetic_energy += 0.5 * rho * u_d * u_d;
            }
            double p = func->value(pt, 4);
            return kinetic_energy + p / (gas_gamma - 1);
        }
    }
}

template <int dim>
void SpeciesFunc<dim>::declare_parameters(ParameterHandler& prm) {
    prm.declare_entry("VariablesType", "Primitive", Patterns::Selection("Primitive|Conserved"),
            "Indicates how the `components` expressions should be interpreted, "
            "as primitive or conserved variables.");

    prm.declare_entry(
        "components",
        SPECIES_FUNC_DEFAULT,
        Patterns::Anything(),
        R"(Expressions for the moments of a five-moment species. 
If `VariablesType == Primitive`, the components will be interpreted as
```
rho, ux, uy, uz, p
```
where rho is the mass density, u is the velocity, and p the scalar pressure.

If `VariablesType == Conserved`, the components will be interpreted as
```
rho, rho*ux, rho*uy, rho*uz, e
```
where e is the scalar total energy, related to the pressure by
```
e = p / (gamma - 1) + 0.5*rho*|u|^2.
```

The expressions may use the formula syntax from the [FunctionParser](https://www.dealii.org/current/doxygen/deal.II/classFunctionParser.html) class.

The spatial coordinates are defined as variables `x, y, z`, and time as the variable `t`.
)");

    prm.declare_entry(
        "constants", "", Patterns::Anything(),
        "Constants to use in the expression syntax. pi is defined by default.");
}

template <int dim>
std::unique_ptr<SpeciesFunc<dim>> SpeciesFunc<dim>::create_from_parameters(ParameterHandler& prm, double gas_gamma) {
    std::string str = prm.get("VariablesType");
    SpeciesFuncVariablesType variables_type;
    if (str == "Primitive") {
        variables_type = PRIMITIVE;
    } else {
        variables_type = CONSERVED;
    }

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

    std::unique_ptr<FunctionParser<dim>> func = std::make_unique<FunctionParser<dim>>(5);
    func->initialize(vnames, expression, constants, true);

    const bool is_zero = (expression == SPECIES_FUNC_DEFAULT);

    return std::make_unique<SpeciesFunc<dim>>(std::move(func), variables_type, gas_gamma, is_zero);
}


template <int dim>
template <typename Number>
Tensor<1, 5, Number> SpeciesFunc<dim>::evaluate(const Point<dim, Number>& p, double t) {
    func->set_time(t);
    return evaluate_function<dim, 5, Number>(*this, p);
}


template class SpeciesFunc<1>;
template class SpeciesFunc<2>;
template Tensor<1, 5, VectorizedArray<double>> SpeciesFunc<1>::evaluate<VectorizedArray<double>>(const Point<1, VectorizedArray<double>>& p, double t);

}  // namespace five_moment
}  // namespace warpii
