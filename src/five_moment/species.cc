#include "species.h"
#include "species_func.h"
#include <deal.II/base/patterns.h>
#include "../common_params.h"

namespace warpii {
namespace five_moment {

template <int dim>
void Species<dim>::declare_parameters(ParameterHandler &prm,
                                      unsigned int n_boundaries) {

    declare_section_documentation(prm, 
            "Constants, boundary conditions and initial conditions for "
            "species i.", true);

    prm.declare_entry("name", "neutral",
                      Patterns::Selection("neutral|ion|electron"),
                      "The name of the species. There is no reason this couldn't be a free-form "
                      "field in the future.");
    prm.declare_entry("charge", "0.0", Patterns::Double(),
            "The nondimensional charge `Z` of the species. "
            "See [Normalization](#Normalization).");
    prm.declare_entry("mass", "1.0", Patterns::Double(0.0),
            "The nondimensional mass `A` of the species. "
            "See [Normalization](#Normalization).");
    {
        for (unsigned int i = 0; i < n_boundaries; i++) {
            prm.enter_subsection("BoundaryCondition_" + std::to_string(i));

            declare_section_documentation(prm, 
                    "Five-moment species boundary condition specification for the boundary "
                    "`boundary_id == i`.", true);

            prm.declare_entry("Type", "Wall", Patterns::Selection("Wall|Outflow|Inflow"),
                              R"(The type of boundary condition.
- `Wall`: an impermeable wall. Normal velocity is set to zero.
- `Outflow`: copy-out boundary condition, suitable for outflow boundaries.
- `Inflow`: a Dirichlet boundary condition suitable for inflow boundaries. See [InflowFunction](#InflowFunction).)");
            prm.enter_subsection("InflowFunction");
            SpeciesFunc<dim>::declare_parameters(prm);
            prm.leave_subsection(); // InflowFunction
            prm.leave_subsection();  // BoundaryCondition_i
        }
    }
    prm.enter_subsection("InitialCondition");
    SpeciesFunc<dim>::declare_parameters(prm);
    prm.leave_subsection();
}

template <int dim>
std::shared_ptr<Species<dim>> Species<dim>::create_from_parameters(
    ParameterHandler &prm, unsigned int n_boundaries, double gas_gamma) {
    std::string name = prm.get("name");
    double charge = prm.get_double("charge");
    double mass = prm.get_double("mass");
    auto bc_map = EulerBCMap<dim>();

    {
        for (unsigned int i = 0; i < n_boundaries; i++) {
            prm.enter_subsection("BoundaryCondition_" + std::to_string(i));

            std::string bc_type = prm.get("Type");
            auto boundary_id = static_cast<types::boundary_id>(i);
            if (bc_type == "Wall") {
                bc_map.set_wall_boundary(boundary_id);
            } else if (bc_type == "Outflow") {
                bc_map.set_supersonic_outflow_boundary(boundary_id);
            } else if (bc_type == "Inflow") {
                prm.enter_subsection("InflowFunction");
                auto inflow_func = SpeciesFunc<dim>::create_from_parameters(prm, gas_gamma);
                bc_map.set_inflow_boundary(boundary_id, std::move(inflow_func));
                prm.leave_subsection(); // InflowFunction
            }
            prm.leave_subsection(); // BoundaryCondition_i
        }
    }
    prm.enter_subsection("InitialCondition");
    std::unique_ptr<SpeciesFunc<dim>> initial_condition = SpeciesFunc<dim>::create_from_parameters(prm, gas_gamma);
    prm.leave_subsection();

    return std::make_shared<Species<dim>>(name, charge, mass, bc_map,
                                          std::move(initial_condition));
}

template class Species<1>;
template class Species<2>;

}  // namespace five_moment

}  // namespace warpii
