#include "fields.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

#include <string>

#include "bc_map.h"
#include "phmaxwell_func.h"
#include "../common_params.h"

using namespace dealii;

namespace warpii {

template <int dim>
void PHMaxwellFields<dim>::declare_parameters(ParameterHandler &prm,
                                              unsigned int n_boundaries) {
    prm.enter_subsection("PHMaxwellFields");

    prm.declare_entry("phmaxwell_gamma", "1.0", Patterns::Double(0.0));
    prm.declare_entry("phmaxwell_chi", "1.0", Patterns::Double(0.0));

    prm.enter_subsection("InitialCondition");
    PHMaxwellFunc<dim>::declare_parameters(prm);
    prm.leave_subsection();  // InitialCondition
                             
    prm.enter_subsection("GeneralSourceTerm");
    PHMaxwellFunc<dim>::declare_parameters(prm);
    prm.leave_subsection(); // GeneralSourceTerm

    for (unsigned int i = 0; i < n_boundaries; i++) {
        prm.enter_subsection("BoundaryCondition_" + std::to_string(i));

        declare_section_documentation(prm, 
                "PHMaxwell boundary condition specification for the boundary "
                "with `boundary_id == i`.", true);

        prm.declare_entry("Type", "Dirichlet",
                          Patterns::Selection("CopyOut|PerfectConductor|Dirichlet"));
        prm.enter_subsection("DirichletFunction");
        PHMaxwellFunc<dim>::declare_parameters(prm);
        prm.leave_subsection();

        prm.leave_subsection();  // BoundaryCondition_i
    }

    prm.leave_subsection();  // PHMaxwellFields
}

template <int dim>
std::shared_ptr<PHMaxwellFields<dim>>
PHMaxwellFields<dim>::create_from_parameters(SimulationInput &input,
                                             unsigned int n_boundaries,
                                             PlasmaNormalization plasma_norm) {
    ParameterHandler& prm = input.prm;

    prm.enter_subsection("PHMaxwellFields");

    double phmaxwell_gamma = prm.get_double("phmaxwell_gamma");
    double phmaxwell_chi = prm.get_double("phmaxwell_chi");

    prm.enter_subsection("InitialCondition");
    const auto ic = PHMaxwellFunc<dim>::create_from_parameters(input);
    prm.leave_subsection();

    prm.enter_subsection("GeneralSourceTerm");
    const auto source = PHMaxwellFunc<dim>::create_from_parameters(input);
    prm.leave_subsection();

    MaxwellBCMap<dim> bc_map;
    for (unsigned int i = 0; i < n_boundaries; i++) {
        prm.enter_subsection("BoundaryCondition_" + std::to_string(i));
        std::string bc_type = prm.get("Type");
        auto boundary_id = static_cast<types::boundary_id>(i);

        if (bc_type == "CopyOut") {
            bc_map.set_copy_out_boundary(boundary_id);
        } else if (bc_type == "PerfectConductor") {
            bc_map.set_perfect_conductor_boundary(boundary_id);
        } else if (bc_type == "Dirichlet") {
            prm.enter_subsection("DirichletFunction");
            auto func = PHMaxwellFunc<dim>::create_from_parameters(input);
            prm.leave_subsection();
            bc_map.set_dirichlet_boundary(boundary_id, std::move(func));
        }
        prm.leave_subsection();  // BoundaryConditions
    }

    prm.leave_subsection(); // PHMaxwellFields
    return std::make_shared<PHMaxwellFields<dim>>(
        phmaxwell_gamma, phmaxwell_chi, plasma_norm, ic, source, bc_map);
}

template class PHMaxwellFields<1>;
template class PHMaxwellFields<2>;

}  // namespace warpii
