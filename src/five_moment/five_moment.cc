#include "five_moment.h"
#include "../simulation_input.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <memory>

#include "../wrapper.h"
#include "../common_params.h"

namespace warpii {
namespace five_moment {

void FiveMomentWrapper::declare_parameters(ParameterHandler &prm) {
    declare_n_dims(prm);
    declare_n_boundaries(prm);
    prm.declare_entry("n_species", "1", Patterns::Integer(),
            "The number of species in the simulation. "
            "Each species should be configured in its respective [Species](#Species_i) subsection.");
    GridWrapper::declare_parameters(prm);
}

std::unique_ptr<AbstractApp> FiveMomentWrapper::create_app(
        SimulationInput& input,
    std::shared_ptr<warpii::Extension> extension) {
    input.reparse(false);

    switch (input.prm.get_integer("n_dims")) {
        case 1: {
            std::shared_ptr<five_moment::Extension<1>> ext = 
                unwrap_extension<five_moment::Extension<1>>(extension);
            FiveMomentApp<1>::declare_parameters(input.prm, ext);
            input.reparse(true);
            return FiveMomentApp<1>::create_from_parameters(input, ext);
        }
        case 2: {
            std::shared_ptr<five_moment::Extension<2>> ext = 
                unwrap_extension<five_moment::Extension<2>>(extension);
            FiveMomentApp<2>::declare_parameters(input.prm, ext);
            input.reparse(true);
            return FiveMomentApp<2>::create_from_parameters(input, ext);
        }
                /*
        case 3: {
            FiveMomentApp<3>::declare_parameters(prm);
            prm.parse_input_from_string(input, "", false);
            return FiveMomentApp<3>::create_from_parameters(prm);
        }
        */
        default: {
            AssertThrow(false, ExcMessage("n_dims must be 1, 2, or 3"));
        }
    }
}

}  // namespace five_moment
}  // namespace warpii
