#pragma once

#include <deal.II/base/parameter_handler.h>
#include "app.h"
#include "simulation_input.h"

namespace warpii {

    using namespace dealii;

class ApplicationWrapper {
    public:
        virtual ~ApplicationWrapper() = default;

        virtual void declare_parameters(ParameterHandler& prm) = 0;

        virtual std::unique_ptr<AbstractApp> create_app(
                SimulationInput& simulation_input,
                std::shared_ptr<Extension> extension) = 0;
};

}
