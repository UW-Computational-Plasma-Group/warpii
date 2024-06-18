#pragma once
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace warpii {
    void declare_n_dims(ParameterHandler& prm);

    void declare_n_boundaries(ParameterHandler& prm);

    void declare_fe_degree(ParameterHandler& prm);

    void declare_t_end(ParameterHandler& prm);

    void declare_write_output(ParameterHandler& prm);

    void declare_n_writeout_frames(ParameterHandler& prm);
}
