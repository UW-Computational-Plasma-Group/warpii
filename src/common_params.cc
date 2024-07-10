#include "common_params.h"

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace warpii {
    void declare_n_dims(ParameterHandler& prm) {
        prm.declare_entry("n_dims", "1", Patterns::Integer(1, 3),
"The number of non-symmetry dimensions in the problem. "
"Note that for a 1- or 2-dimensional problem, vector field quantities will still "
"have 3 components.");
    }

    void declare_n_boundaries(ParameterHandler &prm) {
        prm.declare_entry("n_boundaries", "0", Patterns::Integer(),
"The number of distinct non-periodic boundary_ids in the domain. "
"This must match the state of the actual mesh as constructed from "
"a file or by populating a triangulation.");
    }

    void declare_t_end(ParameterHandler& prm) {
        prm.declare_entry("t_end", "0.0", Patterns::Double(0.0),
            "The end time of time-dependent simulations. Must be non-negative.");
    }

    void declare_fe_degree(ParameterHandler &prm) {
        prm.declare_entry("fe_degree", "2", Patterns::Integer(1, 6),
                "The degree of finite element shape functions to use. "
                "The expected order of convergence is one greater than this. "
                "I.e. if fe_degree == 2, then we use quadratic polynomials and "
                "can expect third order convergence.");
    }

    void declare_write_output(ParameterHandler& prm) {
        prm.declare_entry("write_output", "true", Patterns::Bool(),
                "Whether to write output files");
    }

    void declare_n_writeout_frames(ParameterHandler& prm) {
        prm.declare_entry("n_writeout_frames", "10", Patterns::Integer(0),
                "The number of frames, or snapshots, of the solution to write out. "
                "Frames are evenly spaced in time, and frame 0 is the initial condition. "
                "Setting this to 10 will write out frame 0 through frame 10, "
                "for a total of 11 snapshots.");
    }

    void declare_section_documentation(ParameterHandler& prm, std::string documentation, bool is_multisection) {
        prm.declare_entry("section_documentation", is_multisection ? "multisection" : "", Patterns::Anything(),
                documentation);
    }
}
