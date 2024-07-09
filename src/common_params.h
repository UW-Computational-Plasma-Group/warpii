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

    /**
     * Declare a dummy parameter inside the current section. The dummy parameter
     * is named `section_documentation`, and its documentation will be associated
     * to the current section in the generated documentation.
     *
     * If `is_multisection` is true, multiple occurrences of this section
     * will be grouped together in the documentation under the heading "SectionName_i".
     */
    void declare_section_documentation(ParameterHandler& prm, 
            std::string documentation,
            bool is_multisection=false);
}
