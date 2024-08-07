#include "src/warpii.h"
#include <gtest/gtest.h>

using namespace dealii;
using namespace warpii;

TEST(PlasmaWaveTest, LangmuirWave0D) {
    std::string input = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 10.0

set write_output = false

set fe_degree = 1

set fields_enabled = true

subsection geometry
    set left = 0.0
    set right = 1.0
    set nx = 1
end

subsection PHMaxwellFields
    subsection InitialCondition
        set components = 0; 1.0; 0; \
                         0; 0; 0; \
                         0; 0
    end
end

subsection Species_0
    set name = electron
    set charge = -1.0
    subsection InitialCondition
        set VariablesType = Primitive
        set components = 1.0; 0.0; 0.0; 0.0; 1.0
    end
end

)";
    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.run();
}
