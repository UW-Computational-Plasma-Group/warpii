#include <deal.II/lac/la_parallel_vector.h>
#include "solution_vec.h"

using namespace dealii;

namespace warpii {
namespace maxwell {
    void MaxwellSolutionVec::reinit(const MaxwellSolutionVec& other) {
        mesh_sol.reinit(other.mesh_sol);
    }
}  // namespace maxwell
}  // namespace warpii
