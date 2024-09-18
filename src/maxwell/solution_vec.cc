#include <deal.II/lac/la_parallel_vector.h>
#include "solution_vec.h"

using namespace dealii;

namespace warpii {
namespace maxwell {
    void MaxwellSolutionVec::reinit(const MaxwellSolutionVec& other) {
        mesh_sol.reinit(other.mesh_sol);
    }

    void MaxwellSolutionVec::swap(MaxwellSolutionVec& other) {
        mesh_sol.swap(other.mesh_sol);
    }

    void MaxwellSolutionVec::sadd(double s, double a, MaxwellSolutionVec V) {
        mesh_sol.sadd(s, a, V.mesh_sol);
    }
}  // namespace maxwell
}  // namespace warpii
